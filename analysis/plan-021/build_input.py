"""plan-021 §6.1 — 4 lever input builder + Frenet basis + LGBM extras.

Exports (§4.2):
  build_frenet_basis_3d, to_frenet, build_input_common, build_input_lgbm_extra,
  build_soft_label, ANCHORS_FRENET, DT, H, R_HIT, R_HIT_LOOSE, TAU_LOSS, TAU_CLS
"""
from __future__ import annotations

from typing import Callable

import numpy as np

DT = 0.040
H = 0.080
R_HIT = 0.01
R_HIT_LOOSE = 0.015
TAU_LOSS = 0.001
TAU_CLS = 0.001

# §3.4 — single Frenet-orthogonal codebook
ANCHORS_FRENET = np.array([
    (0.000, 0.000, 0.000),   # 0: origin
    (+0.005, 0.000, 0.000),  # 1: +t̂
    (-0.005, 0.000, 0.000),  # 2: -t̂
    (0.000, +0.005, 0.000),  # 3: +n̂
    (0.000, -0.005, 0.000),  # 4: -n̂
    (0.000, 0.000, +0.005),  # 5: +b̂
    (0.000, 0.000, -0.005),  # 6: -b̂
], dtype=np.float32)  # (7, 3)


# ── Frenet basis ───────────────────────────────────────────────────────


def build_frenet_basis_3d(x: np.ndarray, end_idx: int) -> np.ndarray:
    """§4.2.1 — x (N, T, 3), end_idx → R_wfn (N, 3, 3), columns = [t̂, n̂, b̂].

    Fallback (frenet_basis_degenerate):
      ‖v_last‖<1e-9 → R = I_3
      ‖a_⊥‖<1e-9 → 임의 n̂ via world-z (또는 world-x if |t̂·z|>0.99)
    """
    if x.shape[1] < 3 or end_idx < 2:
        raise ValueError("build_frenet_basis_3d requires T>=3 and end_idx>=2")
    N = x.shape[0]
    v_last = x[:, end_idx] - x[:, end_idx - 1]
    v_prev = x[:, end_idx - 1] - x[:, end_idx - 2]
    a = v_last - v_prev

    v_norm = np.linalg.norm(v_last, axis=1, keepdims=True)
    safe_v = (v_norm > 1e-9).squeeze(-1)

    t_hat = np.zeros((N, 3), dtype=np.float32)
    t_hat[safe_v] = (v_last[safe_v] / v_norm[safe_v]).astype(np.float32)
    t_hat[~safe_v] = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # fallback: world-x

    acc_par_scalar = np.sum(a * t_hat, axis=1, keepdims=True)
    acc_perp = a - acc_par_scalar * t_hat
    perp_norm = np.linalg.norm(acc_perp, axis=1, keepdims=True)
    safe_perp = (perp_norm > 1e-9).squeeze(-1)

    n_hat = np.zeros((N, 3), dtype=np.float32)
    n_hat[safe_perp] = (acc_perp[safe_perp] / perp_norm[safe_perp]).astype(np.float32)
    # fallback for ~safe_perp: world-z (또는 world-x if collinear)
    z_world = np.broadcast_to(np.array([0.0, 0.0, 1.0], dtype=np.float32), (N, 3)).copy()
    near_z = np.abs(np.sum(t_hat * z_world, axis=1)) > 0.99
    z_world[near_z] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    proj = z_world - np.sum(z_world * t_hat, axis=1, keepdims=True) * t_hat
    proj_norm = np.linalg.norm(proj, axis=1, keepdims=True)
    n_fallback = (proj / np.maximum(proj_norm, 1e-9)).astype(np.float32)
    n_hat[~safe_perp] = n_fallback[~safe_perp]

    b_hat = np.cross(t_hat, n_hat).astype(np.float32)

    # all-degenerate (no v) → R = I_3
    R = np.stack([t_hat, n_hat, b_hat], axis=-1)  # (N, 3, 3) columns=[t̂,n̂,b̂]
    R[~safe_v] = np.eye(3, dtype=np.float32)
    return R


def to_frenet(vec: np.ndarray, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """world → frenet: R^T @ (vec − origin). vec (N, 3), R (N, 3, 3), origin (N, 3)."""
    return np.einsum("nij,nj->ni", R.transpose(0, 2, 1), vec - origin).astype(np.float32)


# ── L1: Frenet trajectory (11 × 9) ─────────────────────────────────────


def _build_L1(x: np.ndarray, R_wfn: np.ndarray, origin: np.ndarray) -> np.ndarray:
    N, T, _ = x.shape
    L1 = np.zeros((N, T, 9), dtype=np.float32)
    Rt = R_wfn.transpose(0, 2, 1)
    for t in range(T):
        pos_w = x[:, t]
        v_w = x[:, t] - x[:, t - 1] if t >= 1 else np.zeros_like(pos_w)
        v_pw = x[:, t - 1] - x[:, t - 2] if t >= 2 else np.zeros_like(pos_w)
        a_w = (v_w - v_pw) if t >= 2 else np.zeros_like(pos_w)

        L1[:, t, 0:3] = np.einsum("nij,nj->ni", Rt, pos_w - origin)
        L1[:, t, 3:6] = np.einsum("nij,nj->ni", Rt, v_w)
        L1[:, t, 6:9] = np.einsum("nij,nj->ni", Rt, a_w)
    return L1


# ── L2/L4: F0 residual + soft hit sequences ────────────────────────────


def _build_L2_L4(
    x: np.ndarray,
    R_wfn: np.ndarray,
    f0_baseline_fn: Callable,
) -> tuple[np.ndarray, np.ndarray]:
    """7 past sub-window F0 잔차 (Frenet) + soft hit (1cm / 1.5cm)."""
    N = x.shape[0]
    L2 = np.zeros((N, 7, 3), dtype=np.float32)
    L4 = np.zeros((N, 7, 2), dtype=np.float32)
    Rt = R_wfn.transpose(0, 2, 1)
    for i, t in enumerate(range(4, 11)):
        sub_x = x[:, t - 4:t - 1, :]                       # (N, 3, 3)
        pred_t = f0_baseline_fn(sub_x, end_idx=2)           # (N, 3)
        actual_t = x[:, t]
        residual_w = pred_t - actual_t                       # (N, 3)
        L2[:, i] = np.einsum("nij,nj->ni", Rt, residual_w).astype(np.float32)
        d_t = np.linalg.norm(residual_w, axis=1)            # (N,)
        L4[:, i, 0] = _sigmoid((R_HIT - d_t) / TAU_LOSS)
        L4[:, i, 1] = _sigmoid((R_HIT_LOOSE - d_t) / TAU_LOSS)
    return L2.astype(np.float32), L4.astype(np.float32)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # stable sigmoid
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


# ── public: build_input_common ─────────────────────────────────────────


def build_input_common(X: np.ndarray, f0_baseline_fn: Callable) -> dict:
    """§4.2 + §6.1. X (N, 11, 3), f0_baseline_fn injected.
    returns {"L1": (N,11,9), "L2": (N,7,3), "L4": (N,7,2),
             "R_wfn": (N,3,3), "origin": (N,3) [= x[end_idx], for L1 invariance],
             "pred_F0_world": (N,3) [= F0 산출 80ms 미래 위치, anchor codebook 의 origin]}.

    v1.3 conceptual fix: anchor codebook 은 *F0 prediction* (= 80ms 미래) 주변 ±0.5cm 영역.
    L1 의 Frenet origin (= last observed) 와 anchor 의 origin (= F0_pred) 는 다른 reference.
    """
    end_idx = X.shape[1] - 1
    R_wfn = build_frenet_basis_3d(X, end_idx=end_idx)
    origin = X[:, end_idx].astype(np.float32)
    L1 = _build_L1(X, R_wfn, origin)
    L2, L4 = _build_L2_L4(X, R_wfn, f0_baseline_fn)
    pred_F0_world = f0_baseline_fn(X, end_idx=end_idx).astype(np.float32)
    return {
        "L1": L1, "L2": L2, "L4": L4,
        "R_wfn": R_wfn.astype(np.float32), "origin": origin,
        "pred_F0_world": pred_F0_world,
    }


# ── L5: macro statistic 9D (§6.1.4 self-contained) ────────────────────


def _macro_stat_9d(x: np.ndarray, end_idx: int) -> np.ndarray:
    start = max(0, end_idx - 5)
    pts = x[:, start:end_idx + 1, :]                                # (N, 6, 3)
    v = np.diff(pts, axis=1)                                          # (N, 5, 3)
    speeds = np.linalg.norm(v, axis=2)                                # (N, 5)
    current_speed = speeds[:, -1:] + 1e-9
    mean_speed = speeds.mean(axis=1, keepdims=True) + 1e-9
    path = speeds.sum(axis=1, keepdims=True)
    disp = np.linalg.norm(pts[:, -1] - pts[:, 0], axis=1, keepdims=True)
    straightness = disp / (path + 1e-9)
    speed_slope = (speeds[:, -1:] - speeds[:, :1]) / mean_speed
    speed_cv = speeds.std(axis=1, keepdims=True) / mean_speed
    v0, v1 = v[:, :-1], v[:, 1:]
    turn_cos = (v0 * v1).sum(axis=2) / (np.linalg.norm(v0, axis=2) * np.linalg.norm(v1, axis=2) + 1e-9)
    turn_accum = (1.0 - np.clip(turn_cos, -1, 1)).mean(axis=1, keepdims=True)
    turn_volatility = (1.0 - np.clip(turn_cos, -1, 1)).std(axis=1, keepdims=True)
    acc = np.diff(v, axis=1)                                          # (N, 4, 3)
    acc_norm = np.linalg.norm(acc, axis=2)
    accel_slope = (acc_norm[:, -1:] - acc_norm[:, :1]) / (acc_norm.mean(axis=1, keepdims=True) + 1e-9)
    linear_pred = x[:, end_idx - 1] + (x[:, end_idx - 1] - x[:, end_idx - 2])
    linear_resid = np.linalg.norm(x[:, end_idx] - linear_pred, axis=1, keepdims=True) / mean_speed
    jerk = np.diff(acc, axis=1)                                       # (N, 3, 3)
    jerk_vol = np.linalg.norm(jerk, axis=2).std(axis=1, keepdims=True) / mean_speed
    return np.concatenate([
        path / current_speed, straightness, speed_slope, speed_cv,
        turn_accum, accel_slope, turn_volatility, linear_resid, jerk_vol,
    ], axis=1).astype(np.float32)


# ── L6: EWMA (Frenet [p,v,a] × α ∈ {0.1, 0.3, 0.5}, last value) ──────


def _ewma_last(seq: np.ndarray, alpha: float) -> np.ndarray:
    """seq (N, T, D) → last EWMA value (N, D)."""
    s = seq[:, 0]
    for t in range(1, seq.shape[1]):
        s = alpha * seq[:, t] + (1.0 - alpha) * s
    return s.astype(np.float32)


def build_input_lgbm_extra(X: np.ndarray, L1: np.ndarray | None = None) -> np.ndarray:
    """LGBM 전용 36D = 9D macro stat + 27D EWMA. L1 (N,11,9) 미주입 시 재계산."""
    end_idx = X.shape[1] - 1
    macro = _macro_stat_9d(X, end_idx)                                # (N, 9)
    if L1 is None:
        R_wfn = build_frenet_basis_3d(X, end_idx=end_idx)
        origin = X[:, end_idx].astype(np.float32)
        L1 = _build_L1(X, R_wfn, origin)
    # EWMA on Frenet [p, v, a] 9D × 3 α = 27D
    ewma_list = [_ewma_last(L1, alpha=a) for a in (0.1, 0.3, 0.5)]
    ewma = np.concatenate(ewma_list, axis=1)                          # (N, 27)
    return np.concatenate([macro, ewma], axis=1).astype(np.float32)   # (N, 36)


# ── build_soft_label (classifier target) ───────────────────────────────


def build_soft_label(gt: np.ndarray, R_wfn: np.ndarray, pred_F0_world: np.ndarray) -> np.ndarray:
    """§4.2 v1.3 — gt (N, 3) world → soft prob (N, 7) over ANCHORS_FRENET.

    residual reference = pred_F0_world (= F0 의 80ms 미래 예측 위치). anchor codebook 은
    F0_pred 주변 Frenet ±0.5cm 영역이라 residual_true = (gt − pred_F0_world) 의 Frenet 분해.

    q_k = softmax(-‖ANCHORS_FRENET[k] − residual_true_frenet‖ / TAU_CLS).
    """
    residual_true_frenet = to_frenet(gt, R_wfn, pred_F0_world)         # (N, 3) — Frenet residual
    dist = np.linalg.norm(
        ANCHORS_FRENET[None, :, :] - residual_true_frenet[:, None, :], axis=-1
    )                                                                   # (N, 7)
    logits = -dist / TAU_CLS
    logits = logits - logits.max(axis=1, keepdims=True)                # stability
    ex = np.exp(logits)
    return (ex / ex.sum(axis=1, keepdims=True)).astype(np.float32)
