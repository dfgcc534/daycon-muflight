"""plan-020 baseline_f0 — plan-006 frenet_par120_perp_neg020 1:1 재구현 + torch mirror.

산식: F0 = p0 + 1.98 · v_last + 1.20 · acc_par_vec − 0.20 · acc_perp_vec
Hard evidence (plan-014 G0 preflight reproduce): hit@1cm = 0.6320, hit@1.5cm = 0.8033.

Exports:
  R_HIT, R_HIT_LOOSE      — hit radii (0.01, 0.015)
  D1, PAR, PERP           — F0 coefficients
  f0_baseline(x, end_idx) — numpy baseline reproduce (deterministic, no learning)
  f0_form_torch(seq_feats, coef) — torch mirror for NN coef → 최종 예측 (gradient path 유지)
"""
from __future__ import annotations

import numpy as np
import torch

R_HIT: float = 0.01
R_HIT_LOOSE: float = 0.015
D1: float = 1.98
PAR: float = 1.20
PERP: float = -0.20


def f0_baseline(x: np.ndarray, end_idx: int) -> np.ndarray:
    """numpy baseline. x shape (N, T, 3), end_idx = T-1. returns (N, 3)."""
    p0 = x[:, end_idx]
    v_last = x[:, end_idx] - x[:, end_idx - 1]
    v_prev = x[:, end_idx - 1] - x[:, end_idx - 2]
    acc = v_last - v_prev

    speed = np.linalg.norm(v_last, axis=1, keepdims=True)
    tangent = v_last / (speed + 1e-9)
    acc_par_scalar = np.sum(acc * tangent, axis=1, keepdims=True)
    acc_par_vec = acc_par_scalar * tangent
    acc_perp_vec = acc - acc_par_vec

    return p0 + D1 * v_last + PAR * acc_par_vec + PERP * acc_perp_vec


def f0_form_torch(seq_feats: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """torch mirror — NN coef → 최종 예측 gradient path 보장 (§4.2/§4.3 plan-020).

    seq_feats shape (B, 3, 9) — last 3 timesteps × 9D = [px,py,pz, vx,vy,vz, ax,ay,az]
        per timestep. 단위 = displacement (Δt 분할 없음, baseline_f0 와 통일).
        timestep order: [end_idx-2, end_idx-1, end_idx].
    coef shape (B, 3) = (d1, par, perp). returns (B, 3).
    coef = (1.98, 1.20, -0.20) + identical seq_feats 시 f0_baseline 와 ±1e-6 일치 (smoke).
    """
    p0 = seq_feats[:, 2, 0:3]
    v_last = seq_feats[:, 2, 3:6]
    v_prev = seq_feats[:, 1, 3:6]
    acc = v_last - v_prev

    speed = v_last.norm(dim=1, keepdim=True)
    tangent = v_last / (speed + 1e-9)
    acc_par_s = (acc * tangent).sum(dim=1, keepdim=True)
    acc_par_vec = acc_par_s * tangent
    acc_perp_vec = acc - acc_par_vec

    d1, par, perp = coef[:, 0:1], coef[:, 1:2], coef[:, 2:3]
    return p0 + d1 * v_last + par * acc_par_vec + perp * acc_perp_vec


def build_seq_feats_3step(x: np.ndarray | torch.Tensor, end_idx: int) -> np.ndarray | torch.Tensor:
    """Helper: x shape (N, T, 3) → seq_feats shape (N, 3, 9) for f0_form_torch.
    9D per timestep = [px,py,pz, vx,vy,vz, ax,ay,az] displacement units (Δt 분할 없음).
    timesteps: [end_idx-2, end_idx-1, end_idx]. numpy/torch dispatch by input type.
    """
    is_torch = isinstance(x, torch.Tensor)
    stack = torch.stack if is_torch else np.stack

    feats = []
    for t in (end_idx - 2, end_idx - 1, end_idx):
        p = x[:, t]
        v = x[:, t] - x[:, t - 1]
        a = (x[:, t] - x[:, t - 1]) - (x[:, t - 1] - x[:, t - 2])
        feats.append(stack([p, v, a], axis=-2).reshape(x.shape[0], 9) if not is_torch else torch.cat([p, v, a], dim=-1))
    return stack(feats, axis=1) if not is_torch else torch.stack(feats, dim=1)
