"""plan-017 Voxel CE head — 5×5×5 voxel classification corrector.

Spec: plan-017-low-cost-stage1.md §6.2.

- VOXEL_DEPTH = 5 (axis 별 levels [-2, -1, 0, 1, 2] × VOXEL_WIDTH = [-0.02, -0.01, 0, 0.01, 0.02] m)
- VOXEL_WIDTH = 0.01 m (1cm = hit threshold)
- VOXEL_TOTAL = 125
- 좌표 system: F0_pred 중심 relative offset.

재사용 (변경 X, import only):
- plan014_paradigm.Plan014BiGRUEncoder (input_dim=9, hidden=128) — encoder.
- plan014_paradigm._boundary_weight (np.ndarray) — sample-wise weight.
- plan014_paradigm.Plan014F0Function — numpy F0 prior (gradient X).
- plan014_paradigm.stable_hash_fold — 5-fold split.
- plan014_paradigm.make_seq_features — 9D feature 산출.
- plan014_paradigm.TrainConfig — train cfg dataclass.
"""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pb_0_6822 import plan014_paradigm as pp


# ─────────────────────────────────────────────────────────────────────────
# Constants (plan-017 §6.2.A)
# ─────────────────────────────────────────────────────────────────────────

VOXEL_WIDTH: float = 0.01   # 1cm = hit threshold
VOXEL_DEPTH: int = 7        # 7 levels per axis (-3, -2, -1, 0, 1, 2, 3) = ±3cm range
VOXEL_TOTAL: int = 343      # 7³
HALF: int = (VOXEL_DEPTH - 1) // 2   # 3 (voxel-index half-range)


# ─────────────────────────────────────────────────────────────────────────
# Voxel grid utilities
# ─────────────────────────────────────────────────────────────────────────

def voxel_grid_centers_np() -> np.ndarray:
    """Returns (343, 3) array of voxel center offsets relative to F0_pred.

    voxel_idx = (ix + HALF) * D² + (iy + HALF) * D + (iz + HALF), ix/iy/iz ∈ {-HALF..HALF}.
    HALF=3, D=7. Order: x outermost, z innermost.
    """
    levels = np.arange(-HALF, HALF + 1, dtype=np.float64) * VOXEL_WIDTH   # [-0.03..0.03]
    centers = np.zeros((VOXEL_TOTAL, 3), dtype=np.float64)
    D = VOXEL_DEPTH
    for ix in range(D):
        for iy in range(D):
            for iz in range(D):
                idx = ix * D * D + iy * D + iz
                centers[idx] = [levels[ix], levels[iy], levels[iz]]
    return centers


_VOXEL_CENTERS_NP: np.ndarray = voxel_grid_centers_np()   # (125, 3)


def y_to_voxel_idx(y: np.ndarray, f0_pred: np.ndarray) -> np.ndarray:
    """y (N, 3), f0_pred (N, 3) numpy. Returns (N,) int64 ∈ [0, VOXEL_TOTAL).

    Clamp 후 base-D index. Out-of-window samples → nearest edge voxel.
    """
    offset = y - f0_pred
    voxel_ijk = np.round(offset / VOXEL_WIDTH).astype(np.int64)
    voxel_ijk = np.clip(voxel_ijk, -HALF, HALF)
    D = VOXEL_DEPTH
    voxel_idx = (voxel_ijk[:, 0] + HALF) * D * D + (voxel_ijk[:, 1] + HALF) * D + (voxel_ijk[:, 2] + HALF)
    return voxel_idx


def voxel_idx_to_offset_np(idx: np.ndarray) -> np.ndarray:
    """idx (N,) int → (N, 3) offset numpy."""
    return _VOXEL_CENTERS_NP[idx]


def voxel_idx_to_offset_torch(idx: torch.Tensor, device: torch.device | str) -> torch.Tensor:
    """idx (N,) torch.int → (N, 3) torch.float32 on device."""
    centers = torch.as_tensor(_VOXEL_CENTERS_NP, dtype=torch.float32, device=device)
    return centers[idx]


# ─────────────────────────────────────────────────────────────────────────
# Voxel CE head
# ─────────────────────────────────────────────────────────────────────────

class Plan017VoxelCEHead(nn.Module):
    """Linear head: encoder_out (B, encoder_out_dim) → logits (B, 125)."""

    def __init__(self, encoder_out_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(encoder_out_dim, VOXEL_TOTAL)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


# ─────────────────────────────────────────────────────────────────────────
# Loss + forward predict
# ─────────────────────────────────────────────────────────────────────────

def voxel_ce_loss(logits: torch.Tensor, y: torch.Tensor, f0_pred: torch.Tensor,
                  sample_weight: torch.Tensor | None = None) -> torch.Tensor:
    """logits (B, 125), y (B, 3) torch, f0_pred (B, 3) torch. Returns scalar.

    Torch-native (device 위 round/clamp) — CPU↔GPU round-trip 없음.
    sample_weight (B,) torch on logits.device, requires_grad=False, dtype=float32.
    """
    offset = y - f0_pred                                          # (B, 3) on logits.device
    voxel_ijk = torch.round(offset / VOXEL_WIDTH).clamp(-HALF, HALF).long()
    D = VOXEL_DEPTH
    voxel_idx = ((voxel_ijk[:, 0] + HALF) * D * D
               + (voxel_ijk[:, 1] + HALF) * D
               + (voxel_ijk[:, 2] + HALF))                          # (B,) torch.int64
    ce_per_sample = F.cross_entropy(logits, voxel_idx, reduction="none")
    if sample_weight is not None:
        ce_per_sample = ce_per_sample * sample_weight
    return ce_per_sample.mean()


def hybrid_predict_voxel(seq: torch.Tensor,
                         encoder: pp.Plan014BiGRUEncoder,
                         voxel_head: Plan017VoxelCEHead,
                         f0_pred_detached: torch.Tensor) -> torch.Tensor:
    """Returns hybrid_pred (B, 3) in world frame. f0_pred_detached = caller-side detached."""
    h = encoder(seq)                                              # (B, 256)
    logits = voxel_head(h)                                        # (B, 125)
    argmax_idx = logits.argmax(dim=1)                             # (B,)
    offset = voxel_idx_to_offset_torch(argmax_idx, device=logits.device)   # (B, 3)
    return f0_pred_detached + offset


# ─────────────────────────────────────────────────────────────────────────
# Train one fold (voxel variant)
# ─────────────────────────────────────────────────────────────────────────

def train_one_fold_voxel(
    config: pp.TrainConfig,
    fold_id: int,
    X_train: np.ndarray, Y_train: np.ndarray,
    X_val: np.ndarray, Y_val: np.ndarray,
    f0_function: pp.Plan014F0Function,
    X_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """plan-016 G1 train_one_fold 의 voxel CE variant.

    재사용 (gradient X): F0 numpy + BiGRU encoder.
    신규: Plan017VoxelCEHead + voxel_ce_loss + hybrid_predict_voxel.
    monitor=val_hit (val 위 hybrid_predict_voxel 후 hit@1cm).
    boundary_weight = plan-014 carry (sample-wise weight on CE).
    """
    cfg = config
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    # Sequence features (plan-014 carry, 9D)
    seq_train = pp.make_seq_features(X_train, end_idx=cfg.end_idx)
    seq_val = pp.make_seq_features(X_val, end_idx=cfg.end_idx)

    # F0 frozen numpy
    F0_train_np = f0_function(X_train)
    F0_val_np = f0_function(X_val)

    # Boundary weight (plan-014 E6 carry)
    sw_train_np = pp._boundary_weight(F0_train_np, Y_train) if cfg.boundary_weight_on else None

    # To torch
    seq_train_t = torch.from_numpy(seq_train).to(device)
    seq_val_t = torch.from_numpy(seq_val).to(device)
    Y_train_t = torch.from_numpy(Y_train.astype(np.float32)).to(device)
    Y_val_t = torch.from_numpy(Y_val.astype(np.float32)).to(device)
    F0_train_t = torch.from_numpy(F0_train_np.astype(np.float32)).to(device)
    F0_val_t = torch.from_numpy(F0_val_np.astype(np.float32)).to(device)
    sw_t = torch.from_numpy(sw_train_np.astype(np.float32)).to(device) if sw_train_np is not None else None

    # Model
    encoder = pp.Plan014BiGRUEncoder(input_dim=9, hidden=128, num_layers=2, dropout=0.1).to(device)
    voxel_head = Plan017VoxelCEHead(encoder_out_dim=encoder.out_dim).to(device)
    params = list(encoder.parameters()) + list(voxel_head.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    n_train = X_train.shape[0]

    # Initial val_hit
    encoder.eval(); voxel_head.eval()
    with torch.no_grad():
        init_pred = hybrid_predict_voxel(seq_val_t, encoder, voxel_head, F0_val_t)
        initial_val_hit = pp._hit_rate_torch(init_pred, Y_val_t, threshold_m=0.01)

    best_val_hit = -1.0
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience_left = cfg.patience
    epoch_log: list[dict] = []

    for epoch in range(cfg.epochs):
        encoder.train(); voxel_head.train()
        perm = torch.randperm(n_train, device=device)
        losses = []
        for i in range(0, n_train, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            seq_b = seq_train_t[idx]
            y_b = Y_train_t[idx]
            f0_b = F0_train_t[idx]
            sw_b = sw_t[idx] if sw_t is not None else None

            opt.zero_grad()
            h = encoder(seq_b)
            logits = voxel_head(h)
            loss = voxel_ce_loss(logits, y_b, f0_b, sample_weight=sw_b)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        train_loss = float(np.mean(losses))

        # Validation
        encoder.eval(); voxel_head.eval()
        with torch.no_grad():
            h_val = encoder(seq_val_t)
            logits_val = voxel_head(h_val)
            val_loss = voxel_ce_loss(logits_val, Y_val_t, F0_val_t)
            argmax_idx = logits_val.argmax(dim=1)
            offset = voxel_idx_to_offset_torch(argmax_idx, device=device)
            val_pred = F0_val_t + offset
            val_hit = pp._hit_rate_torch(val_pred, Y_val_t, threshold_m=0.01)

        val_loss_f = float(val_loss.item())
        epoch_log.append({"epoch": epoch + 1, "train_loss": train_loss,
                          "val_loss": val_loss_f, "val_hit": val_hit})

        # monitor=val_hit
        if val_hit > best_val_hit:
            best_val_loss = val_loss_f
            best_val_hit = val_hit
            best_state = {
                "encoder": {k: v.detach().clone() for k, v in encoder.state_dict().items()},
                "voxel_head": {k: v.detach().clone() for k, v in voxel_head.state_dict().items()},
            }
            best_epoch = epoch + 1
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # Restore best
    if best_state is not None:
        encoder.load_state_dict(best_state["encoder"])
        voxel_head.load_state_dict(best_state["voxel_head"])
    encoder.eval(); voxel_head.eval()

    with torch.no_grad():
        oof_pred = hybrid_predict_voxel(seq_val_t, encoder, voxel_head, F0_val_t).cpu().numpy()

        test_pred = None
        if X_test is not None:
            seq_test = pp.make_seq_features(X_test, end_idx=cfg.end_idx)
            F0_test_np = f0_function(X_test)
            seq_test_t = torch.from_numpy(seq_test).to(device)
            F0_test_t = torch.from_numpy(F0_test_np.astype(np.float32)).to(device)
            test_pred = hybrid_predict_voxel(seq_test_t, encoder, voxel_head, F0_test_t).cpu().numpy()

    result = {
        "fold_id": fold_id,
        "n_train": int(n_train),
        "n_val": int(X_val.shape[0]),
        "best_val_loss": best_val_loss,
        "best_val_hit": best_val_hit,
        "best_epoch": best_epoch,
        "initial_val_hit": initial_val_hit,
        "oof_pred": oof_pred,
        "epoch_log": epoch_log,
    }
    if test_pred is not None:
        result["test_pred"] = test_pred
    return result


# ─────────────────────────────────────────────────────────────────────────
# Multi-seed × multi-fold (voxel variant)
# ─────────────────────────────────────────────────────────────────────────

def run_multiseed_kfold_voxel(
    ids_train: list[str],
    X_train: np.ndarray,
    Y_train: np.ndarray,
    ids_test: list[str],
    X_test: np.ndarray,
    config_base: pp.TrainConfig,
    seeds: list[int],
    f0_function: pp.Plan014F0Function | None = None,
    progress_cb=None,
) -> dict[str, Any]:
    """plan-016 run_multiseed_kfold 의 voxel variant.

    OOF aggregation: 좌표 mean over seeds → 5-fold concat → hit@1cm (plan-016 §5.2 carry).
    Test ensemble: 25 model 의 test prediction 좌표 mean.
    """
    if f0_function is None:
        f0_function = pp.Plan014F0Function()

    fold_of = np.array([pp.stable_hash_fold(s) for s in ids_train])
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    n_seeds = len(seeds)
    n_folds = pp.N_FOLDS

    oof_per_seed = np.full((n_seeds, N_train, 3), np.nan, dtype=np.float32)
    test_per_seed_fold = np.full((n_seeds, n_folds, N_test, 3), np.nan, dtype=np.float32)
    fold_results = []

    for si, seed in enumerate(seeds):
        cfg_seed = replace(config_base, seed=seed)
        for f in range(n_folds):
            train_mask = fold_of != f
            val_mask = fold_of == f
            t0 = time.time()
            res = train_one_fold_voxel(
                cfg_seed, fold_id=f,
                X_train=X_train[train_mask], Y_train=Y_train[train_mask],
                X_val=X_train[val_mask], Y_val=Y_train[val_mask],
                f0_function=f0_function,
                X_test=X_test,
            )
            elapsed = time.time() - t0
            oof_per_seed[si, val_mask, :] = res["oof_pred"]
            test_per_seed_fold[si, f, :, :] = res["test_pred"]
            entry = {
                "seed": seed, "fold": f,
                "best_val_hit": res["best_val_hit"],
                "best_val_loss": res["best_val_loss"],
                "best_epoch": res["best_epoch"],
                "elapsed_seconds": elapsed,
            }
            fold_results.append(entry)
            if progress_cb is not None:
                progress_cb(si, f, seed, res, elapsed)

    # Per-seed concat OOF hit
    per_seed_oof_hit = []
    for si in range(n_seeds):
        oof_si = oof_per_seed[si]
        completed = ~np.isnan(oof_si).any(axis=1)
        err = np.linalg.norm(oof_si[completed] - Y_train[completed], axis=-1)
        per_seed_oof_hit.append(float((err <= 0.01).mean()))

    # Coord mean over seeds → 5-fold concat OOF
    oof_pred_all = np.nanmean(oof_per_seed, axis=0).astype(np.float32)
    completed_mask = ~np.isnan(oof_pred_all).any(axis=1)
    overall_oof_hit = float((np.linalg.norm(
        oof_pred_all[completed_mask] - Y_train[completed_mask], axis=-1) <= 0.01).mean())

    fold_oof_hit_per_fold = {}
    for f in range(n_folds):
        m = (fold_of == f) & completed_mask
        if m.sum() > 0:
            err = np.linalg.norm(oof_pred_all[m] - Y_train[m], axis=-1)
            fold_oof_hit_per_fold[int(f)] = float((err <= 0.01).mean())

    test_pred_all = test_per_seed_fold.reshape(n_seeds * n_folds, N_test, 3).mean(axis=0).astype(np.float32)

    return {
        "overall_oof_hit_1cm": overall_oof_hit,
        "oof_pred_all": oof_pred_all,
        "test_pred": test_pred_all,
        "fold_results": fold_results,
        "per_seed_oof_hit_1cm": per_seed_oof_hit,
        "fold_oof_hit_per_fold": fold_oof_hit_per_fold,
        "seeds": list(seeds),
        "n_seeds": n_seeds,
        "n_folds": n_folds,
    }
