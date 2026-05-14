"""plan-014 c5 (G1) smoke test v4 — module 동작 + 재사용 끊김 4가지 assert (§5.4 spec v4).

v4 narrative: F0 = plan-006 frenet_par120_perp_neg020 frozen prior (plain function,
no nn.Parameter, requires_grad 개념 없음). Corrector 만 from-scratch + learnable.

Run:
    pytest tests/test_plan014_smoke.py -x -v
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.pb_0_6822 import plan014_paradigm as pp
from src.io import load_all_samples, load_labels


MODULE_PATH = REPO_ROOT / "src" / "pb_0_6822" / "plan014_paradigm.py"


# ── (b1) AST: no selector/ring_classifier/boundary/plan-006 numpy F0 import ─


def test_b1_reuse_cut_import():
    """plan014_paradigm.py 안에 forbidden module/function import 0."""
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    forbidden_modules = {"selector", "ring_classifier", "boundary"}
    forbidden_functions = {"f0_predict_frenet_par120_perp_neg020"}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                last_part = alias.name.rsplit(".", 1)[-1]
                if last_part in forbidden_modules:
                    found.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            last_part = mod.rsplit(".", 1)[-1]
            if last_part in forbidden_modules:
                found.append(f"from {mod} import ...")
            for alias in node.names:
                if alias.name in forbidden_modules or alias.name in forbidden_functions:
                    found.append(f"from {mod} import {alias.name}")
    assert not found, f"forbidden imports found: {found}"


# ── (b2) F0 frozen verify: reproduce hit@1cm ∈ [0.6270, 0.6370] + requires_grad=False ─


def test_b2_f0_frozen_verify():
    """F0 = plan-006 frenet_par120_perp_neg020 frozen (plain function, no Parameter).
    1. F0 산식 reproduce hit@1cm ∈ [0.6270, 0.6370] (G0 (a) carry)
    2. F0 function 의 attribute 가 nn.Parameter 가 아님 (= frozen, optimizer 가 학습 안 함)
    """
    f0 = pp.Plan014F0Function()
    # 1. F0 function 의 constants 가 nn.Parameter 가 아님 (plain float)
    assert isinstance(f0.d1, float), f"f0.d1 should be plain float (frozen), got {type(f0.d1)}"
    assert isinstance(f0.par, float), f"f0.par should be plain float (frozen), got {type(f0.par)}"
    assert isinstance(f0.perp, float), f"f0.perp should be plain float (frozen), got {type(f0.perp)}"
    assert not isinstance(f0, torch.nn.Module), f"Plan014F0Function should NOT be nn.Module (frozen)"

    # 2. reproduce hit@1cm on real data (skip if data missing)
    if not Path("data/train").exists():
        pytest.skip("data/train missing")
    ids, X = load_all_samples("train")
    label_ids, Y = load_labels()
    assert ids == label_ids
    X = X.astype(np.float32); Y = Y.astype(np.float32)
    F0_pred = f0(X)
    err = np.linalg.norm(F0_pred - Y, axis=1)
    hit1cm = float((err <= 0.01).mean())
    assert 0.6270 <= hit1cm <= 0.6370, f"F0 frozen reproduce hit@1cm={hit1cm:.4f} 가 [0.6270, 0.6370] 밖"


# ── (b3) anchor scale = 0.01 ± 1e-6 for E0a/E0b (E0c center=0) ───────────────


def test_b3_anchor_scale():
    a_abs = pp.compute_anchors_absolute(radius_m=0.01)
    a_fro = pp.compute_anchors_frenet_orthogonal(radius_m=0.01)

    norms_abs = np.linalg.norm(a_abs[1:], axis=-1)
    norms_fro = np.linalg.norm(a_fro[1:], axis=-1)
    assert np.allclose(norms_abs, 0.01, atol=1e-6), f"absolute non-center ‖·‖ ≠ 0.01: {norms_abs}"
    assert np.allclose(norms_fro, 0.01, atol=1e-6), f"frenet_orthogonal non-center ‖·‖ ≠ 0.01: {norms_fro}"

    # K-Means: synthetic residuals
    rng = np.random.default_rng(pp.DEFAULT_SEED)
    residuals_world = rng.normal(0, 0.008, (200, 3)).astype(np.float32)
    R = np.tile(np.eye(3, dtype=np.float32), (200, 1, 1))
    a_km = pp.compute_anchors_kmeans(residuals_world, R, fold_id=0, K=7)
    assert np.allclose(a_km[0], 0.0), f"K-Means anchor[0] should be center origin, got {a_km[0]}"


# ── (b4) soft label target w_k entropy ≥ 0.5 nat ─────────────────────────────


def test_b4_soft_label_entropy():
    """Target w_k Gaussian 분포의 sample-별 entropy 평균 ≥ 0.5 nat (학습 전 분석적 산출)."""
    torch.manual_seed(pp.DEFAULT_SEED)
    rng = np.random.default_rng(pp.DEFAULT_SEED)
    N = 256
    F0_pred = torch.from_numpy(rng.normal(0, 0.01, (N, 3)).astype(np.float32))
    Y = torch.from_numpy(rng.normal(0, 0.01, (N, 3)).astype(np.float32))
    anchors_world = torch.from_numpy(
        np.broadcast_to(pp.compute_anchors_absolute(radius_m=0.01)[None], (N, 7, 3)).copy()
    )
    w_k = pp.gaussian_soft_label(F0_pred, anchors_world, Y, sigma=0.01)
    H = -(w_k * torch.log(w_k + 1e-12)).sum(dim=1)
    mean_H = float(H.mean().item())
    assert mean_H >= 0.5, f"soft label entropy {mean_H:.3f} < 0.5 nat"


# ── (a) smoke train: 1-fold 1-epoch — val_hit_after >= initial_val_hit − 0.05 ─


@pytest.mark.skipif(not Path("data/train").exists(), reason="data/train missing")
def test_a_smoke_train():
    """1-fold 1-epoch smoke: no NaN + val_hit_after >= initial_val_hit − 0.05.

    initial_val_hit = model.eval() + Adam.step()=0 random-init forward + hybrid_predict.
    val_hit_after = identical path 1 epoch 후 measurement.
    F0 frozen — gradient 영향 없음.
    """
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    assert ids == label_ids
    sub = 500
    ids = ids[:sub]
    X = X[:sub].astype(np.float32)
    Y = Y[:sub].astype(np.float32)

    fold_of = np.array([pp.stable_hash_fold(sid) for sid in ids])
    val_mask = fold_of == 0
    train_mask = ~val_mask

    cfg = pp.TrainConfig(
        name="smoke", K=7, encoder_name="bigru", codebook="absolute",
        epochs=1, patience=1, batch_size=128, seed=pp.DEFAULT_SEED,
    )
    f0_function = pp.Plan014F0Function()
    res = pp.train_one_fold(
        cfg, fold_id=0,
        X_train=X[train_mask], Y_train=Y[train_mask],
        X_val=X[val_mask], Y_val=Y[val_mask],
        f0_function=f0_function,
    )
    assert np.isfinite(res["best_val_loss"]), "val_loss is NaN"
    assert res["best_val_hit"] >= res["initial_val_hit"] - 0.05, \
        (f"val_hit_after={res['best_val_hit']:.4f} < initial_val_hit={res['initial_val_hit']:.4f} − 0.05. "
         f"random-init variance margin 위배.")
