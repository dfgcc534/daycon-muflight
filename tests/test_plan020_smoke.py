"""plan-020 c7 — G0 smoke test (18 module import + F0 parity + per-candidate shape/finite).

§4.4 tests:
  - 18 모듈 import (AttributeError 0건)
  - F0 reproduce: 10000 train 위 hit@1cm ∈ [0.6315, 0.6325] (G1 사전 smoke, data 있을 때만)
  - 각 deterministic candidate: shape (N, 3), finite, edge case fallback
  - 각 NN candidate: forward pass shape OK
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PLAN020_DIR = REPO_ROOT / "analysis" / "plan-020"
sys.path.insert(0, str(REPO_ROOT))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, PLAN020_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def synthetic_traj():
    rng = np.random.default_rng(20260518)
    x = np.cumsum(rng.standard_normal((32, 11, 3)) * 0.01, axis=1)
    y = x[:, -1] + rng.standard_normal((32, 3)) * 0.005
    return x.astype(np.float64), y.astype(np.float64)


# ── §4.2 export symbols ───────────────────────────────────────────────


def test_module_imports():
    """All 4 plan-020 modules import without error."""
    bf = _load("baseline_f0")
    fd = _load("formula_deterministic")
    fnn = _load("formula_nn")
    cma_mod = _load("cma_es_fit")
    # No AttributeError on top-level symbol access:
    assert callable(bf.f0_baseline)
    assert callable(bf.f0_form_torch)
    assert isinstance(bf.R_HIT, float) and bf.R_HIT == 0.01
    assert isinstance(bf.R_HIT_LOOSE, float) and bf.R_HIT_LOOSE == 0.015
    assert len(fd.C01_TO_C14) == 14
    assert all(callable(v) for v in fd.C01_TO_C14.values())
    assert hasattr(fnn, "N01_MLPCoef") and hasattr(fnn, "N02_TCNCoef") and hasattr(fnn, "N05_MoE")
    assert callable(cma_mod.fit_candidate)


# ── F0 numpy ↔ torch parity ───────────────────────────────────────────


def test_f0_numpy_torch_parity(synthetic_traj):
    """coef=(D1,PAR,PERP) 입력 시 baseline_f0 (numpy) ↔ f0_form_torch (torch) bit-identical."""
    import torch
    bf = _load("baseline_f0")
    x, _ = synthetic_traj
    pred_np = bf.f0_baseline(x, end_idx=10)
    seq_feats = bf.build_seq_feats_3step(x, end_idx=10)
    coef = torch.tensor([[bf.D1, bf.PAR, bf.PERP]] * x.shape[0], dtype=torch.float64)
    pred_t = bf.f0_form_torch(torch.from_numpy(seq_feats), coef).numpy()
    diff = float(np.abs(pred_np - pred_t).max())
    assert diff < 1e-9, f"numpy/torch mirror diff = {diff}"


# ── 14 deterministic candidates: shape + finite ──────────────────────


def test_all_deterministic_shape_finite(synthetic_traj):
    fd = _load("formula_deterministic")
    x, _ = synthetic_traj
    bad = []
    for name, fn in fd.C01_TO_C14.items():
        try:
            pred = fn(x, end_idx=10)
        except Exception as exc:
            bad.append(f"{name}: raised {type(exc).__name__}: {exc!s:.80}")
            continue
        if pred.shape != (x.shape[0], 3):
            bad.append(f"{name}: shape {pred.shape} != (N, 3)")
            continue
        if not np.isfinite(pred).all():
            bad.append(f"{name}: NaN/Inf present")
    assert not bad, "deterministic candidates failed:\n" + "\n".join(bad)


# ── 3 NN candidates: forward shape ──────────────────────────────────


def test_all_nn_forward_shape(synthetic_traj):
    import torch
    bf = _load("baseline_f0")
    fnn = _load("formula_nn")
    x, _ = synthetic_traj
    # build (B, 11, 9) feats
    seq_11 = np.zeros((x.shape[0], 11, 9), dtype=np.float32)
    for t in range(11):
        seq_11[:, t, 0:3] = x[:, t]
        if t >= 1:
            seq_11[:, t, 3:6] = x[:, t] - x[:, t - 1]
        if t >= 2:
            seq_11[:, t, 6:9] = seq_11[:, t, 3:6] - (x[:, t - 1] - x[:, t - 2])
    seq_3 = seq_11[:, -3:, :]

    n1 = fnn.N01_MLPCoef()
    out_n1 = n1(torch.from_numpy(seq_3))
    assert out_n1.shape == (x.shape[0], 3) and torch.isfinite(out_n1).all()

    n2 = fnn.N02_TCNCoef()
    out_n2 = n2(torch.from_numpy(seq_11))
    assert out_n2.shape == (x.shape[0], 3) and torch.isfinite(out_n2).all()

    n5 = fnn.N05_MoE()
    expert = torch.from_numpy(np.random.default_rng(0).standard_normal((x.shape[0], 4, 3)).astype(np.float32))
    out_n5 = n5(torch.from_numpy(seq_11), expert)
    assert out_n5.shape == (x.shape[0], 3) and torch.isfinite(out_n5).all()


# ── pred_fn dispatchers ──────────────────────────────────────────────


def test_nn_pred_fn_dispatchers(synthetic_traj):
    import torch
    bf = _load("baseline_f0")
    fnn = _load("formula_nn")
    x, y = synthetic_traj
    seq_11 = np.zeros((x.shape[0], 11, 9), dtype=np.float32)
    for t in range(11):
        seq_11[:, t, 0:3] = x[:, t]
        if t >= 1:
            seq_11[:, t, 3:6] = x[:, t] - x[:, t - 1]
        if t >= 2:
            seq_11[:, t, 6:9] = seq_11[:, t, 3:6] - (x[:, t - 1] - x[:, t - 2])
    seq_3 = seq_11[:, -3:, :]

    pred_fn_n1 = fnn.make_pred_fn_n1(bf.f0_form_torch)
    pred = pred_fn_n1(fnn.N01_MLPCoef(), torch.from_numpy(seq_3), None)
    assert pred.shape == (x.shape[0], 3)

    pred_fn_n2 = fnn.make_pred_fn_n2(bf.f0_form_torch)
    pred = pred_fn_n2(fnn.N02_TCNCoef(), torch.from_numpy(seq_11), None)
    assert pred.shape == (x.shape[0], 3)

    pred_fn_n5 = fnn.make_pred_fn_n5()
    expert = torch.from_numpy(np.zeros((x.shape[0], 4, 3), dtype=np.float32))
    pred = pred_fn_n5(fnn.N05_MoE(), torch.from_numpy(seq_11), expert)
    assert pred.shape == (x.shape[0], 3)


# ── F0 reproduce on real data (G1 사전 smoke, data 있을 때만) ─────────


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "train").exists(),
    reason="train data not present — G1 actual reproduce는 c8 단계에서.",
)
def test_f0_reproduce_g1_preflight():
    """data/train 가 있으면 F0 hit@1cm ∈ [0.6315, 0.6325] sanity (G1 pre-check)."""
    from src.io import load_all_samples, load_labels
    bf = _load("baseline_f0")
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    assert ids == label_ids
    pred = bf.f0_baseline(X.astype(np.float64), end_idx=10)
    d = np.linalg.norm(pred - Y.astype(np.float64), axis=1)
    hit_1cm = float((d <= 0.01).mean())
    hit_1_5cm = float((d <= 0.015).mean())
    assert 0.6315 <= hit_1cm <= 0.6325, f"G1 drift: hit@1cm={hit_1cm:.4f}"
    assert 0.8028 <= hit_1_5cm <= 0.8038, f"G1 drift: hit@1.5cm={hit_1_5cm:.4f}"
