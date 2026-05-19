"""plan-023 §4.5 — smoke tests (7 items).

Run: pytest tests/test_plan023_smoke.py -v
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

_PLAN023 = REPO / "analysis" / "plan-023"
_PLAN022 = REPO / "analysis" / "plan-022"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Lazy module loads (T1 verifies import)
@pytest.fixture(scope="module")
def mods():
    # plan-023 modules via sys.path insert pattern (디렉토리 hyphen 회피)
    anchors_largeN = _load("anchors_largeN_t", _PLAN023 / "anchors_largeN.py")
    run_oof_largeN = _load("run_oof_largeN_t", _PLAN023 / "run_oof_largeN.py")
    # plan-022 selector_only_model (carried by run_oof_largeN via importlib)
    som = run_oof_largeN.p022_run.som  # 통해서 접근 (run_oof.py 안 som module reference)
    return {
        "anchors_largeN": anchors_largeN,
        "run_oof_largeN": run_oof_largeN,
        "som": som,
    }


# ── T1: import (ImportError / AttributeError 0건) ────────────────────


def test_t1_import(mods):
    a = mods["anchors_largeN"]
    r = mods["run_oof_largeN"]
    for sym in ("ANCHORS_B1", "ANCHORS_B2", "ANCHORS_B3", "ANCHORS_B4",
                "LAYOUT_NAMES_B"):
        assert hasattr(a, sym), f"anchors_largeN.{sym} missing"
    for sym in ("run_sub_exp_largeN", "run_sweep_largeN", "p022_run",
                "anchors_mod", "TAU_SCAN"):
        assert hasattr(r, sym), f"run_oof_largeN.{sym} missing"


# ── T2: 4 layout dtype + norm bound + shape ──────────────────────────


def test_t2_layout_invariants(mods):
    a = mods["anchors_largeN"]
    expected = {
        "B1_dodeca20":     20,
        "B2_trunc_octa24": 24,
        "B3_icosidodec30": 30,
        "B4_fib50":        50,
    }
    for name, K_expected in expected.items():
        arr = a.LAYOUT_NAMES_B[name]
        assert arr.dtype == np.float32, f"{name}: dtype {arr.dtype} != float32"
        assert arr.shape == (K_expected, 3), \
            f"{name}: shape {arr.shape} != ({K_expected}, 3)"
        norms = np.linalg.norm(arr, axis=1)
        assert norms.max() <= 0.005 + 1e-7, \
            f"{name}: max ‖a‖ = {norms.max():.7f} > 0.005 + 1e-7"
        assert np.isfinite(arr).all(), f"{name}: non-finite"


# ── T3: single-shell std + unique vertex 검증 ─────────────────────────


def test_t3_single_shell_and_unique(mods):
    a = mods["anchors_largeN"]
    for name, arr in a.LAYOUT_NAMES_B.items():
        norms = np.linalg.norm(arr, axis=1)
        assert norms.std() <= 1e-6, \
            f"{name}: norm std {norms.std():.2e} > 1e-6 (single-shell violated)"
        # unique vertex 검증 (cyclic-perm duplicate bug 차단)
        K_expected = arr.shape[0]
        n_unique = np.unique(arr, axis=0).shape[0]
        assert n_unique == K_expected, \
            f"{name}: unique vertex {n_unique} != {K_expected} (duplicate bug)"


# ── T4: LAYOUT_NAMES_B dict 4 entry + key ────────────────────────────


def test_t4_layout_names_keys(mods):
    a = mods["anchors_largeN"]
    expected_keys = {"B1_dodeca20", "B2_trunc_octa24", "B3_icosidodec30", "B4_fib50"}
    assert set(a.LAYOUT_NAMES_B.keys()) == expected_keys
    assert len(a.LAYOUT_NAMES_B) == 4


# ── T5: plan-022 LgbmSelectorOnly K=50 fit/predict (max K) + soft label ──


def test_t5_selector_K50_reuse_smoke(mods):
    s = mods["som"]
    rng = np.random.default_rng(20260519)
    N, K = 200, 50          # K=50 (B4 fib50, plan-023 max)
    X = rng.standard_normal((N, 170)).astype(np.float32)
    q = rng.dirichlet(alpha=np.ones(K), size=N).astype(np.float32)
    model = s.LgbmSelectorOnly(K=K).fit(X, q)
    probs = model.predict(X[:100])
    assert probs.shape == (100, K)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(probs).all()


# ── T6: build_soft_label_with_tau on B1~B4 anchors × 3 τ ─────────────


def test_t6_soft_label_4layout_3tau(mods):
    s = mods["som"]
    a = mods["anchors_largeN"]
    rng = np.random.default_rng(20260519)
    N = 50
    gt = rng.standard_normal((N, 3)) * 0.01
    pred_F0 = rng.standard_normal((N, 3)) * 0.01
    R_wfn = np.tile(np.eye(3, dtype=np.float64)[None], (N, 1, 1))
    for layout_name, anchors in a.LAYOUT_NAMES_B.items():
        K = anchors.shape[0]
        for tau in [0.001, 0.003, 0.005]:
            q = s.build_soft_label_with_tau(gt, R_wfn, pred_F0, anchors, tau)
            assert q.shape == (N, K), \
                f"{layout_name} τ={tau}: shape {q.shape} != ({N}, {K})"
            assert np.allclose(q.sum(axis=1), 1.0, atol=1e-5), \
                f"{layout_name} τ={tau}: row sum != 1"
            assert np.isfinite(q).all(), f"{layout_name} τ={tau}: non-finite"


# ── T7: F0 reproduce sanity carry + samples-per-class lower-bound ───


def test_t7_f0_carry_and_samples_per_class(mods):
    baseline_path = REPO / "analysis" / "plan-020" / "baseline_oof.json"
    if not baseline_path.exists():
        pytest.skip("plan-020 baseline_oof.json 부재 (carry 불가)")
    baseline = json.loads(baseline_path.read_text())
    f0 = baseline["f0_baseline"]
    assert 0.6315 <= f0["hit_1cm_5fold_concat"] <= 0.6325
    assert 0.8028 <= f0["hit_1.5cm_5fold_concat"] <= 0.8038

    # samples-per-class lower-bound 검증 (warn-only, K=50 = 10000/50 = 200 floor)
    p022_carry = _PLAN022 / "baseline_carry.json"
    if p022_carry.exists():
        carry = json.loads(p022_carry.read_text())
        N = carry["n_samples"]
        for K in (20, 24, 30, 50):
            samples_per_class = N / K
            # warn threshold 200, 본 plan 의 K=50 cell 시 samples_per_class ≈ 200
            assert samples_per_class >= 200 - 1, \
                f"K={K}: samples/class {samples_per_class:.0f} < 199 (warn floor)"
