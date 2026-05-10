"""tests/test_ema_extrapolate.py — linear_extrap (B001 비트동등) + ema_extrapolate.

Per plan-003 §4.6.
"""
from __future__ import annotations

import numpy as np

from src.baselines.linear_extrapolate import ema_extrapolate, linear_extrap
from src.baselines.window_polyfit import predict
from src.io import TIMESTEPS_MS


def test_linear_extrap_matches_b001_bitwise():
    """linear_extrap ↔ window_polyfit.predict(window=2, degree=1) at t=80ms."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 11, 3))
    a = linear_extrap(X)
    b = predict(X, window=2, degree=1, t_target=80, timesteps=TIMESTEPS_MS)
    assert np.max(np.abs(a - b)) < 1e-9


def test_linear_extrap_shape():
    X = np.zeros((7, 11, 3))
    out = linear_extrap(X)
    assert out.shape == (7, 3)


def test_ema_finite_output():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 11, 3))
    out = ema_extrapolate(X, alpha=0.5)
    assert np.isfinite(out).all()
    assert out.shape == (20, 3)


def test_ema_linear_motion_equals_linear():
    """Constant velocity → ema = linear (any alpha)."""
    t = np.arange(11)[None, :, None].astype(np.float64)
    v0 = np.array([1.0, 2.0, -0.5])[None, None, :]
    X = (t * v0) * 0.04
    for alpha in (0.1, 0.5, 0.9):
        e = ema_extrapolate(X, alpha=alpha)
        l = linear_extrap(X)
        np.testing.assert_allclose(e, l, atol=1e-12)


def test_ema_alpha_edge_cases():
    """alpha=1 → uniform mean of velocities; alpha→0 → only last velocity (linear)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 11, 3))
    out_a1 = ema_extrapolate(X, alpha=1.0)
    out_lin = linear_extrap(X)
    out_small = ema_extrapolate(X, alpha=1e-6)
    assert np.isfinite(out_a1).all()
    assert np.isfinite(out_small).all()
    # alpha→0 weights peak at most-recent velocity (= linear). At alpha=1e-6,
    # second-most-recent has weight ≈ 1e-6 → expected rel err O(1e-6).
    np.testing.assert_allclose(out_small, out_lin, atol=1e-4)
