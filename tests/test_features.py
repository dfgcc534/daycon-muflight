"""tests/test_features.py — physics + oscillation feature unit tests.

Per plan-003 §4.6.
"""
from __future__ import annotations

import numpy as np

from src.features.oscillation import wingbeat_fft
from src.features.physics import acceleration, curvature, jerk, velocity
from src.training.train_residual import (
    make_feature_fn,
    physics_feature,
    relative_coords_feature,
    wingbeat_feature,
)


# ---------- physics ----------

def _linear_X(v: tuple[float, float, float], dt_sec: float = 0.04, T: int = 11):
    t = np.arange(T)[None, :, None].astype(np.float64)
    return (t * np.array(v)[None, None, :]) * dt_sec


def test_velocity_constant_for_linear_motion():
    v_true = (1.0, 2.0, -0.5)
    X = _linear_X(v_true)
    v = velocity(X)
    np.testing.assert_allclose(v[0, 5], np.array(v_true), atol=1e-12)


def test_acceleration_zero_for_linear_motion():
    X = _linear_X((1.0, 2.0, -0.5))
    a = acceleration(X)
    assert np.max(np.abs(a)) < 1e-10


def test_jerk_zero_for_linear_motion():
    X = _linear_X((1.0, 2.0, -0.5))
    j = jerk(X)
    assert np.max(np.abs(j)) < 1e-8


def test_curvature_zero_for_linear_motion():
    X = _linear_X((1.0, 2.0, -0.5))
    k = curvature(X)
    assert np.max(np.abs(k)) < 1e-10


def test_curvature_circular_motion():
    """Circle radius r=1 in xy plane → κ ≈ 1.0 (finite-diff approximation)."""
    omega = 1.0
    ts = np.arange(11) * 0.04
    X = np.stack(
        [np.cos(omega * ts), np.sin(omega * ts), np.zeros(11)], axis=-1
    )[None]  # (1, 11, 3)
    k = curvature(X)
    # mid-window timesteps: forward-diff approximation should be near 1
    assert 0.9 < float(k[0, 5, 0]) < 1.1


# ---------- wingbeat_fft ----------

def test_wingbeat_fft_shape_and_broadcast():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 11, 3))
    out = wingbeat_fft(X, n_bins=3)
    assert out.shape == (5, 11, 9)
    # broadcast across timesteps: same value at every t
    assert np.allclose(out[:, 0, :], out[:, 5, :])
    assert np.allclose(out[:, 0, :], out[:, 10, :])


def test_wingbeat_fft_dc_component_matches_mean():
    """DC bin (k=0) of rfft = sum(x); magnitude = |sum(x)|."""
    X = np.zeros((1, 11, 3))
    X[0, :, 0] = 1.0  # constant signal
    out = wingbeat_fft(X, n_bins=1)
    # axis 0 DC magnitude = 11 * 1 = 11; axes 1,2 zero
    assert out.shape == (1, 11, 3)
    np.testing.assert_allclose(out[0, 0, 0], 11.0, atol=1e-10)
    np.testing.assert_allclose(out[0, 0, 1], 0.0, atol=1e-10)


# ---------- factory ----------

def test_make_feature_fn_dims():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 11, 3))
    assert make_feature_fn(["relative"])(X).shape == (4, 11, 3)
    assert make_feature_fn(["relative", "physics"])(X).shape == (4, 11, 13)
    assert make_feature_fn(["relative", "wingbeat"])(X).shape == (4, 11, 12)
    assert make_feature_fn(["relative", "physics", "wingbeat"])(X).shape == (4, 11, 22)


def test_make_feature_fn_matches_static():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 11, 3))
    np.testing.assert_array_equal(
        make_feature_fn(["relative"])(X), relative_coords_feature(X)
    )
    np.testing.assert_array_equal(
        make_feature_fn(["relative", "physics"])(X), physics_feature(X)
    )
    np.testing.assert_array_equal(
        make_feature_fn(["relative", "wingbeat"])(X), wingbeat_feature(X)
    )


def test_make_feature_fn_requires_relative():
    import pytest
    with pytest.raises(ValueError):
        make_feature_fn(["physics"])
