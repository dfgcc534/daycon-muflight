import numpy as np
import pytest

from src.baselines.window_polyfit import predict, predict_per_axis
from src.io import TIMESTEPS_MS


def _synthetic(n: int, intercept, slope, quad=None, cubic=None, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = intercept(rng, (n, 3))
    b = slope(rng, (n, 3))
    t = TIMESTEPS_MS[None, :, None].astype(np.float64)
    X = a[:, None, :] + b[:, None, :] * t
    if quad is not None:
        c = quad(rng, (n, 3))
        X = X + c[:, None, :] * t**2
    else:
        c = None
    if cubic is not None:
        d = cubic(rng, (n, 3))
        X = X + d[:, None, :] * t**3
    else:
        d = None
    return X, (a, b, c, d)


def _eval_at(t_target, a, b, c=None, d=None):
    out = a + b * t_target
    if c is not None:
        out = out + c * t_target**2
    if d is not None:
        out = out + d * t_target**3
    return out


def test_predict_linear_window2_exact():
    X, (a, b, _, _) = _synthetic(8, lambda r, s: r.standard_normal(s),
                                 lambda r, s: r.standard_normal(s))
    pred = predict(X, window=2, degree=1, t_target=80)
    expected = _eval_at(80, a, b)
    assert np.allclose(pred, expected, atol=1e-9)


def test_predict_linear_window11_exact():
    X, (a, b, _, _) = _synthetic(8, lambda r, s: r.standard_normal(s),
                                 lambda r, s: r.standard_normal(s))
    pred = predict(X, window=11, degree=1, t_target=80)
    expected = _eval_at(80, a, b)
    assert np.allclose(pred, expected, atol=1e-9)


def test_predict_quadratic_recovers_with_deg2():
    X, (a, b, c, _) = _synthetic(
        8,
        lambda r, s: r.standard_normal(s),
        lambda r, s: r.standard_normal(s) * 0.01,
        quad=lambda r, s: r.standard_normal(s) * 1e-5,
    )
    pred = predict(X, window=11, degree=2, t_target=80)
    expected = _eval_at(80, a, b, c)
    assert np.allclose(pred, expected, atol=1e-6)


def test_predict_cubic_underfit_with_deg2():
    X, (a, b, c, d) = _synthetic(
        8,
        lambda r, s: r.standard_normal(s),
        lambda r, s: r.standard_normal(s) * 0.01,
        quad=lambda r, s: r.standard_normal(s) * 1e-5,
        cubic=lambda r, s: r.standard_normal(s) * 1e-7,
    )
    pred_d2 = predict(X, window=11, degree=2, t_target=80)
    pred_d3 = predict(X, window=11, degree=3, t_target=80)
    expected = _eval_at(80, a, b, c, d)
    err_d2 = np.abs(pred_d2 - expected).max()
    err_d3 = np.abs(pred_d3 - expected).max()
    assert err_d3 < err_d2
    assert err_d3 < 1e-6


def test_predict_validation():
    X = np.zeros((1, 11, 3))
    with pytest.raises(ValueError):
        predict(X, window=2, degree=2)
    with pytest.raises(ValueError):
        predict(X, window=12, degree=1)


def test_predict_per_axis_shape_and_independence():
    X, _ = _synthetic(4, lambda r, s: r.standard_normal(s),
                      lambda r, s: r.standard_normal(s))
    out = predict_per_axis(X, [(2, 1), (3, 1), (5, 2)])
    assert out.shape == (4, 3)
    out_ref = predict_per_axis(X, [(2, 1), (2, 1), (2, 1)])
    out_scalar = predict(X, window=2, degree=1)
    assert np.allclose(out_ref, out_scalar, atol=1e-9)
    with pytest.raises(ValueError):
        predict_per_axis(X, [(2, 1), (2, 1)])
