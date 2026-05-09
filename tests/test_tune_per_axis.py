import numpy as np

from src.baselines.window_polyfit import tune_per_axis
from src.io import TIMESTEPS_MS


def test_tune_recovers_known_optimum():
    rng = np.random.default_rng(0)
    n = 200
    a = rng.standard_normal((n, 3))
    b = rng.standard_normal((n, 3)) * 0.01
    t = TIMESTEPS_MS[None, :, None].astype(np.float64)
    X_clean = a[:, None, :] + b[:, None, :] * t
    noise = rng.standard_normal(X_clean.shape) * 0.01
    X_noisy = X_clean.copy()
    X_noisy[:, :-2, :] += noise[:, :-2, :]
    y = a + b * 80.0
    grid = [(2, 1), (5, 1), (5, 2), (11, 1), (11, 2), (11, 3)]
    chosen, errors = tune_per_axis(X_noisy, y, grid)
    assert len(chosen) == 3
    assert all(c == (2, 1) for c in chosen)
    for axis_errs in errors.values():
        assert axis_errs[(2, 1)] <= axis_errs[(11, 1)]


def test_tune_grid_invalid_filtered():
    n = 5
    X = np.zeros((n, 11, 3))
    y = np.zeros((n, 3))
    grid = [(2, 1), (1, 0), (12, 1), (3, 5)]
    chosen, errors = tune_per_axis(X, y, grid)
    for axis_errs in errors.values():
        assert (2, 1) in axis_errs
        assert (12, 1) not in axis_errs
        assert (3, 5) not in axis_errs
