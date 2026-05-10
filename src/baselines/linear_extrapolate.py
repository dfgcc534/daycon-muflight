"""Linear and EMA-weighted extrapolation baselines for residual-GRU.

Per plan-003 §4.2.

`linear_extrap` is bit-equivalent to B001_linear-2pt (plan-001 closed-form
floor cv=0.01294, LB=0.60). residual = y_true - linear_extrap(X).
"""
from __future__ import annotations

import numpy as np

from src.io import TIMESTEPS_MS


def linear_extrap(
    X: np.ndarray,
    t_target_ms: int = 80,
    timesteps_ms: np.ndarray = TIMESTEPS_MS,
) -> np.ndarray:
    """B001 식 그대로: pred = X[:, -1] + (t_target / dt) * (X[:, -1] - X[:, -2]).

    Equivalent to `window_polyfit.predict(X, window=2, degree=1, t_target=80)`.
    Per §4.2: bit-equivalence verified in tests/test_ema_extrapolate.py.
    """
    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError(f"X must be (n, T, 3); got {X.shape}")
    dt_target = t_target_ms / (timesteps_ms[-1] - timesteps_ms[-2])  # 80 / 40 = 2.0
    return X[:, -1] + dt_target * (X[:, -1] - X[:, -2])


def ema_extrapolate(
    X: np.ndarray,
    alpha: float = 0.5,
    t_target_ms: int = 80,
    timesteps_ms: np.ndarray = TIMESTEPS_MS,
) -> np.ndarray:
    """Exponentially-weighted velocity extrapolation (R003 baseline; R006 if R003 winning).

    v_k = (X[:, -k] - X[:, -k-1]) / dt   (k = 1 .. T-1; k=1 is most recent)
    weights ∝ alpha^(k-1)        (k=1 weight 1, k=2 weight α, ...)
    v_ema = Σ weights * v_k / Σ weights
    pred = X[:, -1] + t_target_ms * v_ema
    """
    if X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError(f"X must be (n, T, 3); got {X.shape}")
    dt_ms = float(timesteps_ms[1] - timesteps_ms[0])  # 40.0 ms
    velocities = (X[:, 1:, :] - X[:, :-1, :]) / dt_ms  # (n, T-1, 3) — units m/ms
    n_v = velocities.shape[1]
    # array-axis k=0 = oldest velocity, k=n_v-1 = most recent
    # weight @ array index k = alpha^(n_v-1-k); equivalently reverse a alpha^k array
    raw = np.array([alpha ** k for k in range(n_v)], dtype=np.float64)
    weights = raw[::-1] / raw.sum()
    v_ema = np.einsum("ntd,t->nd", velocities, weights)
    return X[:, -1] + t_target_ms * v_ema
