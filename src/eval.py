"""Evaluation metrics for muflight (DACON 236716).

Per plan-001 §3.3, §4.2.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def eucl(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - true, axis=-1)


def mean_eucl(pred: np.ndarray, true: np.ndarray) -> float:
    return float(eucl(pred, true).mean())


def per_axis_mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.abs(pred - true).mean(axis=0)


def hit_rate(pred: np.ndarray, true: np.ndarray, radius: float) -> float:
    return float((eucl(pred, true) <= radius).mean())


DEFAULT_RADII: tuple[float, ...] = (0.05, 0.10, 0.20, 0.50)


def summarize(
    pred: np.ndarray,
    true: np.ndarray,
    radii: Sequence[float] = DEFAULT_RADII,
) -> dict:
    e = eucl(pred, true)
    return {
        "mean_eucl": float(e.mean()),
        "median_eucl": float(np.median(e)),
        "p95_eucl": float(np.percentile(e, 95)),
        "max_eucl": float(e.max()),
        "per_axis_mae": per_axis_mae(pred, true).tolist(),
        "hit_rate": {f"{r:.2f}": float((e <= r).mean()) for r in radii},
        "n": int(true.shape[0]),
    }
