import numpy as np

from src.eval import DEFAULT_RADII, eucl, hit_rate, mean_eucl, per_axis_mae, summarize


def test_eucl_zero_when_equal():
    p = np.zeros((5, 3))
    assert np.allclose(eucl(p, p), 0.0)


def test_eucl_known_values():
    p = np.array([[0, 0, 0], [3, 4, 0]], dtype=float)
    t = np.array([[1, 0, 0], [0, 0, 0]], dtype=float)
    assert np.allclose(eucl(p, t), [1.0, 5.0])


def test_mean_and_per_axis():
    p = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    t = np.array([[0, 0, 1], [1, 1, 1]], dtype=float)
    assert mean_eucl(p, t) == 0.5
    assert np.allclose(per_axis_mae(p, t), [0.0, 0.0, 0.5])


def test_hit_rate_boundary():
    p = np.array([[0, 0, 0], [0, 0, 0.1]], dtype=float)
    t = np.zeros_like(p)
    assert hit_rate(p, t, 0.05) == 0.5
    assert hit_rate(p, t, 0.10) == 1.0
    assert hit_rate(p, t, 0.09) == 0.5


def test_summarize_schema():
    p = np.zeros((10, 3))
    t = np.zeros((10, 3))
    out = summarize(p, t)
    expected_keys = {"mean_eucl", "median_eucl", "p95_eucl", "max_eucl",
                     "per_axis_mae", "hit_rate", "n"}
    assert set(out.keys()) == expected_keys
    assert out["n"] == 10
    assert set(out["hit_rate"].keys()) == {f"{r:.2f}" for r in DEFAULT_RADII}
    assert out["mean_eucl"] == 0.0
    assert all(v == 1.0 for v in out["hit_rate"].values())
