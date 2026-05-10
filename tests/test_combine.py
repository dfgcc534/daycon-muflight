"""tests/test_combine.py — winning identification + R006 config auto-generation.

Per plan-003 §4.6 (c12 with src/combine.py).
"""
from __future__ import annotations

from copy import deepcopy

from src.combine import build_r006_config, identify_winning


R001_SUMMARY = {"cv_mean_eucl": 0.013}
R001_CONFIG = {
    "exp_id": "R001_baseline-residual-gru",
    "type": "baseline",
    "plan_id": "003",
    "baseline_id": "B001_linear-2pt",
    "method": "gru-residual",
    "feature_components": ["relative"],
    "baseline_type": "linear",
    "loss_type": "huber",
    "t_target": 80,
    "k": 5,
    "seed": 42,
    "model": {"hidden": 64, "layers": 2, "dropout": 0.1, "input_dim": 3},
    "training": {
        "lr": 1e-3, "weight_decay": 1e-4, "batch": 64,
        "epochs": 100, "early_stop_patience": 10, "seed": 42,
    },
}


# ---------- identify_winning ----------

def test_identify_winning_zero():
    """All R00x cv ≥ R001 cv → no winning."""
    summaries = {
        "R002": {"cv_mean_eucl": 0.014},
        "R003": {"cv_mean_eucl": 0.015},
        "R004": {"cv_mean_eucl": 0.013},  # = R001 → not winning
        "R005": {"cv_mean_eucl": 0.0131},
    }
    win = identify_winning(R001_SUMMARY, summaries)
    assert win == {"R002": False, "R003": False, "R004": False, "R005": False}


def test_identify_winning_one_R002():
    summaries = {
        "R002": {"cv_mean_eucl": 0.012},   # < 0.013 → winning
        "R003": {"cv_mean_eucl": 0.014},
        "R004": {"cv_mean_eucl": 0.014},
        "R005": {"cv_mean_eucl": 0.014},
    }
    win = identify_winning(R001_SUMMARY, summaries)
    assert win == {"R002": True, "R003": False, "R004": False, "R005": False}


def test_identify_winning_two_R002_R003():
    summaries = {
        "R002": {"cv_mean_eucl": 0.012},
        "R003": {"cv_mean_eucl": 0.0125},
        "R004": {"cv_mean_eucl": 0.014},
        "R005": {"cv_mean_eucl": 0.014},
    }
    win = identify_winning(R001_SUMMARY, summaries)
    assert win == {"R002": True, "R003": True, "R004": False, "R005": False}


def test_identify_winning_all_four():
    summaries = {
        "R002": {"cv_mean_eucl": 0.012},
        "R003": {"cv_mean_eucl": 0.011},
        "R004": {"cv_mean_eucl": 0.0125},
        "R005": {"cv_mean_eucl": 0.0124},
    }
    win = identify_winning(R001_SUMMARY, summaries)
    assert all(win.values())


def test_identify_winning_noise_margin():
    """Δ = -0.0005 < 0 but > -0.001 → non-winning if margin=0.001."""
    summaries = {"R002": {"cv_mean_eucl": 0.0125}}  # Δ = -0.0005
    assert identify_winning(R001_SUMMARY, summaries) == {"R002": True}
    assert identify_winning(R001_SUMMARY, summaries, noise_margin=0.001) == {"R002": False}


# ---------- build_r006_config ----------

def test_build_r006_zero_winning():
    """Winning 0 → R006 = R001 except exp_id."""
    win = {"R002": False, "R003": False, "R004": False, "R005": False}
    cfg = build_r006_config(win, R001_CONFIG)
    assert cfg["exp_id"] == "R006_combined-winners"
    assert cfg["feature_components"] == ["relative"]
    assert cfg["baseline_type"] == "linear"
    assert cfg["loss_type"] == "huber"
    assert cfg["model"]["input_dim"] == 3
    # baseline_id stays as R001's (= B001) for winning=0 branch
    assert cfg["baseline_id"] == "B001_linear-2pt"


def test_build_r006_R002_winning():
    win = {"R002": True, "R003": False, "R004": False, "R005": False}
    cfg = build_r006_config(win, R001_CONFIG)
    assert cfg["exp_id"] == "R006_combined-winners"
    assert "physics" in cfg["feature_components"]
    assert cfg["baseline_type"] == "linear"
    assert cfg["loss_type"] == "huber"
    assert cfg["model"]["input_dim"] == 13
    assert cfg["baseline_id"] == "R001_baseline-residual-gru"


def test_build_r006_R003_winning():
    win = {"R002": False, "R003": True, "R004": False, "R005": False}
    cfg = build_r006_config(win, R001_CONFIG)
    assert cfg["baseline_type"] == "ema"
    assert cfg["ema_alpha"] == 0.5
    assert cfg["model"]["input_dim"] == 3


def test_build_r006_R004_winning():
    win = {"R002": False, "R003": False, "R004": True, "R005": False}
    cfg = build_r006_config(win, R001_CONFIG)
    assert "wingbeat" in cfg["feature_components"]
    assert cfg["wingbeat_n_bins"] == 3
    assert cfg["model"]["input_dim"] == 12


def test_build_r006_R005_winning():
    win = {"R002": False, "R003": False, "R004": False, "R005": True}
    cfg = build_r006_config(win, R001_CONFIG)
    assert cfg["loss_type"] == "mse"
    assert cfg["model"]["input_dim"] == 3


def test_build_r006_R002_and_R004_winning():
    """Combined feature_components additive. input_dim = 22."""
    win = {"R002": True, "R003": False, "R004": True, "R005": False}
    cfg = build_r006_config(win, R001_CONFIG)
    assert "physics" in cfg["feature_components"]
    assert "wingbeat" in cfg["feature_components"]
    assert cfg["model"]["input_dim"] == 22


def test_build_r006_does_not_mutate_input():
    """build_r006_config must deepcopy."""
    snap = deepcopy(R001_CONFIG)
    _ = build_r006_config({"R002": True}, R001_CONFIG)
    assert R001_CONFIG == snap
