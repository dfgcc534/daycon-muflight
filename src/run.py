"""Baseline experiment runner.

Reads a YAML config, runs 5-fold CV (or per-config k), writes:
  runs/{type}/{exp_id}/summary.json
  runs/{type}/{exp_id}/history.json (per-fold metrics)
  runs/{type}/{exp_id}/run.log
  runs/{type}/{exp_id}/config.snapshot.yaml
and appends one row to registry.csv.

CLI: python -m src.run configs/baseline/B001_linear-2pt.yaml
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import yaml

from src.baselines.window_polyfit import predict, predict_per_axis
from src.eval import summarize
from src.io import TIMESTEPS_MS, kfold_split, load_all_samples, load_labels

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "runs"
REGISTRY = PROJECT_ROOT / "registry.csv"

REGISTRY_COLS = [
    "id", "plan_id", "type", "status", "started_at", "finished_at",
    "duration_sec", "run_dir", "config_path", "baseline_id", "corrects", "notes",
]


def now_kst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds")


def predict_for_config(X: np.ndarray, cfg: dict) -> np.ndarray:
    pa = cfg.get("per_axis")
    t_target = int(cfg.get("t_target", 80))
    if pa:
        configs = [tuple(c) for c in pa]
        return predict_per_axis(X, configs, t_target=t_target, timesteps=TIMESTEPS_MS)
    return predict(X, int(cfg["window"]), int(cfg["degree"]),
                   t_target=t_target, timesteps=TIMESTEPS_MS)


def append_registry(row: dict) -> None:
    file_exists = REGISTRY.exists() and REGISTRY.stat().st_size > 0
    with REGISTRY.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_COLS)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in REGISTRY_COLS})


def run_baseline(config_path: Path, X=None, y=None, ids=None) -> dict:
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    exp_id = cfg["exp_id"]
    run_dir = RUNS_ROOT / cfg["type"] / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{now_kst()}] {msg}"
        log_lines.append(line)
        print(line, flush=True)

    started = now_kst()
    t0 = time.monotonic()
    log(f"start exp_id={exp_id}, config={config_path}")

    if X is None or y is None or ids is None:
        log("load train + labels")
        ids_X, X = load_all_samples("train")
        ids_y, y = load_labels()
        assert ids_X == ids_y, "id order mismatch between samples and labels"
        ids = ids_X
    log(f"n_train={len(ids)}")

    folds = kfold_split(ids, k=int(cfg.get("k", 5)), seed=int(cfg.get("seed", 42)))
    fold_metrics: list[dict] = []
    oof_preds = np.empty_like(y)
    for fi, (_tr, va) in enumerate(folds):
        pred = predict_for_config(X[va], cfg)
        oof_preds[va] = pred
        s = summarize(pred, y[va])
        s["fold"] = fi
        fold_metrics.append(s)
        log(f"fold {fi}: mean_eucl={s['mean_eucl']:.5f} "
            f"per_axis_mae={[round(v, 4) for v in s['per_axis_mae']]}")

    arr_mean = np.array([f["mean_eucl"] for f in fold_metrics])
    cv_mean = float(arr_mean.mean())
    cv_std = float(arr_mean.std(ddof=0))
    cv_per_axis = np.mean([f["per_axis_mae"] for f in fold_metrics], axis=0).tolist()
    radii_keys = list(fold_metrics[0]["hit_rate"].keys())
    cv_hit = {k: float(np.mean([f["hit_rate"][k] for f in fold_metrics]))
              for k in radii_keys}
    oof_summary = summarize(oof_preds, y)
    log(f"CV mean_eucl={cv_mean:.5f} ± {cv_std:.5f} | "
        f"OOF mean_eucl={oof_summary['mean_eucl']:.5f}")

    duration = round(time.monotonic() - t0, 3)
    finished = now_kst()

    summary = {
        "exp_id": exp_id,
        "type": cfg["type"],
        "plan_id": str(cfg.get("plan_id", "001")),
        "started_at": started,
        "finished_at": finished,
        "duration_sec": duration,
        "n_train": len(ids),
        "k": int(cfg.get("k", 5)),
        "cv_mean_eucl": cv_mean,
        "cv_std_eucl": cv_std,
        "cv_per_axis_mae": cv_per_axis,
        "cv_hit_rate": cv_hit,
        "oof_summary": oof_summary,
        "fold_metrics": fold_metrics,
        "config": cfg,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "history.json").write_text(json.dumps(fold_metrics, indent=2))
    (run_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (run_dir / "run.log").write_text("\n".join(log_lines) + "\n")

    append_registry({
        "id": exp_id,
        "plan_id": str(cfg.get("plan_id", "001")),
        "type": cfg["type"],
        "status": "complete",
        "started_at": started,
        "finished_at": finished,
        "duration_sec": duration,
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "config_path": str(Path(config_path).relative_to(PROJECT_ROOT)),
        "baseline_id": cfg.get("baseline_id", "") or "",
        "corrects": "",
        "notes": f"cv_mean_eucl={cv_mean:.5f}±{cv_std:.5f}",
    })
    log(f"DONE in {duration}s, run_dir={run_dir}")
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("config", type=Path)
    args = p.parse_args(argv)
    run_baseline(args.config)


if __name__ == "__main__":
    main()
