"""src/combine.py — winning component identification + R006 config builder.

Per plan-003 §3.4 + §8.1.

Winning rule (§3.4): paired mean Δ = (R00x.cv_mean_eucl - R001.cv_mean_eucl) < -noise_margin.
Default noise_margin = 0.0 (보수적; 0 ≤ Δ ≤ 0.001 영역도 non-winning).

build_r006_config (component-axis mapping):
  R002 (physics)  → feature_components += ["physics"]
  R003 (ema)      → baseline_type = "ema", ema_alpha = 0.5
  R004 (wingbeat) → feature_components += ["wingbeat"], wingbeat_n_bins = 3
  R005 (mse)      → loss_type = "mse"
input_dim 재계산 = 3 + 10*("physics" in fc) + 9*("wingbeat" in fc).

CLI: `python -m src.combine` runs the c12 workflow end-to-end:
  read R001~R005 summary → identify winning → write winning_trace.md →
  generate R006 config → branch (winning=0 cp / winning≥1 train) →
  update summary + registry + fallback flag check.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shutil
from copy import deepcopy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_ROOT = PROJECT_ROOT / "runs"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"
REGISTRY = PROJECT_ROOT / "registry.csv"

REGISTRY_COLS = [
    "id", "plan_id", "type", "status", "started_at", "finished_at",
    "duration_sec", "run_dir", "config_path", "baseline_id", "corrects", "notes",
]

R001_ID = "R001_baseline-residual-gru"
R006_ID = "R006_combined-winners"
ABLATION_MAP = {
    "R002": "R002_physics-features",
    "R003": "R003_ema-extrapolate",
    "R004": "R004_wingbeat-oscillation",
    "R005": "R005_loss-mse",
}
FALLBACK_DELTA_THRESHOLD = 0.001  # R006.cv > R001.cv + 0.001 → fallback


def _now_kst() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(timespec="seconds")

# Component-to-axis mapping (§3.4 table).
# Keys are R002..R005 short names; values describe the config diff vs R001.
COMPONENT_AXES: dict[str, dict] = {
    "R002": {"feature_components_add": ["physics"]},
    "R003": {"baseline_type": "ema", "ema_alpha": 0.5},
    "R004": {"feature_components_add": ["wingbeat"], "wingbeat_n_bins": 3},
    "R005": {"loss_type": "mse"},
}


def identify_winning(
    r001_summary: dict,
    r00x_summaries: dict[str, dict],
    noise_margin: float = 0.0,
) -> dict[str, bool]:
    """For each R002..R005, mark winning if cv_mean_eucl < R001.cv_mean_eucl - noise_margin.

    r001_summary:        loaded summary.json of R001
    r00x_summaries:      {"R002": summary_dict, "R003": ..., ...}
    return:              {"R002": bool, "R003": bool, ...}  (only keys present in input)
    """
    r001_cv = float(r001_summary["cv_mean_eucl"])
    return {
        short: (float(s["cv_mean_eucl"]) - r001_cv) < -noise_margin
        for short, s in r00x_summaries.items()
    }


def _recompute_input_dim(
    feature_components: list[str], wingbeat_n_bins: int = 3
) -> int:
    base = 3  # relative
    if "physics" in feature_components:
        base += 10  # vel(3) + acc(3) + jerk(3) + curvature(1)
    if "wingbeat" in feature_components:
        base += 3 * wingbeat_n_bins
    return base


def build_r006_config(
    winning: dict[str, bool],
    r001_config: dict,
) -> dict:
    """Build R006 config dict from R001 config + winning flags.

    Always sets exp_id="R006_combined-winners".
    Winning ≥ 1: baseline_id="R001_baseline-residual-gru".
    Winning = 0: bit-equal to R001 except exp_id (R006 = direct R001 copy; learning skipped).
    """
    cfg = deepcopy(r001_config)
    cfg["exp_id"] = "R006_combined-winners"

    fc = list(cfg.get("feature_components", ["relative"]))
    n_winning = sum(1 for w in winning.values() if w)

    if n_winning == 0:
        # R006 = R001 비트 동일 (단 exp_id 만 변경, baseline_id 유지)
        cfg["feature_components"] = fc
        return cfg

    # baseline_id 갱신 (winning ≥ 1 시 R001 이 reference)
    cfg["baseline_id"] = "R001_baseline-residual-gru"

    if winning.get("R002"):
        if "physics" not in fc:
            fc.append("physics")
    if winning.get("R003"):
        cfg["baseline_type"] = "ema"
        cfg["ema_alpha"] = 0.5
    if winning.get("R004"):
        if "wingbeat" not in fc:
            fc.append("wingbeat")
        cfg["wingbeat_n_bins"] = 3
    if winning.get("R005"):
        cfg["loss_type"] = "mse"

    cfg["feature_components"] = fc
    wb = int(cfg.get("wingbeat_n_bins", 3))
    cfg["model"]["input_dim"] = _recompute_input_dim(fc, wingbeat_n_bins=wb)
    return cfg


# ---------- c12 workflow ----------

def _load_summary(exp_id: str) -> dict:
    return json.loads((RUNS_ROOT / "baseline" / exp_id / "summary.json").read_text())


def _load_config_yaml(exp_id: str) -> dict:
    return yaml.safe_load((CONFIGS_ROOT / "baseline" / f"{exp_id}.yaml").read_text())


def _write_winning_trace(
    out_path: Path,
    r001_summary: dict,
    r00x_summaries: dict[str, dict],
    winning: dict[str, bool],
    fallback: bool | None = None,
    r006_summary: dict | None = None,
) -> None:
    r001_cv = float(r001_summary["cv_mean_eucl"])
    lines = [
        "# plan-003 winning trace (c12 / G3.5)",
        "",
        "## §1. Ablation paired Δ vs R001",
        "",
        f"R001 reference cv_mean_eucl = **{r001_cv:.6f}**",
        "",
        "| exp_id | cv_mean_eucl | Δ vs R001 | winning? |",
        "|---|---:|---:|:-:|",
    ]
    for short, full_id in ABLATION_MAP.items():
        s = r00x_summaries[short]
        cv = float(s["cv_mean_eucl"])
        delta = cv - r001_cv
        win = "YES" if winning.get(short) else "no"
        lines.append(f"| {full_id} | {cv:.6f} | {delta:+.6f} | {win} |")
    lines += ["", "## §2. R006 config 자동 생성", ""]
    n_w = sum(1 for w in winning.values() if w)
    if n_w == 0:
        lines += [
            f"- winning components = **0** → R006 = R001 직접 복제 (학습 skip).",
            f"- decision-note: spec-default — winning 0개, R006 ckpt + summary 모두 R001 의 비트 동일 사본.",
        ]
    else:
        lines += [
            f"- winning components = **{n_w}** ({[s for s, w in winning.items() if w]}).",
            f"- additive 합산 가정으로 모든 winning component 를 R006 config 에 반영.",
        ]
    if r006_summary is not None:
        r006_cv = float(r006_summary["cv_mean_eucl"])
        lines += [
            "",
            "## §3. R006 결과 + fallback verdict",
            "",
            f"- R006 cv_mean_eucl = **{r006_cv:.6f}**",
            f"- vs R001 Δ = {r006_cv - r001_cv:+.6f}",
        ]
        if fallback is True:
            lines += [
                f"- **fallback = True** (R006.cv > R001.cv + {FALLBACK_DELTA_THRESHOLD})",
                "- LB submission = R001 의 csv (combined_no_improvement warn 박제).",
            ]
        else:
            lines += [
                f"- **fallback = False** (R006.cv ≤ R001.cv + {FALLBACK_DELTA_THRESHOLD})",
                "- LB submission = R006 의 csv.",
            ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")


def _copy_r001_to_r006(r001_summary: dict, r001_config: dict) -> dict:
    """winning=0 분기: R006 dir = R001 비트 동일 (단 summary.exp_id 만 변경).

    Returns the R006 summary dict (after exp_id rename).
    """
    src_dir = RUNS_ROOT / "baseline" / R001_ID
    dst_dir = RUNS_ROOT / "baseline" / R006_ID
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)

    # text artifacts
    for name in ("summary.json", "history.json", "config.snapshot.yaml", "run.log"):
        shutil.copy2(src_dir / name, dst_dir / name)
    # ckpt dir
    (dst_dir / "ckpt").mkdir()
    for ckpt in (src_dir / "ckpt").glob("fold*.pt"):
        shutil.copy2(ckpt, dst_dir / "ckpt" / ckpt.name)

    # rename exp_id in summary.json (keep baseline_id, config dict内 exp_id 도)
    s = json.loads((dst_dir / "summary.json").read_text())
    s["exp_id"] = R006_ID
    if "config" in s and isinstance(s["config"], dict):
        s["config"]["exp_id"] = R006_ID
    (dst_dir / "summary.json").write_text(json.dumps(s, indent=2))

    # config.snapshot.yaml: exp_id 만 R006 으로
    snap = yaml.safe_load((dst_dir / "config.snapshot.yaml").read_text())
    snap["exp_id"] = R006_ID
    (dst_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(snap, sort_keys=False))

    return s


def _append_registry_row(
    exp_id: str, r006_summary: dict, config_path: Path, notes: str,
) -> None:
    started = r006_summary.get("started_at") or _now_kst()
    finished = r006_summary.get("finished_at") or started
    duration = float(r006_summary.get("duration_sec", 0.0))
    row = {
        "id": exp_id,
        "plan_id": "003",
        "type": "baseline",
        "status": "complete",
        "started_at": started,
        "finished_at": finished,
        "duration_sec": duration,
        "run_dir": str((RUNS_ROOT / "baseline" / exp_id).relative_to(PROJECT_ROOT)),
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "baseline_id": R001_ID,
        "corrects": "",
        "notes": notes,
    }
    file_exists = REGISTRY.exists() and REGISTRY.stat().st_size > 0
    with REGISTRY.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_COLS)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise-margin", type=float, default=0.0)
    args = ap.parse_args(argv)

    # 1. read R001~R005 summary
    r001_summary = _load_summary(R001_ID)
    r001_config = _load_config_yaml(R001_ID)
    r00x_summaries = {short: _load_summary(full) for short, full in ABLATION_MAP.items()}

    # 2. identify winning
    winning = identify_winning(r001_summary, r00x_summaries, noise_margin=args.noise_margin)
    print(f"[combine] winning = {winning}")

    # 3. write R006 config yaml
    r006_cfg = build_r006_config(winning, r001_config)
    r006_cfg_path = CONFIGS_ROOT / "baseline" / f"{R006_ID}.yaml"
    r006_cfg_path.write_text(yaml.safe_dump(r006_cfg, sort_keys=False))
    print(f"[combine] wrote {r006_cfg_path}")

    n_winning = sum(1 for w in winning.values() if w)
    fallback_flag: bool

    if n_winning == 0:
        # 4a. winning=0 → cp R001 → R006
        r006_summary = _copy_r001_to_r006(r001_summary, r001_config)
        notes = "winning=0, copied from R001"
        _append_registry_row(R006_ID, r006_summary, r006_cfg_path, notes)
        # fallback false (R006.cv == R001.cv, ≤ R001.cv + 0.001)
        fallback_flag = False
        print(f"[combine] winning=0 → R006 = R001 cp; fallback={fallback_flag}")
    else:
        # 4b. winning ≥ 1 → train R006 via run_baseline
        from src.run import run_baseline  # lazy import to avoid torch cost on cp branch
        print(f"[combine] winning={n_winning} → train R006")
        summary = run_baseline(r006_cfg_path)
        r006_summary = summary
        # registry row already appended by run_baseline
        # fallback check
        delta = float(r006_summary["cv_mean_eucl"]) - float(r001_summary["cv_mean_eucl"])
        fallback_flag = delta > FALLBACK_DELTA_THRESHOLD
        print(f"[combine] R006.cv={r006_summary['cv_mean_eucl']:.6f}  Δ={delta:+.6f}  fallback={fallback_flag}")

    # 5. winning_trace.md
    trace_path = ANALYSIS_ROOT / "plan-003" / "winning_trace.md"
    _write_winning_trace(
        trace_path, r001_summary, r00x_summaries, winning,
        fallback=fallback_flag, r006_summary=r006_summary,
    )
    print(f"[combine] wrote {trace_path}")

    # 6. fallback flag persisted in winning_trace.md (read by c13 to choose lb_exp_id)
    print(f"[combine] FINAL: lb_exp_id = "
          f"{R001_ID if fallback_flag else R006_ID}  (fallback={fallback_flag})")


if __name__ == "__main__":
    main()
