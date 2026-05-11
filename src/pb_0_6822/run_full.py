"""plan-004 c5 — orchestrator for src/pb_0_6822/{selector,boundary}.py.

Runs:
- smoke: notebook cell 10 + cell 12 1-fold pilot (mirror exactly) for extraction validation
- full:  5-fold selector full-fit + boundary corrector full-fit with --make-test → submission.csv

Usage:
    .venv/bin/python -m src.pb_0_6822.run_full --smoke
    .venv/bin/python -m src.pb_0_6822.run_full
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
OUT_DIR = REPO / "runs/baseline/P001_pb-0-6822-fullrun"


def ensure_data_extracted() -> None:
    """data/open.zip → data/{train,test,train_labels.csv,sample_submission.csv} auto-extract."""
    needed = ["train_labels.csv", "sample_submission.csv", "train", "test"]
    if all((DATA_ROOT / p).exists() for p in needed):
        return
    zip_path = DATA_ROOT / "open.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"data/open.zip not found at {zip_path}")
    print(f"[setup] extracting {zip_path} ...", flush=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_ROOT)
    print(f"[setup] extracted to {DATA_ROOT}", flush=True)


def _call_main(main_func, argv: list) -> None:
    """Save/restore sys.argv around argparse-based main() — mirrors notebook cell 8."""
    old = sys.argv[:]
    try:
        sys.argv = [main_func.__name__, *[str(a) for a in argv]]
        start = time.time()
        main_func()
        print(f"[DONE] {main_func.__name__} elapsed={time.time() - start:.1f}s", flush=True)
    finally:
        sys.argv = old


def run_smoke() -> None:
    """1-fold smoke mirroring notebook cell 10 + cell 12 (1 epoch, fold-limit=1)."""
    from src.pb_0_6822 import boundary, selector

    smoke_dir = OUT_DIR / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    selector_out = smoke_dir / "selector"
    boundary_out = smoke_dir / "boundary"

    # Notebook cell 10 args (verbatim hyperparams)
    _call_main(selector.SELECTOR_MAIN, [
        "--root", DATA_ROOT,
        "--out-dir", selector_out,
        "--models", "attn_gru",
        "--folds", 5, "--fold-limit", 1,
        "--pre-epochs", 1, "--fine-epochs", 1, "--freeze-fine-epochs", 1,
        "--epoch-plus", 0, "--min-epochs", 1, "--patience", 1,
        "--hidden", 48, "--batch", 4096,
        "--lr", 0.001, "--fine-lr-scale", 0.12,
        "--prior-strength", 0.65, "--regime-prior-strength", 0.45,
        "--pairwise-loss-weight", 0.25, "--pairwise-margin", 0.12, "--pairwise-min-label-gap", 0.04,
        "--fine-distill-weight", 0.55, "--fine-distill-temp", 0.07,
        "--reverse-pretrain", "--norm-real-only",
        "--device", "auto", "--seed", 20260506, "--log-every", 1, "--skip-full",
    ])

    score_bank = selector_out / "oof_selector_scores.npz"
    assert score_bank.exists(), f"missing {score_bank}"

    # Notebook cell 12 args
    _call_main(boundary.BOUNDARY_MAIN, [
        "--root", DATA_ROOT,
        "--out-dir", boundary_out,
        "--fold", 0, "--folds", 5,
        "--score-bank", score_bank,
        "--epochs", 1, "--fine-epochs", 1, "--min-epochs", 1, "--patience", 1,
        "--hidden", 64, "--batch", 8192,
        "--lr", 0.001, "--fine-lr-scale", 0.18,
        "--cap", 0.006, "--apply-scale", 1.0,
        "--device", "auto", "--seed", 20260606, "--save-val-pred",
    ])

    _print_summary(selector_out, boundary_out, "smoke")


def run_full() -> None:
    """Full 5-fold selector + corrector full-fit with submission generation.

    Hyperparam budget: notebook default 의 ~70% for Mac mps ~60min wall-time.
    """
    from src.pb_0_6822 import boundary, selector

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Selector: full 5-fold + full-fit (no --skip-full → test_selector_scores.npz auto-generated)
    _call_main(selector.SELECTOR_MAIN, [
        "--root", DATA_ROOT,
        "--out-dir", OUT_DIR,
        "--models", "attn_gru",
        "--folds", 5, "--fold-limit", 5,
        # epoch budget: ~70% of notebook default
        "--pre-epochs", 10,           # default 14
        "--fine-epochs", 8,           # default 10
        "--freeze-fine-epochs", 3,    # default 3
        "--epoch-plus", 5,            # default 10
        "--min-epochs", 5,            # default 5
        "--patience", 4,              # default 8
        "--hidden", 48, "--batch", 4096,
        "--lr", 0.001, "--fine-lr-scale", 0.12,
        "--prior-strength", 0.65, "--regime-prior-strength", 0.45,
        "--pairwise-loss-weight", 0.25, "--pairwise-margin", 0.12, "--pairwise-min-label-gap", 0.04,
        "--fine-distill-weight", 0.55, "--fine-distill-temp", 0.07,
        "--reverse-pretrain", "--norm-real-only",
        "--device", "auto", "--seed", 20260506, "--log-every", 1,
        # NO --skip-full → full-fit + test_selector_scores.npz
    ])

    score_bank = OUT_DIR / "oof_selector_scores.npz"
    test_score_bank = OUT_DIR / "test_selector_scores.npz"
    assert score_bank.exists(), f"missing OOF: {score_bank}"
    assert test_score_bank.exists(), f"missing test score bank: {test_score_bank}"

    # Boundary: full-fit (single fold 0 for report + --make-test trains separate full-data model for submission)
    _call_main(boundary.BOUNDARY_MAIN, [
        "--root", DATA_ROOT,
        "--out-dir", OUT_DIR,
        "--fold", 0, "--folds", 5,
        "--score-bank", score_bank,
        "--test-score-bank", test_score_bank,
        # epoch budget: ~70% of notebook default
        "--epochs", 12,         # default 50
        "--fine-epochs", 8,     # default 20
        "--min-epochs", 5,      # default 10
        "--patience", 4,        # default 8
        "--hidden", 64, "--batch", 8192,
        "--lr", 0.001, "--fine-lr-scale", 0.18,
        "--cap", 0.006, "--apply-scale", 1.0,
        "--device", "auto", "--seed", 20260606, "--save-val-pred",
        "--make-test",
    ])

    _print_summary(OUT_DIR, OUT_DIR, "full")

    # Copy soft submission to canonical name
    soft_csv = OUT_DIR / "submission_boundary_tiny_soft.csv"
    final_csv = OUT_DIR / "submission.csv"
    if soft_csv.exists():
        final_csv.write_bytes(soft_csv.read_bytes())
        print(f"[setup] submission.csv ← {soft_csv.name}", flush=True)


def _print_summary(selector_out: Path, boundary_out: Path, label: str) -> None:
    sel_report = selector_out / "tcn_gru_selector_report.json"
    bnd_report = boundary_out / "boundary_tiny_correction_report.json"
    summary: dict = {"label": label}
    if sel_report.exists():
        sr = json.loads(sel_report.read_text())
        summary["selector_device"] = sr.get("device")
        summary["selector_covered_rows"] = sr.get("covered_rows")
        models = sr.get("model_oof", {})
        if "attn_gru" in models:
            soft = models["attn_gru"].get("soft", {}).get("metrics", {})
            gate = models["attn_gru"].get("argmax_soft_gate", {}).get("metrics", {})
            summary["selector_soft_hit"] = soft.get("hit")
            summary["selector_gate_hit"] = gate.get("hit")
    if bnd_report.exists():
        br = json.loads(bnd_report.read_text())
        summary["boundary_soft_hit"] = br.get("soft", {}).get("metrics", {}).get("hit")
        summary["boundary_gate_hit"] = br.get("gate", {}).get("metrics", {}).get("hit")
        summary["boundary_argmax_hit"] = br.get("argmax", {}).get("hit")
        summary["boundary_oracle_hit"] = br.get("candidate_oracle", {}).get("hit")
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="plan-004 orchestrator")
    parser.add_argument("--smoke", action="store_true", help="1-fold smoke (notebook cell 10/12 mirror)")
    args = parser.parse_args()

    ensure_data_extracted()
    if args.smoke:
        run_smoke()
    else:
        run_full()
    return 0


if __name__ == "__main__":
    sys.exit(main())
