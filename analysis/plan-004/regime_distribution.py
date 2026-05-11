"""plan-004 c9 / G3.5 — 18-regime × 27-candidate distribution audit.

Reconstructs (audit-side):
 (a) regime-level train sample histogram (18 bins)
 (b) (regime, candidate) hit-rate marginalize (18 × 27 table)
 (c) degenerate regime flag (sample < 50 — empirical Bayes shrinkage threshold)
 (d) hyper-specialized cell flag
       positive: ratio = hit[r,c] / regime_mean[r] >= 1.5
       negative: ratio <= 0.5 AND hit[r,c] >= 0.01 (low-end floor to cut random-zero noise)

Spec @ plans/plan-004-pb-0-6822-fullrun.md §8.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pb_0_6822 import selector

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
ANALYSIS_DIR = REPO / "analysis/plan-004"


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    ids, train_y = selector.read_labels(DATA_ROOT / "train_labels.csv")
    train_x = selector.load_stack(DATA_ROOT / "train", ids)
    end_idx = train_x.shape[1] - 1
    n_total = len(ids)
    print(f"[load] n_total={n_total} end_idx={end_idx} train_y.shape={train_y.shape}")

    bins = selector.fit_regime_bins(train_x, end_idx)
    regimes = selector.assign_regimes(train_x, end_idx, bins)
    hist = np.bincount(regimes, minlength=18)
    assert hist.sum() == n_total, f"hist sum {hist.sum()} != n_total {n_total}"

    cands = selector.make_candidates(train_x, end_idx, horizon=2)
    err = np.linalg.norm(cands - train_y[:, None, :], axis=2)
    R_HIT = selector.R_HIT

    hit_table = np.zeros((18, 27), dtype=np.float64)
    mean_dist_table = np.zeros((18, 27), dtype=np.float64)
    for r in range(18):
        mask = regimes == r
        if mask.sum() > 0:
            hit_table[r] = (err[mask] <= R_HIT).mean(axis=0)
            mean_dist_table[r] = err[mask].mean(axis=0)

    degenerate_regimes = [int(r) for r in range(18) if hist[r] < 50]
    regime_means = hit_table.mean(axis=1)
    hyper_cells: list[dict] = []
    for r in range(18):
        if regime_means[r] <= 0:
            continue
        for c in range(27):
            ratio = hit_table[r, c] / regime_means[r]
            if ratio >= 1.5 or (ratio <= 0.5 and hit_table[r, c] >= 0.01):
                hyper_cells.append({
                    "regime": int(r),
                    "candidate": int(c),
                    "candidate_name": selector.CANDIDATES[c].name,
                    "hit_rate": float(hit_table[r, c]),
                    "regime_mean": float(regime_means[r]),
                    "ratio": float(ratio),
                })

    summary = {
        "n_total": int(n_total),
        "R_HIT": float(R_HIT),
        "regime_count": 18,
        "candidate_count": 27,
        "regime_histogram": [int(h) for h in hist],
        "regime_bin_edges": {k: [float(x) for x in v] for k, v in bins.items()},
        "candidate_names": [c.name for c in selector.CANDIDATES],
        "hit_table": hit_table.tolist(),
        "mean_dist_table": mean_dist_table.tolist(),
        "regime_means": regime_means.tolist(),
        "degenerate_regimes": degenerate_regimes,
        "degenerate_count": len(degenerate_regimes),
        "hyper_specialized_cells": hyper_cells,
        "hyper_specialized_count": len(hyper_cells),
    }

    (ANALYSIS_DIR / "regime_distribution.json").write_text(json.dumps(summary, indent=2))
    print(f"[save] {ANALYSIS_DIR / 'regime_distribution.json'}")
    write_markdown(summary)
    print(f"[save] {ANALYSIS_DIR / 'regime_distribution.md'}")


def write_markdown(summary: dict) -> None:
    hist = summary["regime_histogram"]
    cand_names = summary["candidate_names"]
    hit_table = summary["hit_table"]
    regime_means = summary["regime_means"]

    lines: list[str] = []
    lines.append("# Regime × Candidate Distribution (plan-004)\n")
    lines.append(
        f"- `n_total` = **{summary['n_total']}** train samples\n"
        f"- `R_HIT` = **{summary['R_HIT']} m** (1 cm)\n"
        f"- 18 regime × 27 candidate grid\n"
        f"- `degenerate_count` = **{summary['degenerate_count']}** (sample < 50)\n"
        f"- `hyper_specialized_count` = **{summary['hyper_specialized_count']}** cells\n"
    )

    # regime histogram
    lines.append("\n## §1. Regime Histogram (sample counts)\n")
    lines.append("Regime encoding: `regime = speed_bin*6 + curve_bin*2 + fatigue_bin`")
    lines.append(f"- speed bins: {summary['regime_bin_edges'].get('speed')}")
    lines.append(f"- curvature bins: {summary['regime_bin_edges'].get('curvature')}")
    lines.append(f"- speed_slope bins: {summary['regime_bin_edges'].get('speed_slope')}\n")
    lines.append("| regime | sample_count | regime_mean_hit |")
    lines.append("|--------|-------------:|---------------:|")
    for r in range(18):
        lines.append(f"| {r:>6} | {hist[r]:>12} | {regime_means[r]:.4f} |")

    # 18×27 hit-rate table
    lines.append("\n## §2. 18×27 Hit-Rate Table\n")
    header = "| regime | " + " | ".join(f"c{c:02d}" for c in range(27)) + " |"
    sep = "|--------|" + "|".join(["-----:"] * 27) + "|"
    lines.append(header)
    lines.append(sep)
    for r in range(18):
        row_vals = " | ".join(f"{hit_table[r][c]:.3f}" for c in range(27))
        lines.append(f"| {r:>6} | {row_vals} |")

    lines.append("\n**Candidate index legend:**\n")
    lines.append("| c## | name |")
    lines.append("|----:|------|")
    for c, name in enumerate(cand_names):
        lines.append(f"| c{c:02d} | `{name}` |")

    # degenerate
    lines.append("\n## §3. Degenerate Regimes (sample < 50)\n")
    if summary["degenerate_regimes"]:
        lines.append("| regime | sample_count |")
        lines.append("|-------:|-------------:|")
        for r in summary["degenerate_regimes"]:
            lines.append(f"| {r:>6} | {hist[r]:>12} |")
    else:
        lines.append("_None — all 18 regimes satisfy empirical Bayes shrinkage threshold._")

    # hyper-specialized
    lines.append("\n## §4. Hyper-Specialized Cells\n")
    if summary["hyper_specialized_cells"]:
        lines.append(
            "Criterion: `ratio = hit[r,c] / regime_mean[r]` >= 1.5 (positive over-specialization) "
            "OR `ratio <= 0.5 AND hit[r,c] >= 0.01` (negative under-specialization, low-end floor)\n"
        )
        lines.append("| regime | candidate | name | hit_rate | regime_mean | ratio |")
        lines.append("|-------:|----------:|------|---------:|------------:|------:|")
        for cell in summary["hyper_specialized_cells"]:
            lines.append(
                f"| {cell['regime']:>6} | {cell['candidate']:>9} | `{cell['candidate_name']}` | "
                f"{cell['hit_rate']:.4f} | {cell['regime_mean']:.4f} | {cell['ratio']:.2f} |"
            )
    else:
        lines.append("_None — no cell exceeds the specialization thresholds._")

    (ANALYSIS_DIR / "regime_distribution.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
