"""plan-022 §7 — paradigm analysis: collect_cells / select_best / marginals.

21 cell (7 layout × 3 τ_cls) 결과 집계 + best cell + layout/τ marginal.

CLI:
  python analysis/plan-022/paradigm_analysis.py \
         --out-json analysis/plan-022/paradigm_analysis.json \
         --out-md analysis/plan-022/paradigm_analysis.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve().parent

LAYOUT_IDS = [
    "A1_octa7",
    "A2_ico13",
    "A3_cubocta13",
    "A4_2shell13",
    "A5_cube8",
    "A6_bcc14",
    "A7_fib13",
]
TAU_SCAN = [0.001, 0.003, 0.005]


def collect_cells(plan_dir: Path) -> dict[str, dict]:
    """7 results_A{n}.json → flatten 21 cell dict."""
    cells: dict[str, dict] = {}
    for layout_id in LAYOUT_IDS:
        sub_exp_key = layout_id.split("_")[0]               # "A1", ...
        path = plan_dir / f"results_{sub_exp_key}.json"
        results = json.loads(path.read_text())               # {"tau_0.001": {...}, ...}
        for tau_key, cell in results.items():
            tau_str = tau_key.replace("tau_", "").replace(".", "")  # "0001", "0003", "0005"
            cell_key = f"{layout_id}_tau{tau_str[1:]}"               # drop leading "0"
            cells[cell_key] = {
                **cell,
                "layout": layout_id,
                "tau_cls": float(tau_key.replace("tau_", "")),
            }
    return cells


def select_best(cells: dict[str, dict]) -> tuple[str, dict]:
    """(pass_both, Δ_1cm + Δ_1.5cm, alphabetic) lexicographic max — deterministic."""
    best_key, best_cell = max(
        cells.items(),
        key=lambda kv: (
            kv[1]["pass_both"],
            kv[1]["delta_1cm"] + kv[1]["delta_1.5cm"],
            tuple(-ord(c) for c in kv[0]),
        ),
    )
    return best_key, best_cell


def marginals(cells: dict[str, dict]) -> dict:
    """layout-axis (per layout best τ) + τ-axis (per τ best layout) marginals."""
    # layout axis: 각 layout 의 best τ
    by_layout: dict[str, dict] = {}
    for layout_id in LAYOUT_IDS:
        layout_cells = {k: v for k, v in cells.items() if v["layout"] == layout_id}
        best_key, best_cell = select_best(layout_cells)
        by_layout[layout_id] = {**best_cell, "best_cell_key": best_key}

    # τ axis: 각 τ 의 best layout
    by_tau: dict[str, dict] = {}
    for tau in TAU_SCAN:
        tau_cells = {k: v for k, v in cells.items() if abs(v["tau_cls"] - tau) < 1e-9}
        best_key, best_cell = select_best(tau_cells)
        by_tau[f"tau_{tau:.3f}"] = {**best_cell, "best_cell_key": best_key}

    return {"by_layout": by_layout, "by_tau": by_tau}


def render_md(cells: dict[str, dict], best: tuple[str, dict],
              margins: dict, dropped: list[str]) -> str:
    """plan-022 §7 markdown report."""
    best_key, best_cell = best
    lines = [
        "# plan-022 paradigm_analysis — 21 cell sweep",
        "",
        f"F0 baseline: hit@1cm = 0.6320, hit@1.5cm = 0.8033.",
        "",
        "## 21 cell grid (layout × τ_cls)",
        "",
        "| layout | K | τ=0.001 Δ_1cm/Δ_1.5cm pass | τ=0.003 Δ_1cm/Δ_1.5cm pass | τ=0.005 Δ_1cm/Δ_1.5cm pass |",
        "|---|---|---|---|---|",
    ]
    for layout_id in LAYOUT_IDS:
        row = [layout_id, ""]
        K_val = None
        for tau in TAU_SCAN:
            tau_str = f"{tau:.3f}".replace(".", "")
            cell_key = f"{layout_id}_tau{tau_str[1:]}"
            c = cells[cell_key]
            K_val = c["K"]
            pass_mark = "✓" if c["pass_both"] else "✗"
            row.append(f"{c['delta_1cm']:+.4f} / {c['delta_1.5cm']:+.4f} {pass_mark}")
        row[1] = str(K_val)
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        f"## Best cell 🏆",
        "",
        f"- **{best_key}** (K={best_cell['K']}, τ_cls={best_cell['tau_cls']})",
        f"- hit@1cm = {best_cell['hit_1cm']:.4f}, hit@1.5cm = {best_cell['hit_1.5cm']:.4f}",
        f"- Δ_1cm = **{best_cell['delta_1cm']:+.4f}** (pass criterion +0.005)",
        f"- Δ_1.5cm = **{best_cell['delta_1.5cm']:+.4f}** (pass criterion +0.005)",
        f"- pass_both = **{best_cell['pass_both']}**",
        f"- max_class_ratio = {best_cell['max_class_ratio']:.4f}",
        f"- fold_var_1cm = {best_cell['fold_var_1cm']:.4f}, fold_var_1.5cm = {best_cell['fold_var_1.5cm']:.4f}",
        "",
        "## Layout-axis marginal (각 layout 의 best τ)",
        "",
        "| layout | best τ | Δ_1cm | Δ_1.5cm | Δ sum | pass_both | max_class |",
        "|---|---|---|---|---|---|---|",
    ]
    for layout_id, m in margins["by_layout"].items():
        s = m["delta_1cm"] + m["delta_1.5cm"]
        lines.append(
            f"| {layout_id} | {m['tau_cls']:.3f} | {m['delta_1cm']:+.4f} | "
            f"{m['delta_1.5cm']:+.4f} | {s:+.4f} | {m['pass_both']} | "
            f"{m['max_class_ratio']:.3f} |"
        )

    lines += [
        "",
        "## τ_cls-axis marginal (각 τ 의 best layout)",
        "",
        "| τ_cls | best layout | Δ_1cm | Δ_1.5cm | Δ sum | pass_both | max_class |",
        "|---|---|---|---|---|---|---|",
    ]
    for tau_key, m in margins["by_tau"].items():
        s = m["delta_1cm"] + m["delta_1.5cm"]
        lines.append(
            f"| {tau_key.replace('tau_', '')} | {m['layout']} | "
            f"{m['delta_1cm']:+.4f} | {m['delta_1.5cm']:+.4f} | {s:+.4f} | "
            f"{m['pass_both']} | {m['max_class_ratio']:.3f} |"
        )

    n_pass = sum(1 for c in cells.values() if c["pass_both"])
    lines += [
        "",
        f"## G3 PASS — {n_pass}/21 cell pass_both=True",
        "",
        f"- Dropped (soft_label_collapse): {len(dropped)} cells ({dropped if dropped else 'none'})",
        f"- Effective denominator: {21 - len(dropped)} cells",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", type=Path, default=_THIS / "paradigm_analysis.json")
    ap.add_argument("--out-md", type=Path, default=_THIS / "paradigm_analysis.md")
    args = ap.parse_args()

    cells = collect_cells(_THIS)
    print(f"[paradigm] collected {len(cells)} cells")

    # soft_label_collapse drop (max_class_ratio > 0.95)
    dropped = [k for k, c in cells.items() if c["max_class_ratio"] > 0.95]
    if dropped:
        print(f"[paradigm] dropped {len(dropped)} cells (max_class > 0.95): {dropped}")

    effective_cells = {k: v for k, v in cells.items() if k not in dropped}
    best_key, best_cell = select_best(effective_cells)
    margins = marginals(effective_cells)

    out: dict[str, Any] = {
        "cells": cells,
        "dropped_cells": dropped,
        "best_cell_key": best_key,
        "best_cell": best_cell,
        "marginals": margins,
        "n_total": len(cells),
        "n_pass_both": sum(1 for c in cells.values() if c["pass_both"]),
        "n_effective": len(effective_cells),
    }
    args.out_json.write_text(json.dumps(out, indent=2, default=str))
    print(f"[paradigm] wrote {args.out_json}")

    md = render_md(cells, (best_key, best_cell), margins, dropped)
    args.out_md.write_text(md)
    print(f"[paradigm] wrote {args.out_md}")

    print(f"\n[paradigm] best = {best_key}")
    print(f"    Δ_1cm = {best_cell['delta_1cm']:+.4f}, Δ_1.5cm = {best_cell['delta_1.5cm']:+.4f}")
    print(f"    pass_both = {best_cell['pass_both']}, sum = {best_cell['delta_1cm'] + best_cell['delta_1.5cm']:.4f}")


if __name__ == "__main__":
    main()
