"""plan-022 §6.2 + §6.3 — 5-fold OOF runner for 21-cell (layout × τ_cls) sweep.

CLI:
  # 단일 cell:
  python analysis/plan-022/run_oof.py --layout A1_octa7 --tau 0.001 \
         --out-json /tmp/cell.json

  # 단일 layout (3 τ):
  python analysis/plan-022/run_oof.py --layout A2_ico13 \
         --out-json analysis/plan-022/results_A2.json

  # 21-cell 전수 sweep:
  python analysis/plan-022/run_oof.py --all \
         --out-dir analysis/plan-022/
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.io import load_all_samples, load_labels                       # noqa: E402
from src.pb_0_6822.selector import stable_fold_id                       # noqa: E402

# plan-022 modules
_spec = importlib.util.spec_from_file_location("anchors_022", _THIS / "anchors.py")
anchors_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(anchors_mod)

_spec = importlib.util.spec_from_file_location("som_022", _THIS / "selector_only_model.py")
som = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(som)
# som.LgbmSelectorOnly, build_soft_label_with_tau, bf, p021_build, p021_dh

N_FOLDS = 5
R_HIT = 0.01
R_HIT_LOOSE = 0.015
TAU_SCAN = [0.001, 0.003, 0.005]


# ── helpers ────────────────────────────────────────────────────────────


def assign_folds(ids: list[str]) -> np.ndarray:
    return np.asarray([stable_fold_id(str(sid), N_FOLDS) for sid in ids], dtype=int)


# ── §6.2 per-cell 5-fold OOF ───────────────────────────────────────────


def run_oof_cell(
    X: np.ndarray,
    Y: np.ndarray,
    folds: np.ndarray,
    anchors: np.ndarray,
    tau_cls: float,
    verbose: bool = True,
) -> dict:
    """단일 (layout, τ) cell 의 5-fold OOF."""
    common = som.p021_build.build_input_common(X, som.bf.f0_baseline)
    extra = som.p021_build.build_input_lgbm_extra(X, L1=common["L1"])
    N = X.shape[0]
    X_lgbm = np.concatenate(
        [
            common["L1"].reshape(N, 99),
            common["L2"].reshape(N, 21),
            common["L4"].reshape(N, 14),
            extra,
        ],
        axis=1,
    ).astype(np.float32)
    R_wfn = common["R_wfn"]
    pred_F0 = common["pred_F0_world"]
    K = anchors.shape[0]

    pred_world = np.zeros((N, 3), dtype=np.float32)
    probs_all = np.zeros((N, K), dtype=np.float32)

    for k in range(N_FOLDS):
        tr = np.where(folds != k)[0]
        va = np.where(folds == k)[0]
        q_train = som.build_soft_label_with_tau(
            Y[tr], R_wfn[tr], pred_F0[tr], anchors, tau_cls
        )
        model = som.LgbmSelectorOnly(K=K).fit(X_lgbm[tr], q_train)
        probs = model.predict(X_lgbm[va])
        probs_all[va] = probs

        # final_frenet = Σ_a probs[a] · anchors[a]   (corrector-free)
        final_frenet = probs @ anchors                          # (n_va, 3)
        pred_world[va] = (
            np.einsum("nij,nj->ni", R_wfn[va], final_frenet) + pred_F0[va]
        )
        if verbose:
            d = np.linalg.norm(pred_world[va] - Y[va], axis=1)
            print(
                f"    fold {k}: hit@1cm={float((d <= R_HIT).mean()):.4f}",
                flush=True,
            )

    # F0 deterministic — fold-leakage 면제 (= plan-021 carry, §3.1 / §6.2 박제)
    d_cell = np.linalg.norm(pred_world - Y, axis=1)
    d_f0 = np.linalg.norm(pred_F0 - Y, axis=1)
    hit_cell_1 = float((d_cell <= R_HIT).mean())
    hit_cell_15 = float((d_cell <= R_HIT_LOOSE).mean())
    hit_f0_1 = float((d_f0 <= R_HIT).mean())
    hit_f0_15 = float((d_f0 <= R_HIT_LOOSE).mean())

    per_fold_1, per_fold_15 = [], []
    for k in range(N_FOLDS):
        m = folds == k
        per_fold_1.append(float((d_cell[m] <= R_HIT).mean()) if m.any() else 0.0)
        per_fold_15.append(
            float((d_cell[m] <= R_HIT_LOOSE).mean()) if m.any() else 0.0
        )

    delta_1 = hit_cell_1 - hit_f0_1
    delta_15 = hit_cell_15 - hit_f0_15
    max_class_ratio = float(probs_all.mean(axis=0).max())

    return {
        "K": int(K),
        "tau_cls": float(tau_cls),
        "hit_1cm": hit_cell_1,
        "hit_1.5cm": hit_cell_15,
        "delta_1cm": delta_1,
        "delta_1.5cm": delta_15,
        "max_class_ratio": max_class_ratio,
        "fold_var_1cm": float(np.std(per_fold_1)),
        "fold_var_1.5cm": float(np.std(per_fold_15)),
        "pass_both": bool(delta_1 >= 0.005 and delta_15 >= 0.005),
    }


# ── §6.3 per-sub-exp (1 layout × 3 τ_cls) ──────────────────────────────


def run_sub_exp(
    X: np.ndarray,
    Y: np.ndarray,
    folds: np.ndarray,
    layout_name: str,
    anchors: np.ndarray,
    verbose: bool = True,
) -> dict:
    """layout 의 3 τ_cls cell 측정 → dict {tau_str: cell_dict, ...}."""
    out: dict[str, Any] = {}
    for tau in TAU_SCAN:
        t0 = time.time()
        if verbose:
            print(
                f"  [{layout_name}] τ_cls={tau:.3f} K={anchors.shape[0]} ...",
                flush=True,
            )
        cell = run_oof_cell(X, Y, folds, anchors, tau, verbose=verbose)
        cell["elapsed_sec"] = float(time.time() - t0)
        if verbose:
            print(
                f"  [{layout_name}] τ_cls={tau:.3f}  "
                f"hit@1cm={cell['hit_1cm']:.4f}  hit@1.5cm={cell['hit_1.5cm']:.4f}  "
                f"Δ_1cm={cell['delta_1cm']:+.4f}  Δ_1.5cm={cell['delta_1.5cm']:+.4f}  "
                f"pass={cell['pass_both']}  "
                f"max_class={cell['max_class_ratio']:.3f}  "
                f"{cell['elapsed_sec']:.0f}s",
                flush=True,
            )
        out[f"tau_{tau:.3f}"] = cell
    return out


# ── §6.3 full sweep (21 cell) ──────────────────────────────────────────


def run_sweep(
    X: np.ndarray, Y: np.ndarray, folds: np.ndarray, verbose: bool = True
) -> dict[str, dict]:
    """7 layout × 3 τ_cls = 21 cell 전수 측정."""
    sweep: dict[str, dict] = {}
    for layout_name, anchors in anchors_mod.LAYOUT_NAMES.items():
        print(f"\n=== {layout_name} ===", flush=True)
        sweep[layout_name] = run_sub_exp(X, Y, folds, layout_name, anchors, verbose=verbose)
    return sweep


# ── CLI ────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--layout",
        default=None,
        choices=list(anchors_mod.LAYOUT_NAMES.keys()),
        help="단일 layout (3 τ_cls cell 측정). 미지정 시 --all 필수.",
    )
    ap.add_argument("--tau", type=float, default=None,
                    help="단일 cell 측정 (--layout 와 같이). 미지정 시 3 τ scan.")
    ap.add_argument("--all", action="store_true", help="21-cell 전수 sweep")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="--all 모드에서 layout 별 results_A{n}.json 자동 저장.")
    args = ap.parse_args()

    if not args.all and args.layout is None:
        ap.error("--layout <name> 또는 --all 중 하나 필요")

    t0 = time.time()
    print("[plan-022 run_oof] loading data ...", flush=True)
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    assert ids == label_ids, "ids mismatch"
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    folds = assign_folds(ids)
    print(
        f"[plan-022] N={X.shape[0]} folds={np.bincount(folds).tolist()}",
        flush=True,
    )

    if args.all:
        sweep = run_sweep(X, Y, folds)
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            for layout_name, sub_exp in sweep.items():
                sub_exp_key = layout_name.split("_")[0]
                path = args.out_dir / f"results_{sub_exp_key}.json"
                path.write_text(json.dumps(sub_exp, indent=2))
                print(f"  wrote {path}", flush=True)
        if args.out_json:
            args.out_json.write_text(json.dumps(sweep, indent=2))
            print(f"  wrote {args.out_json}", flush=True)
    elif args.tau is not None:
        anchors = anchors_mod.LAYOUT_NAMES[args.layout]
        cell = run_oof_cell(X, Y, folds, anchors, args.tau)
        print(json.dumps(cell, indent=2, default=str), flush=True)
        if args.out_json:
            args.out_json.write_text(json.dumps(cell, indent=2, default=str))
            print(f"  wrote {args.out_json}", flush=True)
    else:
        anchors = anchors_mod.LAYOUT_NAMES[args.layout]
        sub_exp = run_sub_exp(X, Y, folds, args.layout, anchors)
        if args.out_json:
            args.out_json.write_text(json.dumps(sub_exp, indent=2))
            print(f"  wrote {args.out_json}", flush=True)

    print(f"\n[plan-022 run_oof] total {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
