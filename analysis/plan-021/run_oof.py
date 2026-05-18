"""plan-021 §5 / §6 / §7 — 5-fold OOF runner with sub-exp A/B dispatch.

CLI:
  python analysis/plan-021/run_oof.py --candidate f0_baseline --out-json analysis/plan-021/baseline_carry.json
  python analysis/plan-021/run_oof.py --candidate A_lgbm --out-json analysis/plan-021/results_lgbm.json
  python analysis/plan-021/run_oof.py --candidate B_gru --device cuda:1 --out-json analysis/plan-021/results_gru.json
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.io import load_all_samples, load_labels  # noqa: E402
from src.pb_0_6822.selector import stable_fold_id  # noqa: E402

_THIS_DIR = Path(__file__).parent
_PLAN020_DIR = REPO_ROOT / "analysis" / "plan-020"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bf = _load("baseline_f0", _PLAN020_DIR / "baseline_f0.py")
bi = _load("build_input", _THIS_DIR / "build_input.py")
dh = _load("dual_head_model", _THIS_DIR / "dual_head_model.py")


N_FOLDS = 5
END_IDX = 10
R_HIT = 0.01
R_HIT_LOOSE = 0.015


# ── helpers ────────────────────────────────────────────────────────────


def assign_folds(ids: list[str]) -> np.ndarray:
    return np.asarray([stable_fold_id(str(sid), N_FOLDS) for sid in ids], dtype=int)


def hit_rates(pred: np.ndarray, gt: np.ndarray, folds: np.ndarray) -> dict:
    d = np.linalg.norm(pred - gt, axis=1)
    per_1, per_15 = [], []
    for k in range(N_FOLDS):
        m = folds == k
        per_1.append(float((d[m] <= R_HIT).mean()) if m.any() else 0.0)
        per_15.append(float((d[m] <= R_HIT_LOOSE).mean()) if m.any() else 0.0)
    return {
        "hit_1cm_5fold_concat": float((d <= R_HIT).mean()),
        "hit_1.5cm_5fold_concat": float((d <= R_HIT_LOOSE).mean()),
        "hit_1cm_per_fold": per_1,
        "hit_1.5cm_per_fold": per_15,
        "fold_variance_1cm": float(np.std(per_1)),
        "fold_variance_1.5cm": float(np.std(per_15)),
    }


def paired_delta(pred_cand: np.ndarray, pred_f0: np.ndarray, gt: np.ndarray, R: float) -> float:
    d_c = np.linalg.norm(pred_cand - gt, axis=1)
    d_f = np.linalg.norm(pred_f0 - gt, axis=1)
    return float((d_c <= R).astype(float).mean() - (d_f <= R).astype(float).mean())


# ── F0 baseline ────────────────────────────────────────────────────────


def run_f0_baseline(X: np.ndarray, Y: np.ndarray, folds: np.ndarray) -> tuple[dict, np.ndarray]:
    pred = bf.f0_baseline(X, end_idx=END_IDX)
    metrics = hit_rates(pred, Y, folds)
    metrics["candidate"] = "f0_baseline"
    metrics["n_samples"] = int(X.shape[0])
    return metrics, pred


# ── sub-exp A: LGBM ────────────────────────────────────────────────────


def run_oof_lgbm(X: np.ndarray, Y: np.ndarray, folds: np.ndarray, pred_f0: np.ndarray, verbose: bool = True) -> dict:
    """5-fold OOF for LGBM dual head. X (N, 11, 3)."""
    common = bi.build_input_common(X, bf.f0_baseline)
    extra = bi.build_input_lgbm_extra(X, L1=common["L1"])
    N = X.shape[0]
    X_lgbm = np.concatenate([
        common["L1"].reshape(N, 99),
        common["L2"].reshape(N, 21),
        common["L4"].reshape(N, 14),
        extra,
    ], axis=1).astype(np.float32)
    R_wfn = common["R_wfn"]
    pred_F0 = common["pred_F0_world"]              # v1.3 — anchor reference

    pred_world = np.zeros((N, 3), dtype=np.float32)
    for k in range(N_FOLDS):
        train_idx = np.where(folds != k)[0]
        val_idx = np.where(folds == k)[0]

        # v1.3 — residual reference = pred_F0_world (F0 의 80ms 미래)
        q_train = bi.build_soft_label(Y[train_idx], R_wfn[train_idx], pred_F0[train_idx])
        residual_true_frenet_train = bi.to_frenet(Y[train_idx], R_wfn[train_idx], pred_F0[train_idx])
        residual_targets_train = (
            residual_true_frenet_train[:, None, :] - bi.ANCHORS_FRENET[None, :, :]
        )

        model = dh.LgbmDualHead()
        model.fit(X_lgbm[train_idx], q_train, residual_targets_train)
        probs, reg_offset = model.predict(X_lgbm[val_idx])
        combined = bi.ANCHORS_FRENET[None] + reg_offset
        final_frenet = (probs[:, :, None] * combined).sum(axis=1)
        # v1.3 — final_world = pred_F0_world + R_wfn @ Frenet mixture
        pred_world[val_idx] = (
            np.einsum("nij,nj->ni", R_wfn[val_idx], final_frenet) + pred_F0[val_idx]
        )
        if verbose:
            d = np.linalg.norm(pred_world[val_idx] - Y[val_idx], axis=1)
            print(f"  [A_lgbm] fold {k}: val_hit@1cm={float((d <= R_HIT).mean()):.4f}", flush=True)

    metrics = hit_rates(pred_world, Y, folds)
    metrics["candidate"] = "A_lgbm"
    metrics["n_samples"] = N
    metrics["delta_1cm"] = paired_delta(pred_world, pred_f0, Y, R_HIT)
    metrics["delta_1.5cm"] = paired_delta(pred_world, pred_f0, Y, R_HIT_LOOSE)
    metrics["pass_both"] = bool(metrics["delta_1cm"] >= 0.005 and metrics["delta_1.5cm"] >= 0.005)
    return metrics


# ── sub-exp B: GRU ─────────────────────────────────────────────────────


def run_oof_gru(
    X: np.ndarray, Y: np.ndarray, folds: np.ndarray, pred_f0: np.ndarray,
    seeds: list[int], device: str = "cuda:1", epochs: int = 50, batch_size: int = 256,
    lr: float = 1e-3, weight_decay: float = 1e-4, early_stop_patience: int = 10,
    verbose: bool = True,
) -> dict:
    import torch

    if torch.cuda.is_available() and device.startswith("cuda"):
        try:
            dev = torch.device(device)
            torch.zeros(1, device=dev)
        except (RuntimeError, AssertionError):
            dev = torch.device("cuda:0") if torch.cuda.device_count() else torch.device("cpu")
    else:
        dev = torch.device("cpu")
    if verbose:
        print(f"  [B_gru] device = {dev}", flush=True)

    common = bi.build_input_common(X, bf.f0_baseline)
    N = X.shape[0]
    seq_all = torch.from_numpy(common["L1"]).float()                                    # (N, 11, 9)
    flat_all = torch.from_numpy(
        np.concatenate([common["L2"].reshape(N, 21), common["L4"].reshape(N, 14)], axis=1)
    ).float()                                                                            # (N, 35)
    R_all = torch.from_numpy(common["R_wfn"]).float()                                    # (N, 3, 3)
    pred_F0_all = torch.from_numpy(common["pred_F0_world"]).float()                      # (N, 3) — v1.3
    Y_t = torch.from_numpy(Y.astype(np.float32))                                         # (N, 3)
    q_all = torch.from_numpy(
        bi.build_soft_label(Y, common["R_wfn"], common["pred_F0_world"])                 # v1.3
    ).float()                                                                            # (N, 7)

    pred_world = np.zeros((N, 3), dtype=np.float32)
    train_hit_log: dict[int, Any] = {}

    for k in range(N_FOLDS):
        train_idx = np.where(folds != k)[0]
        val_idx = np.where(folds == k)[0]
        best_seed = None
        best_train_hit = -1.0
        best_state = None
        best_epoch = -1
        seed_log: dict[int, dict] = {}

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            model = dh.GRUDualHead().to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            seq_tr = seq_all[train_idx].to(dev)
            flat_tr = flat_all[train_idx].to(dev)
            R_tr = R_all[train_idx].to(dev)
            pf0_tr = pred_F0_all[train_idx].to(dev)              # v1.3
            Y_tr = Y_t[train_idx].to(dev)
            q_tr = q_all[train_idx].to(dev)

            N_tr = len(train_idx)
            train_hit_history = []
            plateau = 0
            best_epoch_seed = 0
            best_train_hit_seed = -1.0
            best_state_seed = None

            for epoch in range(epochs):
                model.train()
                perm = torch.randperm(N_tr, device=dev)
                for i in range(0, N_tr, batch_size):
                    idx = perm[i:i + batch_size]
                    logits, reg_off = model(seq_tr[idx], flat_tr[idx])
                    probs = torch.softmax(logits, dim=1)
                    combined = model.ANCHORS[None] + reg_off
                    final_frenet = (probs[:, :, None] * combined).sum(dim=1)
                    final_world = torch.einsum("nij,nj->ni", R_tr[idx], final_frenet) + pf0_tr[idx]
                    tau, ub = dh.tau_for_epoch(epoch)
                    loss = dh.soft_ce_loss(logits, q_tr[idx]) + dh.smooth_hit_loss(
                        final_world, Y_tr[idx], tau=tau, use_boundary=ub
                    )
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                # train hit
                model.eval()
                with torch.no_grad():
                    train_preds = []
                    for i in range(0, N_tr, batch_size):
                        seg = slice(i, i + batch_size)
                        final = model.predict_world(seq_tr[seg], flat_tr[seg], R_tr[seg], pf0_tr[seg])
                        train_preds.append(final)
                    train_pred = torch.cat(train_preds, dim=0)
                d_tr = (train_pred - Y_tr).norm(dim=1)
                train_hit = float((d_tr <= R_HIT).float().mean().item())
                train_hit_history.append(train_hit)

                if train_hit > best_train_hit_seed + 1e-5:
                    best_train_hit_seed = train_hit
                    best_epoch_seed = epoch
                    best_state_seed = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
                    plateau = 0
                else:
                    plateau += 1
                    if plateau >= early_stop_patience:
                        break

            seed_log[seed] = {
                "best_train_hit": best_train_hit_seed,
                "best_epoch": best_epoch_seed,
                "stopped_epoch": epoch,
            }
            if best_train_hit_seed > best_train_hit:
                best_train_hit = best_train_hit_seed
                best_seed = seed
                best_state = best_state_seed
                best_epoch = best_epoch_seed

        train_hit_log[k] = {"seeds": seed_log, "best_seed": best_seed, "best_epoch": best_epoch}
        if verbose:
            print(f"  [B_gru] fold {k}: best_seed={best_seed} epoch={best_epoch} train_hit={best_train_hit:.4f}", flush=True)

        # eval best on val_k
        model = dh.GRUDualHead().to(dev)
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            val_preds = []
            for i in range(0, len(val_idx), batch_size):
                vi = val_idx[i:i + batch_size]
                final = model.predict_world(
                    seq_all[vi].to(dev), flat_all[vi].to(dev),
                    R_all[vi].to(dev), pred_F0_all[vi].to(dev),
                )
                val_preds.append(final.cpu().numpy())
            pred_world[val_idx] = np.concatenate(val_preds, axis=0)

    metrics = hit_rates(pred_world, Y, folds)
    metrics["candidate"] = "B_gru"
    metrics["n_samples"] = N
    metrics["delta_1cm"] = paired_delta(pred_world, pred_f0, Y, R_HIT)
    metrics["delta_1.5cm"] = paired_delta(pred_world, pred_f0, Y, R_HIT_LOOSE)
    metrics["pass_both"] = bool(metrics["delta_1cm"] >= 0.005 and metrics["delta_1.5cm"] >= 0.005)
    metrics["train_hit_log"] = train_hit_log
    return metrics


# ── CLI ────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True,
                    help="f0_baseline | A_lgbm | B_gru | all")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[20260518, 20260519, 20260520])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[plan-021 run_oof] loading data ...", flush=True)
    ids, X = load_all_samples(split="train")
    label_ids, Y = load_labels()
    assert ids == label_ids
    X = X.astype(np.float64); Y = Y.astype(np.float64)
    folds = assign_folds(ids)
    print(f"[plan-021] N={X.shape[0]} folds={np.bincount(folds).tolist()}", flush=True)

    f0_metrics, pred_f0 = run_f0_baseline(X, Y, folds)
    print(f"[plan-021] F0: hit@1cm={f0_metrics['hit_1cm_5fold_concat']:.4f} hit@1.5cm={f0_metrics['hit_1.5cm_5fold_concat']:.4f}", flush=True)

    results: dict[str, Any] = {"f0_baseline": f0_metrics}

    if args.candidate in ("A_lgbm", "all"):
        print("\n[plan-021] === A_lgbm ===", flush=True)
        t_sub = time.time()
        results["A_lgbm"] = run_oof_lgbm(X, Y, folds, pred_f0)
        print(f"  [A_lgbm] elapsed {time.time() - t_sub:.1f}s", flush=True)

    if args.candidate in ("B_gru", "all"):
        print("\n[plan-021] === B_gru ===", flush=True)
        t_sub = time.time()
        results["B_gru"] = run_oof_gru(
            X, Y, folds, pred_f0,
            seeds=args.seeds, device=args.device, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
        )
        print(f"  [B_gru] elapsed {time.time() - t_sub:.1f}s", flush=True)

    print(f"\n[plan-021 run_oof] total {time.time() - t0:.1f}s", flush=True)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2, default=str))
        print(f"[plan-021] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
