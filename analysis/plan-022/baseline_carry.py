"""plan-022 §5.1 — F0 baseline carry + dataset hash sanity.

plan-020 baseline_oof.json 의 hit metric 그대로 carry (재학습 X). dataset hash
도 비교 (mismatch → halt). 산출 = analysis/plan-022/baseline_carry.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent      # depth 3
sys.path.insert(0, str(REPO))

from src.io import load_all_samples                        # noqa: E402


def main() -> None:
    # (a) plan-020 metric carry
    baseline_path = REPO / "analysis" / "plan-020" / "baseline_oof.json"
    baseline = json.loads(baseline_path.read_text())
    f0 = baseline["f0_baseline"]
    h1 = f0["hit_1cm_5fold_concat"]
    h15 = f0["hit_1.5cm_5fold_concat"]
    assert 0.6315 <= h1 <= 0.6325, f"f0 1cm drift: {h1}"
    assert 0.8028 <= h15 <= 0.8038, f"f0 1.5cm drift: {h15}"

    # (b) dataset hash sanity
    ids, _ = load_all_samples(split="train")
    data_hash = hashlib.sha256(
        ("|".join(sorted(map(str, ids)))).encode()
    ).hexdigest()[:16]

    expected_hash = baseline.get("dataset_hash")
    if expected_hash is not None:
        assert data_hash == expected_hash, \
            f"dataset shift: {data_hash} != {expected_hash}"
        hash_source = "plan-020 baseline_oof.json"
    else:
        # legacy fallback — plan-020 미박제. 본 plan-022 가 신규 박제, plan-023+ carry.
        hash_source = "plan-022 baseline_carry.json (legacy seed)"

    out = {
        "f0_hit_1cm": float(h1),
        "f0_hit_1.5cm": float(h15),
        "dataset_hash": data_hash,
        "dataset_hash_source": hash_source,
        "n_samples": int(len(ids)),
        "source_baseline": "analysis/plan-020/baseline_oof.json",
    }
    out_path = REPO / "analysis" / "plan-022" / "baseline_carry.json"
    out_path.write_text(json.dumps(out, indent=2))

    print(f"[plan-022 G1] hit@1cm = {h1:.4f}  hit@1.5cm = {h15:.4f}  ✓", flush=True)
    print(f"[plan-022 G1] dataset_hash = {data_hash}  ({hash_source})", flush=True)
    print(f"[plan-022 G1] N = {len(ids)}", flush=True)
    print(f"[plan-022 G1] wrote {out_path.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
