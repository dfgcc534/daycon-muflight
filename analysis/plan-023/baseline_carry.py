"""plan-023 §5.1 — F0 baseline reproduce + dataset hash carry from plan-022.

Run: python analysis/plan-023/baseline_carry.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.io import load_all_samples                                  # noqa: E402


def main() -> None:
    # (a) plan-020 metric carry
    baseline_path = REPO / "analysis/plan-020/baseline_oof.json"
    baseline = json.loads(baseline_path.read_text())
    f0 = baseline["f0_baseline"]
    h1cm = float(f0["hit_1cm_5fold_concat"])
    h15 = float(f0["hit_1.5cm_5fold_concat"])
    assert 0.6315 <= h1cm <= 0.6325, f"f0 1cm drift: {h1cm}"
    assert 0.8028 <= h15 <= 0.8038, f"f0 1.5cm drift: {h15}"

    # (b) plan-022 baseline_carry.json 의 dataset_hash carry
    p022_carry = json.loads(
        (REPO / "analysis/plan-022/baseline_carry.json").read_text()
    )
    expected_hash = p022_carry["dataset_hash"]

    # Hash 산식 = plan-022 §5.1 carry (sha256 of "|".join(sorted-by-str id list), first 16 hex chars)
    ids, _X = load_all_samples(split="train")
    data_hash = hashlib.sha256(
        ("|".join(sorted(map(str, ids)))).encode()
    ).hexdigest()[:16]
    assert data_hash == expected_hash, \
        f"dataset shift: {data_hash} != {expected_hash}"

    out = {
        "f0_hit_1cm": h1cm,
        "f0_hit_1.5cm": h15,
        "dataset_hash": data_hash,
        "n_samples": len(ids),
        "source_baseline": "analysis/plan-020/baseline_oof.json",
        "carry_from": "analysis/plan-022/baseline_carry.json",
    }
    (REPO / "analysis/plan-023/baseline_carry.json").write_text(
        json.dumps(out, indent=2)
    )
    print(json.dumps(out, indent=2))
    print(
        f"\nG1 PASS — f0 hit@1cm={h1cm:.4f} ∈ [0.6315, 0.6325] ✓ "
        f"AND hit@1.5cm={h15:.4f} ∈ [0.8028, 0.8038] ✓ "
        f"AND dataset_hash {data_hash} == plan-022 {expected_hash} ✓"
    )


if __name__ == "__main__":
    main()
