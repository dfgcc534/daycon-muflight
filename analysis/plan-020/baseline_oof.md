# plan-020 STAGE 1 G1 — F0 baseline 5-fold OOF reproduce

## 결과 (2026-05-18)

| metric | value | spec range | pass |
|---|---|---|---|
| `hit_1cm_5fold_concat` | **0.6320** | [0.6315, 0.6325] | ✓ |
| `hit_1.5cm_5fold_concat` | **0.8033** | [0.8028, 0.8038] | ✓ |
| `fold_variance_1cm` | 0.0052 | < 0.05 | ✓ |
| `fold_variance_1.5cm` | 0.0087 | < 0.05 | ✓ |

**G1 PASS** — plan-006/plan-014 hard evidence 와 정확 일치 (drift 0).

## Per-fold

| fold | hit@1cm | hit@1.5cm | n |
|---|---|---|---|
| 0 | 0.6401 | 0.8129 | 2020 |
| 1 | 0.6292 | 0.7953 | 2047 |
| 2 | 0.6304 | 0.8053 | 1921 |
| 3 | 0.6351 | 0.8119 | 2020 |
| 4 | 0.6250 | 0.7912 | 1992 |
| concat | 0.6320 | 0.8033 | 10000 |

## Protocol

- N = 10000 train samples (`src.io.load_all_samples(split="train")`)
- Fold split: `stable_fold_id(str(sample_id), n_folds=5)` (plan-004 `src/pb_0_6822/selector.py` L185, MD5 32-bit prefix mod 5)
- F0 산식: `f0_baseline(x, end_idx=10)` = plan-006 `frenet_par120_perp_neg020` (d1=1.98, par=1.20, perp=-0.20)
- Wall time: 0.8 s (CPU, deterministic)

→ G2.D 진입 가능 (c9).
