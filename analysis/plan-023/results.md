---
plan_id: 023
date: 2026-05-19 (Asia/Seoul)
status: all_complete
best_sub_exp: B4_fib50_tau001
best_hit_1cm: 0.6532
best_hit_1.5cm: 0.8108
best_delta_1cm: +0.0212
best_delta_1.5cm: +0.0075
best_delta_sum: +0.0287
band: positive
based_on:
  - 022 (A6_bcc14_tau001, sum 0.0279)
dataset_hash: b91502db94fab67d
n_samples: 10000
---

# plan-023 results — Large-N Anchor Layout Sweep (K > 14)

## 1. plan-022 → plan-023 narrative bridge

plan-022 winner = **A6_bcc14_tau001** (K=14, BCC, τ=0.001 sharp): Δ_1cm +0.0208 / Δ_1.5cm +0.0071 / sum 0.0279. K∈{7,8,13,14} 범위에서 saturate 미확인. 사용자 narrative "14N+" 세션 = K > 14 영역의 추가 lever 측정 요청.

plan-023 = K∈{20, 24, 30, 50} 정 N면체 vertex (B1 dodecahedron / B2 truncated octahedron / B3 icosidodecahedron / B4 Fibonacci spiral) × τ_cls ∈ {0.001, 0.003, 0.005} = 12 cell 전수 measurement, model/input/fold/soft-label/runner 100% plan-022 carry.

**핵심 발견**: K=14 → K=20~30 = **plateau** (sum 0.027x 수렴) → K=50 = **revival** (sum 0.0287 = plan-022 갱신 +0.0008). 즉, large-N lever 는 살아있지만 K∈{20,30} 영역에서는 dormant 하다가 K=50 의 quasi-uniform 분포로 깨어남.

## 2. 12 cell paired Δ table (layout × τ_cls grid)

| layout | K | τ=0.001 Δ_1cm/Δ_1.5cm pass | τ=0.003 Δ_1cm/Δ_1.5cm pass | τ=0.005 Δ_1cm/Δ_1.5cm pass |
|---|---|---|---|---|
| B1_dodeca20      | 20 | +0.0193 / +0.0080 ✓ | +0.0124 / +0.0054 ✓ | +0.0081 / +0.0039 ✗ |
| B2_trunc_octa24  | 24 | +0.0200 / +0.0072 ✓ | +0.0122 / +0.0057 ✓ | +0.0083 / +0.0036 ✗ |
| B3_icosidodec30  | 30 | +0.0199 / +0.0077 ✓ | +0.0123 / +0.0057 ✓ | +0.0081 / +0.0040 ✗ |
| **B4_fib50** 🏆  | **50** | **+0.0212 / +0.0075 ✓** | +0.0124 / +0.0054 ✓ | +0.0076 / +0.0041 ✗ |

8/12 cell PASS_BOTH (paired Δ ≥ +0.005 둘 다). drop 0 (모든 cell soft_label_collapse 회피).

## 3. Best cell 🏆

**B4_fib50_tau001** (K=50, τ_cls=0.001m):

| metric | value | vs plan-022 A6_bcc14_tau001 |
|---|---|---|
| hit@1cm | 0.6532 | A6: 0.6528, +0.0004 |
| hit@1.5cm | 0.8108 | A6: 0.8104, +0.0004 |
| Δ_1cm | +0.0212 | A6: +0.0208, +0.0004 |
| Δ_1.5cm | +0.0075 | A6: +0.0071, +0.0004 |
| Δ sum | +0.0287 | A6: +0.0279, **+0.0008** ✓ plan-022 갱신 |
| max_class_ratio | 0.0316 | A6: 0.105 (-0.073, 70% drop) |
| fold_var_1cm | 0.0046 | informational |
| fold_var_1.5cm | 0.0070 | informational |
| pass_both | True | A6: True |

## 4. Layout-axis marginal (K trend, each layout's best τ)

| layout | K | best τ | Δ sum | pass_both | max_class | uniform 1/K | ratio |
|---|---|---|---|---|---|---|---|
| A1_octa7 (plan-022) | 7 | 0.001 | +0.0262 | True | 0.232 | 0.143 | 1.62 |
| A6_bcc14 (plan-022) | 14 | 0.001 | **+0.0279** | True | 0.105 | 0.071 | 1.48 |
| B1_dodeca20 | 20 | 0.001 | +0.0273 | True | 0.073 | 0.050 | 1.45 |
| B2_trunc_octa24 | 24 | 0.001 | +0.0272 | True | 0.066 | 0.042 | 1.58 |
| B3_icosidodec30 | 30 | 0.001 | +0.0276 | True | 0.050 | 0.033 | 1.50 |
| **B4_fib50** | **50** | 0.001 | **+0.0287** | True | 0.032 | 0.020 | 1.58 |

**K trend 해석**: K=7 → 14 = 향상 (+0.0017), K=14 → 20~30 = plateau (-0.0007 ~ -0.0003), K=30 → 50 = revival (+0.0011). plateau 영역의 원인 = "polyhedron vertex set 의 angular density 가 충분히 fine grid 도달, 새 lever 없음". K=50 revival 원인 = "Fibonacci spiral quasi-uniform 이 polyhedron-style edge symmetry 의 *bias-free* 분포 제공 → uniform-baseline 대비 1.58x 의 distributed 상태 유지하면서 fine grid 효과".

## 5. τ_cls-axis marginal (each τ's best layout)

| τ_cls | best layout | K | Δ sum | pass_both |
|---|---|---|---|---|
| 0.001 | **B4_fib50** | 50 | **+0.0287** | True |
| 0.003 | B3_icosidodec30 | 30 | +0.0180 | True |
| 0.005 | B3_icosidodec30 | 30 | +0.0121 | False |

τ=0.001 sharp 가 모든 layout 에서 dominant (plan-022 carry trend). τ=0.005 loose 는 4 layout 모두 fail — paired Δ_1.5cm < +0.005 한계.

## 6. Mode collapse 완화 finding (K↑ effect)

| K | layout | max_class_ratio (τ=0.001) | uniform 1/K | ratio_to_uniform | distributed level |
|---|---|---|---|---|---|
| 7 | A1_octa7 (plan-022) | 0.232 | 0.143 | 1.62 | most collapsed |
| 14 | A6_bcc14 (plan-022) | 0.105 | 0.071 | 1.48 | mid |
| 20 | B1_dodeca20 | 0.073 | 0.050 | 1.45 | mid |
| 24 | B2_trunc_octa24 | 0.066 | 0.042 | 1.58 | mid |
| 30 | B3_icosidodec30 | 0.050 | 0.033 | 1.50 | distributed |
| **50** | **B4_fib50** | **0.032** | **0.020** | **1.58** | **most distributed** |

**H3 verification**: K=50 τ=0.001 의 `max_class_ratio = 0.0316` ≤ K=20 τ=0.001 의 `max_class_ratio = 0.0727 × 0.5 = 0.0364`. **H3 PASS ✓** — K↑ 시 effective temperature 변화로 자연 distributed 화 확인. ratio_to_uniform 은 모든 K 에서 1.45~1.62 범위 (uniform 대비 ~1.5x mass concentration) 유지 — 즉 absolute 값은 K↑ 시 1/K 비례 감소하지만 relative 분산은 fixed.

## 7. plan-022 best 대비 향상 cell

| cell | K | Δ_1cm | Δ_1.5cm | sum | vs plan-022 A6 sum |
|---|---|---|---|---|---|
| **B4_fib50_tau001** | 50 | +0.0212 | +0.0075 | **+0.0287** | **+0.0008** |

12 cell 중 sum > 0.0279 = **1/12** (B4_fib50_tau001 only).
12 cell 중 Δ_1cm > 0.0208 = **1/12** (B4_fib50_tau001).
12 cell 중 Δ_1.5cm > 0.0071 = **4/12** (B1_tau001 0.0080 / B3_tau001 0.0077 / B4_tau001 0.0075 / A6_tau001 0.0071 ↑).

**Δ_1.5cm 만 보면 4 cell 이 plan-022 갱신** — 1.5cm hit-zone coverage 는 large-N 영역에서 광범위 개선 신호. 단, Δ_1cm 동시 갱신은 K=50 만 가능 (B1/B2/B3 의 K↑ effect 가 1cm 메트릭에서는 plateau 임을 재확인).

## 8. Paradigm-level finding (K>14 lever effect)

**핵심**: K>14 lever는 saturate 아니지만 **K=20~30 plateau + K=50 revival** 의 비단조 함수. plan-022 의 K=14 winner 가 K=20~30 sweep 의 best 와 동등하다는 것은 "Platonic/Archimedean vertex set 의 angular density 만 늘리는 lever 가 K=14 이상에서 zero gain" 의미.

K=50 revival 의 잠재 mechanism (추가 검증 필요):
1. **Fibonacci quasi-uniform 분포 효과**: B4 만 vertex-transitive 아니라 jittered uniform. polyhedron-style symmetry breaking 이 LGBM K-class softmax 의 *bias mode* 를 약화시켜 1cm 정확도 +0.0004 추가 lever 제공.
2. **N samples per class lower bound**: K=50 시 samples/class = 10000/50 = 200 — `samples_per_class_low` warn floor. 단 모든 cell finite + max_class < 0.95 → samples/class 통계 충분.
3. **effective grid spacing**: K=50 fib50 인접 anchor 각도 ~14° vs K=14 BCC 인접 anchor ~45°. 1.5cm hit-zone (각도 등가 ~17°) 해상도 K=14 미달, K=50 충분.

**판단**: K=50 fib50 이 K=14 BCC 대비 +0.0008 sum 향상 = 작지만 분명한 lever 존재. 다만 단일 plan 의 N=10000 sample 위 noise 한계 (fold_var_1cm ~ 0.005) 안 마진이라 **추가 확증 필요 (follow-up plan-024/025)**.

## 9. Follow-up plan 후보

- **plan-024 (가칭)**: B4_fib50_tau001 winner 위 **corrector reg head 재투입** — plan-021 의 dead head 가 K=50 fib50 layout 위에선 살아날 가능성 검증 (plan-022 followed_by 슬롯의 corrector ablation 을 plan-023 best 위로 carry).
- **plan-025 (가칭)**: **N × radius shell 2D sweep** — 본 plan single shell 0.005m 한계. radius 0.003 / 0.005 / 0.0075m × N ∈ {30, 50, 100} 측정. radial lever 가 K↑ 와 결합 시 추가 향상 여지.
- **plan-026 (가칭, lower priority)**: **N > 50 진행** — N=72 (geodesic icosahedron 2-freq) / N=100 fib 측정. samples/class < 200 위험 zone 진입, mode collapse 위험 / runtime budget 위험 박제 필요.

## 10. Severe / warn 박제

- **`all_negative` warn**: 미발동 — 8/12 cell paired Δ ≥ +0.005 둘 다 통과.
- **`soft_label_collapse` warn**: 0 cell drop — 모든 max_class_ratio 0.032~0.073 << 0.95.
- **`samples_per_class_low` warn**: K=50 cell 의 samples/class = 200 — floor 정확 도달. metric finite + max_class 정상 → spec 안 (warn 조건은 `samples/class < 200`, equality 통과).
- **`lgbm_numerical` severe**: 미발동 — 12 cell 모두 metric finite.
- **`f0_reproduce_drift` severe**: G1 0.6320 / 0.8033 정확 carry, ±0.0005 안.

## 11. Dataset hash + reproducibility 박제

- `dataset_hash = b91502db94fab67d` (plan-020 / 021 / 022 carry 정확 일치, sha256 of "|".join(sorted(map(str, ids)))[:16] from 10000 train ids)
- `n_samples = 10000`
- 5-fold split: `stable_fold_id(str(sample_id), 5)` (MD5 prefix mod 5), fold distribution = [2020, 2047, 1921, 2020, 1992]
- LGBM hparam: n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=20260519, objective=multiclass
- 12 cell deterministic re-run 시 동일 결과 (no seed except LGBM internal).
- Total runtime: G2.B1 1093s + G2.B2 1439s + G2.B3 2004s + G2.B4 4683s = 9219s ≈ 154 min ≈ 2.5h. parallel 4-process oversubscription (load avg 192, 96 core 2x over) 시도 후 serial 전환 박제 (decision-note c6 carry).
