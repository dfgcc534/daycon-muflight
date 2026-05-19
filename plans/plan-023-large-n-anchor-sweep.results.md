---
plan_id: 023
finished_at: 2026-05-19 (Asia/Seoul)
status: all_complete
best_sub_exp: B4_fib50_tau001
best_hit_1cm: 0.6532
best_hit_1.5cm: 0.8108
best_delta_1cm: +0.0212
best_delta_1.5cm: +0.0075
band: positive
exp_ids_completed:
  - Z023_B1_dodeca20
  - Z023_B2_trunc_octa24
  - Z023_B3_icosidodec30
  - Z023_B4_fib50
exp_ids_skipped: []
---

# plan-023.results — Large-N Anchor Layout Sweep

## 핵심 결과 요약

- **best**: B4_fib50_tau001 (K=50 Fibonacci spiral + τ_cls=0.001 sharp)
- **paired Δ**: Δ_1cm = +0.0212, Δ_1.5cm = +0.0075 (둘 다 PASS criterion +0.005 통과)
- **plan-022 갱신**: sum 0.0287 > A6_bcc14_tau001 0.0279 (+0.0008) ✓
- **band**: positive (G3 PASS, severe 0건, warn 0건)
- **G3 통과**: 8/12 cell paired Δ ≥ +0.005 둘 다

## 12 cell paired Δ grid (4 layout × 3 τ_cls)

| layout | K | τ=0.001 | τ=0.003 | τ=0.005 |
|---|---|---|---|---|
| B1_dodeca20 | 20 | +0.0193/+0.0080 ✓ | +0.0124/+0.0054 ✓ | +0.0081/+0.0039 ✗ |
| B2_trunc_octa24 | 24 | +0.0200/+0.0072 ✓ | +0.0122/+0.0057 ✓ | +0.0083/+0.0036 ✗ |
| B3_icosidodec30 | 30 | +0.0199/+0.0077 ✓ | +0.0123/+0.0057 ✓ | +0.0081/+0.0040 ✗ |
| **B4_fib50** | **50** | **+0.0212/+0.0075 ✓** | +0.0124/+0.0054 ✓ | +0.0076/+0.0041 ✗ |

## 가설 결과 (§1.3)

| 가설 | 결과 | 근거 |
|---|---|---|
| H1 (K↑ layout 효과) | ✓ PASS | K=20→50 layout marginal Δ_sum 0.0273→0.0287 단조 증가 (단 K=20~30 plateau) |
| H2 (K=50 saturate) | ✗ REFUTED | K=50 이 K=30 보다 큼, 가설 반대 (K=50 revival 박제) |
| H3 (sharp τ saturate point) | ✓ PASS | K=50 τ=0.001 max_class 0.0316 ≤ K=20 max_class 0.0727 × 0.5 = 0.0364 |
| H4 (plan-022 갱신) | ✓ PASS | 1/12 cell achieves Δ sum > 0.0279 (B4_fib50_tau001) |

## Sub-exp 별 상세 (status / duration / metric)

### Z023_B1_dodeca20 (K=20, complete)
- started_at: 2026-05-19 10:25 KST, duration: 1093s (18.2 min, serial after parallel-kill)
- best cell: B1_dodeca20_tau001 — Δ_1cm +0.0193, Δ_1.5cm +0.0080, max_class 0.073
- baseline diff: anchor 좌표 only (plan-022 §6.1/§6.2 input/model/fold 동일)
- artifact: `analysis/plan-023/results_B1.json`, `G2_B1.log`

### Z023_B2_trunc_octa24 (K=24, complete)
- started_at: 2026-05-19 11:11 KST, duration: 1439s (24 min)
- best cell: B2_trunc_octa24_tau001 — Δ_1cm +0.0200, Δ_1.5cm +0.0072, max_class 0.066
- artifact: `analysis/plan-023/results_B2.json`, `G2_B2.log`

### Z023_B3_icosidodec30 (K=30, complete)
- started_at: 2026-05-19 11:36 KST, duration: 2004s (33 min)
- best cell: B3_icosidodec30_tau001 — Δ_1cm +0.0199, Δ_1.5cm +0.0077, max_class 0.050
- artifact: `analysis/plan-023/results_B3.json`, `G2_B3.log`

### Z023_B4_fib50 (K=50, complete) 🏆
- started_at: 2026-05-19 12:09 KST, duration: 4683s (78 min)
- best cell: **B4_fib50_tau001** — Δ_1cm +0.0212, Δ_1.5cm +0.0075, max_class 0.032
- artifact: `analysis/plan-023/results_B4.json`, `G2_B4.log`

## 외부 시스템 결과 (DACON LB)

out-of-scope (plan §9 박제 — train OOF 만, no LB submit).

## 특이사항

- **Parallel oversubscription 박제 (decision-note c6)**: 첫 시도에서 4 sub-exp 병렬 background 실행 (load avg 192 = 96 core 의 2x over). 113 min 후에도 fold 0 미완료 → kill + serial 전환. serial 총 154 min 으로 완료. 향후 plan 의 G2 multi-sub-exp 는 **serial default**, parallel 시 `n_jobs=N/k` 명시 박제 권장.
- **K trend non-monotonic**: K=7(0.0262) < K=14(0.0279) > K=20/24/30(0.0272~0.0276) < K=50(0.0287). K=20~30 plateau 영역이 polyhedron vertex set 의 angular density 한계 추정.
- **Worktree data symlink (decision-note c5)**: worktree 에 data/ unzipped 부재 → /workspace/daycon-muflight/data/ 의 main repo 사본 symlink 사용.

## 다음 단계 후보

- plan-024 (가칭): B4_fib50_tau001 위 corrector reg head 재투입 (plan-022 followed_by 의 corrector 슬롯을 본 plan best 위로 carry)
- plan-025 (가칭): N × radius shell 2D sweep (radial lever 측정)
- plan-026 (가칭, lower priority): N>50 progression (samples/class < 200 risk zone)
