---
plan_id: 022
finished_at: 2026-05-19 (Asia/Seoul)
status: all_complete
band: positive
best_sub_exp: A6_bcc14_tau001
best_hit_1cm: 0.6528
best_hit_1.5cm: 0.8104
best_delta_1cm: +0.0208
best_delta_1.5cm: +0.0071
exp_ids_completed:
  - Z022_A1_octa7
  - Z022_A2_ico13
  - Z022_A3_cubocta13
  - Z022_A4_2shell13
  - Z022_A5_cube8
  - Z022_A6_bcc14
  - Z022_A7_fib13
exp_ids_skipped: []
lb_score: null
---

# plan-022.results pair (WORKFLOW.md §11)

핵심 결과는 `analysis/plan-022/results.md` 의 11 항목에 박제. 본 pair file 은
frontmatter 4-way 토큰 일치 (WORKFLOW.md §4 / §11) 의무 충족용 stub.

## 핵심 결과 요약

- **best**: A6_bcc14_tau001 (BCC 14 anchor + τ_cls=0.001 sharp soft label)
- **paired Δ**: Δ_1cm = +0.0208, Δ_1.5cm = +0.0071 (둘 다 PASS criterion +0.005 통과)
- **pass_both cell 수**: 10/21 cell
- **band**: positive (G3 PASS, severe 0건, warn 0건)

## 상세 분석 위치

- `analysis/plan-022/results.md` — 11 항목 G_final 종합
- `analysis/plan-022/paradigm_analysis.{json,md}` — 21 cell grid + marginals
- `analysis/plan-022/results_A{1..7}.{json,md}` — 7 sub-exp 개별 결과

## Follow-up

- plan-023: A6_bcc14 + corrector reg head 재투입 ablation
- plan-024: A6_bcc14 + GRU sub-exp + ensemble
- plan-025: DACON LB 측정 (사용자 quota confirm 필수)
