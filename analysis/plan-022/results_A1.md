# plan-022 results_A1 — A1_octa7 (octahedron + center, K=7)

3 τ_cls cell 측정 (5-fold OOF, plan-021 LGBM hparam 그대로). F0 baseline 대비 paired Δ.

| τ_cls | hit@1cm | hit@1.5cm | Δ_1cm | Δ_1.5cm | pass_both | max_class | fold_var_1cm | fold_var_1.5cm |
|---|---|---|---|---|---|---|---|---|
| **0.001** | **0.6514** | **0.8101** | **+0.0194** ✓ | **+0.0068** ✓ | **🎉 True** | 0.232 | 0.0054 | 0.0064 |
| 0.003 | 0.6432 | 0.8081 | +0.0112 ✓ | +0.0048 ✗ | False | 0.195 | 0.0047 | 0.0081 |
| 0.005 | 0.6393 | 0.8067 | +0.0073 ✓ | +0.0034 ✗ | False | 0.176 | 0.0051 | 0.0083 |

F0 baseline: hit@1cm = 0.6320, hit@1.5cm = 0.8033 (plan-020/021 carry).

## Finding

- **Best cell = A1_octa7_tau001** — pass_both=True, Δ_1cm=+0.0194, Δ_1.5cm=+0.0068.
- plan-021 A LGBM full (with reg head, τ=0.001): hit@1cm=0.6488, Δ_1cm=+0.0168, Δ_1.5cm=+0.0037 (1.5cm 미달).
- **본 plan-022 A1_octa7_tau001 (corrector-free)**: hit@1cm=0.6514, Δ_1cm=+0.0194, Δ_1.5cm=+0.0068. **plan-021 보다 Δ_1cm +0.0026 향상 + Δ_1.5cm +0.0031 향상 (PASS 으로 전환)** — reg head 제거가 오히려 더 나음 (plan-021 reg_offset 의 0.5% bound 사용이 노이즈 / 과적합 risk 였을 가능성).
- τ_cls 가 sharp 할수록 (0.001 → 0.005) Δ 모두 감소 — anchor 가 7 개로 적을 때 sharp soft label 이 유리.
- max_class_ratio 모두 < 0.25 (collapse threshold 0.95 와 매우 멀음).

## G2.A1 합격

- 3 cell 모두 metric finite ✓
- 3 cell 모두 max_class_ratio < 0.95 ✓
- G2.A1 PASS, soft_label_collapse 미발동.

## G3 진척

A1_octa7_tau001 PASS_BOTH = 21 cell sweep 중 첫 ≥ 1 cell PASS_BOTH 달성 → G3 사전 PASS. 나머지 6 sub-exp (A2-A7) 측정 후 best 선택 + paradigm finding.
