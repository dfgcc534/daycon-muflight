# plan-022 results_A4 — A4_2shell13 (2-shell octahedron: center + inner 0.25cm + outer 0.5cm)

| τ_cls | hit@1cm | hit@1.5cm | Δ_1cm | Δ_1.5cm | pass_both | max_class |
|---|---|---|---|---|---|---|
| **0.001** | **0.6496** | **0.8094** | **+0.0176** ✓ | **+0.0061** ✓ | **🎉 True** | 0.116 |
| 0.003 | 0.6401 | 0.8070 | +0.0081 ✓ | +0.0037 ✗ | False | 0.094 |
| 0.005 | 0.6365 | 0.8058 | +0.0045 ✗ | +0.0025 ✗ | False | 0.088 |

## Finding — H4 refuted

- Best A4_tau001: Δ_1cm +0.0176 / Δ_1.5cm +0.0061, **sum 0.0237 < A1/A2/A3 sum 0.0262/0.0269/0.0270**.
- 가설 H4 (2-shell multi-scale coverage 로 1.5cm 향상) **refuted**: inner shell 0.0025m anchor 가 1.5cm reach 향상 효과 없음. 오히려 1cm 도 손해 (-0.0018 vs A1).
- 추정 원인: inner shell anchor 가 final mixture norm 을 줄여 F0 근접 예측 산출 → F0-perfect sample 은 이미 center (anchor 0) 가 가져가서 inner shell 이 redundant + sample 분산 (probs mass 가 inner / outer 로 spread) 만 야기.
- τ_cls=0.005 에서 Δ_1cm < +0.005 → 단일 cell fail.

G2.A4 PASS (3 cell metric finite + max_class_ratio < 0.95). 696s.
