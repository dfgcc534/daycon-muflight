# plan-022 results_A7 — A7_fib13 (Fibonacci spiral 12 + center, K=13, quasi-uniform)

| τ_cls | hit@1cm | hit@1.5cm | Δ_1cm | Δ_1.5cm | pass_both | max_class |
|---|---|---|---|---|---|---|
| **0.001** | **0.6521** | **0.8106** | **+0.0201** ✓ | **+0.0073** ✓ | **🎉 True** | 0.151 |
| **0.003** | 0.6427 | 0.8085 | +0.0107 ✓ | +0.0052 ✓ | **🎉 True** | 0.109 |
| 0.005 | 0.6376 | 0.8064 | +0.0056 ✓ | +0.0031 ✗ | False | 0.097 |

## Finding

- Best A7_tau001: Δ_1cm +0.0201, Δ_1.5cm +0.0073, **Δ sum 0.0274 = sweep 2위** (vs A6 0.0279).
- quasi-uniform sphere covering (Fibonacci spiral) 이 Platonic isotropic (icosahedron A2) 보다 marginal 우월 (sum 0.0274 vs 0.0269). center 보존 + 비결정적 spiral 패턴이 cuboctahedron axis-aligned (A3 sum 0.0270) 와 유사한 성능.
- 2 cell PASS_BOTH (τ=0.001, τ=0.003) — A2, A6 와 동일 robustness.
- max_class_ratio K=13 standard.

G2.A7 PASS. 826s.
