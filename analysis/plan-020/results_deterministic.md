# plan-020 STAGE 2 G2.D — 14 deterministic 후보 5-fold OOF

## 결과 표 (2026-05-18, reduced spec popsize=10/maxiter=50/seeds=3)

| # | candidate | family | hit@1cm | Δ_1cm | hit@1.5cm | Δ_1.5cm | pass (둘 다 ≥ +0.005) |
|---|---|---|---|---|---|---|---|
| — | **F0 baseline** | — | 0.6320 | — | 0.8033 | — | — |
| C01 | helix (3 param α/β/γ) | F1 회전 | 0.5874 | −0.0446 | 0.7912 | −0.0121 | ✗ |
| C02 | CTRA (0 param) | F1 회전 | 0.5070 | −0.1250 | 0.6898 | −0.1135 | ✗ |
| C03 | CTRV (0 param) | F1 회전 | 0.5207 | −0.1113 | 0.7187 | −0.0846 | ✗ |
| C04 | IMM (3 param w_diag) | F1 회전 | 0.5980 | −0.0340 | 0.7974 | −0.0059 | ✗ |
| **C05** | **per-regime F0 (54 param)** | **F2 data-driven** | **0.6503** | **+0.0183** | **0.8086** | **+0.0053** | **✓** |
| C06 | Quintic Hermite (0 param) | F3 고차 미분 | 0.0096 | −0.6224 | 0.0260 | −0.7773 | ✗ |
| C07 | Jerk-aware cubic (0 param) | F3 고차 미분 | 0.3929 | −0.2391 | 0.5847 | −0.2186 | ✗ |
| C08 | Singer (1 param τ_a) | F4 noise-adaptive | 0.5951 | −0.0369 | 0.7851 | −0.0182 | ✗ |
| C09 | Kalman (2 param log_q/log_r) | F4 noise-adaptive | 0.2374 | −0.3946 | 0.3846 | −0.4187 | ✗ |
| C10 | Bishop frame (1 param λ) | F5 기하학 | 0.6320 | +0.0000 | 0.8033 | +0.0000 | ✗ (= F0, λ=1 stuck) |
| C11 | SE(3) twist (0 param) | F5 기하학 | 0.3450 | −0.2870 | 0.5323 | −0.2710 | ✗ |
| C12 | Wingbeat FFT (1 param f_c) | F6 도메인 정보 | 0.0008 | −0.6312 | 0.0015 | −0.8018 | ✗ (CMA fit fail, default fallback broken) |
| C13 | Lévy prior (degenerate) | F6 도메인 정보 | 0.6320 | +0.0000 | 0.8033 | +0.0000 | ✗ (= F0, design-by-limitation §N+2 #7) |
| C14 | Trajectory KNN (1 param k) | F7 비모수 | 0.3404 | −0.2916 | 0.5336 | −0.2697 | ✗ |

**Wall time**: 594.5 s (~10 min) for 14 후보 × 5 fold, reduced CMA-ES spec.

## Winner: C05_per_regime_f0

- 18-regime × 3 param (d1, par, perp) per regime, CMA-ES per-fold per-regime fit
- hit@1cm: 0.6320 → 0.6503 (Δ +0.0183, sample-level paired)
- hit@1.5cm: 0.8033 → 0.8086 (Δ +0.0053)
- Per-fold: [0.6515, 0.6450, 0.6580, 0.6485, 0.6483] / [0.8108, 0.8005, 0.8085, 0.8163, 0.8067]
- → **F0 단일 공식의 *18-regime 별 계수 분리* 가 본 plan-020 의 핵심 lever**.

## Family-level winners

| family | winner | Δ_1cm | Δ_1.5cm | pass |
|---|---|---|---|---|
| F1 회전 | C04 IMM | −0.0340 | −0.0059 | ✗ |
| **F2 data-driven** | **C05 per-regime F0** | **+0.0183** | **+0.0053** | **✓** |
| F3 고차 미분 | C07 jerk cubic | −0.2391 | −0.2186 | ✗ |
| F4 noise-adaptive | C08 Singer | −0.0369 | −0.0182 | ✗ |
| F5 기하학 | C10 Bishop (λ=1) | +0.0000 | +0.0000 | ✗ (F0 stuck) |
| F6 도메인 정보 | C13 Lévy (=F0) | +0.0000 | +0.0000 | ✗ (degenerate) |
| F7 비모수 | C14 KNN | −0.2916 | −0.2697 | ✗ |

## 진단

- **F0 family-level lever ablation 의 의미 있는 결과 = F2 (data-driven, per-regime)**. 다른 family (회전 / 고차 미분 / noise / 기하 / 도메인 / 비모수) 모두 F0 보다 worse → paradigm-level 보강 lever 가 *산식 구조 변경* 이 아니라 *계수 의 regime-conditional 분리* 임을 명확화.
- C10 Bishop 의 λ=1 stuck = CMA-ES 의 saddle point + 본 spec 의 orthonormality 항등성 → 본 후보의 *비-자명 lever 발견 실패*.
- C12 wingbeat 의 CMA-ES fit fail 5/5 + default fallback 도 broken (val_hit ~0) — FFT mask 가 zero-trajectory 만들 가능성. **v1.5 fix 후보**.
- C09 Kalman 의 worse 결과 — F0 의 par/perp asymmetric coef structure 가 raw KF 보다 task-specific 으로 informative.
- C06 Quintic explosion — 5차 다항식의 80ms extrapolation 의 instability 확정.

## G2.D PASS

- 17 후보 중 ≥ 1 paired Δ ≥ +0.005 *둘 다* 통과: **YES (C05)** → G3 진입 가능.
- 14 candidate 모두 metric finite ✓.
- G2.D `formula_numerical` severe 없음.
