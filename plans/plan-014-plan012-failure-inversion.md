---
plan_id: 014
version: 0 (draft — narrative only)
date: 2026-05-14 (Asia/Seoul)
status: draft
based_on:
  - 012
followed_by: []
scope: plan-012 (codebook bake-off + hybrid) 의 진짜 실패 원인은 *paradigm 자체의 한계* 가 아니라 **plan-004 selector + plan-006 F0 의 재사용 강박** 이라는 가설. plan-012 의 7 observable failure mode (F1~F7) 가 모두 이 한 가지 root cause 의 증상. 따라서 본 plan = "재사용 끊기 (from-scratch redesign) vs plan-012 minimal-patch" 1축 head-to-head 실험. 본 파일은 *narrative 박제만* — 구체 spec 은 추후 채움.
exp_ids: []
lb_score: null
---

# plan-014 v0 — Reuse-Strap Inversion (from-scratch vs minimal-patch, draft)

## §0. 한 줄 목적

> **plan-012 의 5-fold OOF 0.6350 plateau (target 0.66 대비 -0.025) 는 paradigm 자체의 한계가 아니라 *plan-004 selector + plan-006 F0 의 재사용 강박* 이 누적된 결과라는 가설.**
>
> plan-012 의 7 observable failure mode (DCM collapse / anchor scale mismatch / F0 trivial dominance / shared encoder bottleneck / hard label CE noise / codebook geometry uniformity / frozen F0 path) 는 **단일 root cause = "기존 모듈을 그대로 쓰면서 hybrid head 만 위에 얹은 minimal-patch design"** 의 7가지 증상.
>
> 따라서 본 plan 의 실험 축 = 단 1개:
>
> **"from-scratch redesign (재사용 끊기) vs plan-012 minimal-patch" 의 head-to-head.**
>
> 둘의 OOF gap = paradigm 의 진짜 잠재력 측정. plan-013 의 "paradigm 폐기" 정당성 판단의 최종 evidence.
>
> **LB 제출 정책**: 미정 (narrative draft 단계).

---

## §0.5 Quick Reference

### plan-012 의 재사용 강박 (★ 본 plan 의 핵심 진단)

plan-012 ring_classifier.py 는 paradigm shift 라고 self-label 했지만 실제 코드는 다음 *minimal patch*:

```
self.scorer  = base.CandidateAttentionGRUSelector(...)   # ← plan-004 의 27-way selector 그대로
self.reg_head = nn.Sequential(...)                       # ← 위에 작은 MLP 추가
F0 = f0_predict_frenet_par120_perp_neg020(...)           # ← plan-006 numpy 함수 그대로 (no grad)
anchor radius = 0.005m                                   # ← plan-004/006 era scale hardcode
```

7 observable failure mode → 단일 root cause:

| failure mode | 재사용 강박과의 인과 |
|---|---|
| F4 shared encoder bottleneck | plan-004 selector encoder 를 두 head 가 공유 (= 원래 1-task 용 모듈 재사용) |
| F1 DCM collapse | F4 의 직접 결과 — encoder 가 mode classification 용으로 학습 못 함 |
| F7 frozen F0 path | plan-006 numpy F0 재사용 → gradient 없음, F0 학습 불가 |
| F3 F0 trivial dominance | F7 의 결과 — F0 hit sample 63% 가 학습 signal 에 무의미하게 남음 |
| F2 anchor scale mismatch | 재사용 시 plan-004/006 의 작은 scale (0.005m) 답습, task fit 재검토 없음 |
| F5 hard label CE noise | F2 의 결과 — anchor 가 hit zone 안에 갇혀 argmin label = noise |
| F6 codebook geometry uniformity | F1 의 결과 — encoder 가 어차피 center 만 고르니 anchor 위치 무관 |

→ 7-lever ablation 은 *증상 치료*. root cause 1개 = **"plan-004 selector + plan-006 F0 의 재사용 끊기"**.

### 1축 head-to-head (전략 narrative)

- **A. plan-012 minimal-patch baseline (reproduce)**
  - plan-004 `CandidateAttentionGRUSelector` reuse + plan-006 numpy F0 + reg_head 패치 그대로
  - 5-fold OOF 0.6350 (GPU) 재현 + DCM/loss curve 박제

- **B. from-scratch redesign (재사용 끊기)**
  - **B-encoder**: classifier 전용 encoder + reg_head 전용 encoder (or 별도 projection branch) — F1/F4 inversion
    - 양자택일 (별도 encoder vs 별도 projection branch) 은 §2 spec 단계에서 결정. **단**, 별도 encoder = F4 (shared encoder bottleneck) 의 full inversion / 별도 projection branch = encoder 본체 공유 → F4 partial inversion. 후자 채택 시 self-label "재사용 끊기" 강도 약화 → §2 채움 단계에서 trade-off 명시 필요.
  - **B-F0**: F0 coefficient 를 learnable parameter 로 (= plan-007 Step 4 통합) — F3/F7 inversion
  - **B-anchor**: anchor radius 를 task-fit scale 로 재정의 (hit radius 0.01m 동급 또는 학습 가능) — F2 inversion
  - **B-label**: hard label CE → distance-weighted soft label — F5 inversion
  - 모두 *paradigm 은 유지* (codebook + classification + regression hybrid 의 high-level idea), 단 *컴포넌트는 from scratch*

- **C. head-to-head**: A vs B 의 5-fold OOF gap = paradigm 의 진짜 잠재력.

### Target (judgement criteria)

- **B − A ≥ +0.020**: 재사용 강박이 진짜 원인 = paradigm 부활. plan-015 = B 의 polish + LB 제출.
- **0.005 ≤ B − A < 0.020**: 재사용이 partial 원인. paradigm + 재사용 양쪽 다 limit.
- **B − A < 0.005**: paradigm 자체 limit 확정. plan-012 의 결론 강화 + plan-013 의 "paradigm 폐기" 정당화.

### G-gates (stub — 추후 채움)

- G0: preflight (A baseline reproduce + DCM/loss 박제) [TODO]
- G1: B 4 컴포넌트 (encoder/F0/anchor/label) 의 from-scratch 구현 완료 [TODO]
- G2: head-to-head 1-fold A vs B 비교 + B ≥ A + 0.005 [TODO]
  - **fold mapping (narrative 가정)**: 1-fold gap +0.005 ≈ 5-fold OOF gap +0.005~+0.010 (fold noise 한 폭 가정). 즉 G2 통과 = §Target 의 최소 band ("0.005 ≤ B − A") 진입 *후보*. 정확한 mapping 은 §2/§3 spec 채움 단계에서 결정.
- G3: B 의 5-fold + submission (B 가 win 시 = §Target 의 "0.005 ≤ B − A" 이상) [TODO]
- G_final: synthesis + plan-015 후보 + 3 파일 sync [TODO]

### Commit chain (stub — 추후 채움)

| # | type | spec section | status |
|---|---|---|---|
| c1 | docs | `plans/plan-014-plan012-failure-inversion.md` v0 narrative draft | [DONE] 4657ff7 + 2a0f755 |

---

## §1. 배경 / 동기 (narrative)

### §1.1 plan-012 의 사망 진단 재검토

plan-012 results.md = "paradigm reframe 은 F0 raw hit 위 +0.002~0.003 만 추가 — paradigm 자체의 limit 확인". 그러나 plan-012 의 코드를 보면 *paradigm* 과 *재사용 강박* 이 구별되지 않은 채 같이 limit 으로 묶임:

- plan-012 는 self-label = "paradigm reframe (residual regression → classification + hybrid)"
- 실제 코드 = "plan-004 selector + plan-006 F0 위에 hybrid head 만 얹은 minimal patch"
- → plan-012 가 measured limit = **"minimal patch 의 limit"** 일 뿐, **"paradigm 의 limit"** 은 아직 측정 안 됨

본 plan 의 핵심 통찰:

> **plan-012 의 7 failure mode 는 7개 독립 문제가 아니라 1개 root cause (재사용 강박) 의 7가지 증상.**

따라서:
- 7-lever ablation = 증상 치료 ≠ 원인 치료
- 진짜 falsification test = "재사용 끊기" 1축

### §1.2 재사용 강박의 trap chain (구조적 인과)

```
"plan-004 selector reuse"  ─┐
                            │
                            ▶ encoder 가 candidate-selection 표현만 학습 가능 (F4)
                            │
                            ▶ classifier head 가 mode 학습 못 함 = center collapse (F1)
                            │
                            ▶ codebook geometry 가 결과에 무관 (F6)

"plan-006 numpy F0 reuse" ──┐
                            │
                            ▶ F0 gradient 없음 (F7)
                            │
                            ▶ F0 정확 prediction 이 hit 63% 에서 이미 답 = head 가 학습할 거리 없음
                            │
                            ▶ F0 hit 63% sample 학습 signal 무의미 (F3)
                            │
                            ▶ hard label = noise (F5)

"plan-004/006 era scale"  ──┐
                            │
                            ▶ anchor 0.005m 답습 → hit zone 내부 갇힘 (F2)
```

→ 7 mode 의 *원인이 1개* 라는 진단이 옳다면, "재사용 끊기" 1축 실험만으로 7 mode 가 동시에 풀려야 함. 풀리지 않으면 paradigm 자체 limit 확정.

### §1.3 plan-013 과의 path 분기

- **plan-013** (직전): paradigm 폐기 + plan-004 framework 회귀 → G2 0/3 axis FAIL, G1 0.6381 fallback submission
- **plan-014** (본 plan): paradigm 부활 시도 — 단 *재사용 강박만 끊고* paradigm 자체는 유지
- 두 plan 의 결과 조합 의 LB 분기 (**용어**: plan-014 "positive" = §0.5 Target band 의 "B − A ≥ +0.020" (paradigm 부활) 또는 그 이상; "negative" = "B − A < 0.005" (paradigm 자체 limit 확정). plan-013 0.68 thresh = 본 plan 의 가정값):
  - plan-013 LB ≥ 0.68 + plan-014 negative → paradigm 폐기 정당화 (plan-004 framework path 가 정답)
  - plan-013 LB < 0.68 + plan-014 positive → paradigm 부활 (from-scratch redesign 이 정답)
  - 둘 다 negative → 더 deep path-pivot (plan-012 Candidate A = KNN/GP/Diffusion)

### §1.4 본 plan 의 정직성 원칙

- B (from-scratch) 가 A (minimal-patch) 와 ΔOOF < 0.005 이면 *재사용이 원인이 아니라 paradigm 자체가 limit* 임을 인정 — negative result 도 동등 가치
- 7-lever 증상 치료 유혹 회피 — 1축 head-to-head 만 측정. 작은 lever 들 의 마진 줍기 X
- 본 plan 은 **"paradigm 의 한계 vs 재사용의 한계" 의 명시적 falsification test** 로 설계

---

## §2. 실험 설계 [TODO]

(narrative draft 단계 — 추후 채움)

## §3. 합격 기준 / G-gate 정량 spec [TODO]

(narrative draft 단계 — 추후 채움)

## §4~§N. 코드 / Phase 별 spec [TODO]

(narrative draft 단계 — 추후 채움)
