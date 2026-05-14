# New Ideas — paradigm-shift 후보 점검

> plan-011 의 4-axis corrector ablation (G1 (b) FAIL, 0/4 axes strict +0.005) 이후 paradigm-shift 후보.
> 데이터 shape: 11 step × 40ms regular grid, +80ms horizon, ~10K samples, Hit-rate 지표.
> 현 baseline: plan-006 frenet_par120_perp_neg020 LB 0.6692, plan-004 corrector LB 0.6822.

---

## Batch A — 1차 검토 (IMM / FNO / Neural ODE)

### A.1 IMM-KF (Interacting Multiple Model Kalman Filter) — ★★ 조건부 가치

**개념**: CV (등속) / CA (등가속) / CT (등선회) 등 3~4 개 물리 필터 병렬 + Markov transition probability 로 soft blending. 레이더 추적 도메인 표준.

**현 task 쓸모**:
- ✅ 도메인 fit: radar 추적 ↔ 모기 mode-switch 직접 대응. DL 과 신호원 독립 → ensemble 다양성.
- ✅ 11 step / 10K sample 에서 안정 (파라미터 적음).
- ⚠️ plan-008 의 27-candidate selector 가 *discrete IMM* — IMM 3-4 mode 는 *더 coarse*.
- ⚠️ plan-006 frenet_par120_perp_neg020 (LB 0.6692) = CV + anisotropic perp hand-tuned 버전. 의미있게 이기려면 mode-transition signal 이 11 step 안에 잡혀야 하나 plan-011 결과는 그 signal 약함을 시사.
- ❌ Lévy flight (heavy-tail, 급선회) ≠ Gaussian process noise — KF 핵심 가정 위배.

**Verdict**: 단독 main 으로는 LB 0.66~0.68 plateau. **plan-012 backup path 또는 ensemble member 로만 가치**.

### A.2 FNO / FEDformer (주파수 도메인 변환) — ★ 부적합 (치명적)

**개념**: 11-step (x, y, z) 시퀀스 → FFT → 주파수 spectrum 위 예측 → iFFT 로 좌표 복원. 관성(저주파) ↔ 날갯짓 노이즈(고주파) 분리.

**현 task 쓸모**:
- ❌ **N=11 → FFT 무의미**. Nyquist 까지 6 frequency bin, amplitude estimate variance 폭주. FEDformer/FNO 표준 benchmark (ETTh1, Navier-Stokes) 는 모두 N=96~336 위 동작.
- ❌ Wingbeat aliasing 가설 자체는 맞음 (모기 300-700Hz × 25Hz 샘플링) 그러나 **11 sample 로는 aliased phase 복원 불가** — 그냥 noise.
- ❌ +80ms = 2 step → frequency-domain extrapolation 은 spatial 보다 더 ill-posed (단기 horizon 일수록 spatial 우위).
- ❌ 10K × 11 = operator learning 에 필요 scale 의 1/10~1/100.

**Verdict**: **Skip**. 가설은 멋있으나 데이터 길이가 fatal.

### A.3 Neural ODE (신경 상미분 방정식) — ★★ 제한적

**개념**: 신경망을 dh(t)/dt = f(h(t), t, θ) 로 정의. 11 점은 ODE 초기 궤적, ODE solver (Runge-Kutta) 로 +80ms 적분.

**현 task 쓸모**:
- ❌ **regular 40ms grid → Neural ODE 의 main advantage (irregular timestamp 처리) 무효화**.
- ❌ 11-step initial trajectory → ODE underdetermined (dynamics 학습 data 부족).
- ❌ Lévy flight 급선회 = discrete event → ODE smooth dynamics 가정 위배.
- ⚠️ "Spline polynomial-explosion" (plan-002 S004 cv 0.033) 은 **이미 plan-006 Frenet 외삽이 해결한 문제** — Neural ODE 가 추가 해결할 게 없음.
- ✅ horizon-flexible (+40ms, +80ms 자유) — 본 대회 +80ms 고정 → 활용 못 함.

**Verdict**: **Skip (primary)**. ensemble smoothing prior 로는 미미하게 의미 있음.

---

## Batch B — 2차 검토 (KNN / SE(3) / Koopman)

### B.1 Trajectory KNN + Displacement Mapping — ★★★ 가장 강한 후보

**개념**: 10K train 의 (정규화된) 11-step trajectory 위 cosine/DTW 유사도 → top-K 의 +80ms 변위를 retrieve → 현재 위치에 더함. 패턴 매칭 접근.

**현 task 쓸모**:
- ✅ **이미 [analysis/plan-011/next_plan_candidates.md](analysis/plan-011/next_plan_candidates.md) 후보 3A 로 박제** — 본 batch 는 *구체적 구현 spec 으로 격상*.
- ✅ Lévy flight 대응: 급선회 사례가 train 10K 안에 *반드시 존재* → template matching 으로 회수.
- ✅ N=11 = 패턴 매칭에 충분 (모델 학습엔 짧지만 fingerprint 로는 명확).
- ✅ Hit-rate 메트릭 fit: top-1 retrieval = 분포의 *mode* → Hit-rate 와 정렬됨 (top-k mean 보다 우월).
- ⚠️ Normalize-rotate critical — plan-004 의 v_last 회전 정규화 그대로 재사용 필요.
- ⚠️ Train/test 도메인 차이 (실내/야외 mix) 시 retrieval 성능 저하 가능.
- ⚠️ **+80ms displacement 가 trajectory-shape-specific 가 아니라 모기 종/풍속/환경-specific** 일 가능성 → plan-012 c1~c2 에서 displacement clustering 통해 *조기 검증* 필요 (fatal hypothesis).

**Verdict**: **plan-012 backup 1순위 격상**. 단, 단독 KNN 보다 *27-pool 확장 변형* (B.1.1) 이 ROI 우월.

### B.1.1 (Sub-variant) KNN-Augmented 27-Candidate Pool — ★★★ user 제안 ★ 핵심 spec

**개념**: 기존 plan-008 의 selector framework + 27-candidate pool 골격 *보존*. `selector.make_candidates()` 에 KNN-displacement 후보 k=3~5 개를 *추가*. Selector 가 Frenet vs KNN 후보 선택을 자연스럽게 학습.

**구체적 구현 spec**:

```python
# src/pb_0_6822/knn_candidates.py (신규)
# ──────────────────────────────────────────────────
# Step 1: train pool 구축 (한 번만, plan-012 c1)
# ──────────────────────────────────────────────────
# - 모든 train sample 의 11-step trajectory 를 (v_last frame) 정규화
#   normalize: p0=0 (last position), v_last → +x axis 회전 (yaw, pitch)
# - 정규화 trajectory 의 displacement vec (11 step × 3) → flatten 33-dim
# - 동시에 +80ms 정답 displacement (3-dim) 도 same frame 정규화
# - Faiss IndexFlatL2 (cosine 은 L2-norm 후 L2 index 등가) 로 33-dim 색인

# ──────────────────────────────────────────────────
# Step 2: inference 시 후보 생성 (plan-012 c2)
# ──────────────────────────────────────────────────
def make_knn_candidates(x, end_idx, faiss_index, train_targets, k=5):
    """
    x: (B, T, 3) raw trajectory
    end_idx: 마지막 input step (보통 10)
    Returns: (B, k, 3) — KNN-derived candidate positions in raw frame
    """
    # 1. v_last frame 정규화 (plan-004 selector 와 동일)
    p0 = x[:, end_idx]                       # (B, 3)
    v_last = x[:, end_idx] - x[:, end_idx-1] # (B, 3)
    R = build_rotation(v_last)               # (B, 3, 3) v_last → +x
    x_norm = (x - p0[:, None]) @ R           # (B, T, 3)

    # 2. Faiss query (33-dim flatten)
    query = x_norm.flatten(start_dim=1)      # (B, 33)
    _, knn_idx = faiss_index.search(query, k) # (B, k)

    # 3. retrieve target displacement (already normalized frame)
    delta_norm = train_targets[knn_idx]      # (B, k, 3)

    # 4. inverse rotate → raw frame
    delta_raw = delta_norm @ R.transpose(-2, -1)  # (B, k, 3)
    knn_cands = p0[:, None] + delta_raw      # (B, k, 3)
    return knn_cands

# ──────────────────────────────────────────────────
# Step 3: 기존 27-pool 에 append (plan-012 c3)
# ──────────────────────────────────────────────────
# src/pb_0_6822/selector.py:make_candidates 끝부분에:
#   knn_cands = make_knn_candidates(x, end_idx, faiss_idx, train_tgt, k=5)
#   candidates = torch.cat([candidates, knn_cands], dim=1)  # (B, 27+k, 3)
# Selector head 의 output dim 32 → 32 (27→32) 로 확장.
```

**Hyperparameter spec (plan-012 §0.5 grid)**:
- k ∈ {3, 5, 8} — top-k retrieval count
- distance metric ∈ {L2 (cosine 후), DTW (fastdtw lib)} — DTW 는 cost ↑↑ 이므로 c2 에서 L2 먼저
- query 정규화: v_last frame **without** pitch (yaw only, pitch 는 over-rotate risk)
- train target leakage 방지: **fold 별 separate Faiss index** (OOF 일관성)
- displacement clustering pre-check (B.1 의 fatal hypothesis 검증): K-means k=5~10 on normalized targets, silhouette ≥ 0.3 만 통과

**합격 기준 (plan-012 G_KNN)**:
- ★ (a) fold-0 OOF ≥ 0.6500 (plan-011 best Phase In ID = 0.6450 + 0.005)
- ★ (b) selector_logit 분포에서 KNN 후보의 *avg log-prob* ≥ Frenet 후보의 60% 이상 (KNN 후보가 무시되지 않음을 검증)

**Risk**:
- displacement clustering silhouette < 0.3 시 ⇒ B.1 fatal hypothesis 확정 ⇒ 후보 폐기 (early kill, plan-012 c1 종료).
- 27-pool 확장 시 selector output dim 변화 → plan-004 checkpoint state_dict 부분 load 필요 (frozen GRU + new head).

**Implementation 비용**: 1~2 day (Faiss CPU index + normalize util 재사용 + selector forward shape 수정).

**예상 ΔOOF**: +0.005~0.015 vs L0 anchor (KNN 신호 실재 시).

**Verdict**: **plan-012 main path 의 sub-task 1순위**. CNN encoder 확장 (기존 후보 1) 과 *병렬 실험* 가능 (input axis ↔ candidate pool 축 독립).

### B.2 Lie Group SE(3) 기반 기하학적 외삽 — ★★ 제한적

**개념**: 각 시점간 움직임을 SE(3) 변환 행렬로 정의 → se(3) (Lie algebra) 공간으로 log-매핑 → 선형화된 속도/회전 벡터에 최근 가중치 → exp 매핑 으로 +80ms 변환 추정.

**현 task 쓸모**:
- ⚠️ **모기는 point particle** — LiDAR position 만 관측, orientation 미관측. SE(3) 의 회전 성분은 trajectory tangent 에서 *유도* 해야 함.
- ⚠️ Tangent 에서 유도한 SE(3) = **본질적으로 Frenet frame 과 동등**. 즉 plan-006 frenet_par120_perp_neg020 (LB 0.6692) **이미 SE(3) 외삽의 단순화 버전**.
- ✅ Frenet → SE(3) marginal 확장 = explicit angular velocity smoothing (se(3) angular component). plan-006 의 perp_neg020 보다 정교한 회전 처리 가능.
- ⚠️ Lévy flight 급선회 = se(3) angular component 급변화 = noise. SE(3) 자체가 해결책 아님 (log-linearization smooth 가정 의존).
- ✅ Implementation 가벼움 (`pypose.SE3`, `liegroups`).

**Verdict**: plan-006 의 mathematical generalization. ΔOOF +0.002~0.005. 단독 main 으로는 부족. **plan-010 의 corrector redesign 위 보조 module 로 가치** — Frenet 외삽의 anisotropic perp scaling 을 SE(3) se(3) angular weighting 으로 대체.

### B.3 Koopman Operator 선형 근사 — ★ 부적합

**개념**: MLP encoder 로 11 점 → 고차원 observable space z_t → linear evolution z_{t+1} = A z_t. A^2 로 +80ms (2 step) 예측 → decode.

**현 task 쓸모**:
- ❌ **N=11 → 10 transitions 으로 d×d A 행렬 추정** (d=64~128) = **severely underdetermined** (10 vs 4K-16K 파라미터). DMD 표준 use-case (fluid dynamics) 는 N>>d.
- ⚠️ Global Koopman (모든 trajectory 공유 A): 모기 individual variation 무시 → 평균값 회귀.
- ⚠️ Sample-conditioned Koopman (A = MLP(initial_state)): **본질적으로 GRU/LSTM 의 simpler linear form** — R001 baseline-residual-gru 가 이미 capable, 차별화 없음.
- ⚠️ Koopman main advantage = long-horizon linear stability. **2-step horizon 에서 numerical advantage 무시 가능**.

**Verdict**: **Skip**. N=11 이 Koopman 핵심 가정 (data-rich linear regime) 위반.

---

## Batch C — 3차 검토 (MDN / Path Signatures / SupCon-KNN)

### C.1 MDN (Mixture Density Network) — ★★~★★★ F axis 의 진정한 측정

**개념**: 출력을 (x,y,z) 좌표 대신 GMM 파라미터 (π_k, μ_k, Σ_k). Inference 시 density mode (가장 좁고 높은 π·N) 선택 → mean 회귀 회피.

**현 task 쓸모**:
- ✅ Hit-rate ↔ density mode 정렬 (mean 회귀 회피) — 진짜 통찰.
- ✅ Lévy flight multimodal (좌/우 분기) 가설 fit.
- ⚠️ **plan-008 의 27-cand selector + argmax = 이미 *discrete* MDN**. argmax(π_k) over 27 fixed modes. 연속 MDN 의 marginal 가치 = 27 mode 가 *cover 못 하는* 영역 도달.
- ⚠️ plan-011 F4 (LearnableSingleCandidate) 가 F3/F4 parity bug 로 catastrophic 실패 (OOF 0.0980/0.0322). MDN = F4 의 multi-modal 일반화 → **F3/F4 parity fix 가 선행 조건**.
- ⚠️ MDN training 불안정 (mode collapse 표준 failure). N=11 + 10K + GMM 3-component → π_k 가 한 component 로 collapse 흔함.
- ✅ **Minimal-change variant (C.1' 권장)**: selector 의 27 mode 위 **per-candidate variance σ_k 학습** + mixture density argmax → 기존 골격 보존하면서 mode-seeking 도입. plan-011 L axis 의 *진정한 측정* 으로 격상.

**Verdict**: 단독 연속 MDN = ★★ (mode collapse risk). **C.1' variance-aware 27-cand MDN = ★★★** (조건부, F3/F4 parity fix 후 진입).

### C.2 Path Signatures (Rough Path Theory) — ★★ 흥미로운 input 대안

**개념**: 11-step trajectory 의 iterated integral 을 fixed-size signature feature (truncation level k 까지) 로 추출. `signatory` lib.

**현 task 쓸모**:
- ✅ N=11 + truncation k=2~3 → 39~120 features = 다루기 좋은 fixed-size. CNN 1D 와 차원 비교 가능.
- ✅ Lead-lag 변환 추가 시 LiDAR jittering 의 noise robustness 일부 확보 (낮은 order sig 한정).
- ⚠️ **"적분이라 노이즈에 강함" 은 order 1~2 에서만 참**. Order 3+ iterated integral 은 finer geometric detail capture = noise amplification. truncation 상한 = task 의 real 한계.
- ⚠️ Signature 는 temporal ordering 의 일부 정보 손실 — 마지막 위치 p0 는 별도 concat 필수.
- ⚠️ plan-011 In axis ID (CNN 64-dim, fold-0 OOF 0.6450) = 신호 가장 강한 axis. Signature 가 CNN 을 *이길지 합세할지* 불확실.
- ✅ `signatory` lib 안정적, 구현 1 day.

**Verdict**: **plan-012 In axis 추가 sub-experiment 슬롯** (CNN encoder 확장과 병렬 비교). 단독 main 으로는 부족, **CNN-encoder + signature concat** hybrid 가 ROI 최대화.

### C.3 SupCon-KNN — ★★★ B.1.1 의 직접 upgrade

**개념**: 가벼운 encoder + Supervised Contrastive Loss (positive pair = **미래 +80ms 변위 유사**). Latent space 위 KNN retrieval (Faiss). Coord-level distance 의 noise/도메인 취약성 보완.

**현 task 쓸모**:
- ✅ **"Positive pair = 미래 변위 유사" frame 이 진짜 novel insight**. Standard contrastive (SimCLR augment-pair) 와 본질 다름.
- ✅ B.1.1 의 L2 거리 약점 (coord-level 유사가 displacement 유사 보장 못 함) 을 정확히 보완.
- ✅ Domain generalization: 실내/야외 + 모기 종 차이가 latent 에서 quotient out → train/test 도메인 차이 mitigation.
- ✅ **MDN mode-seeking + KNN retrieval 통합**: SupCon latent = invariant 함수, latent 위 KNN = displacement mode 의 discrete retrieval = mean 회귀 회피 = Hit-rate 직접 fit.
- ⚠️ Positive pair 정의: 미래 변위 유사 의 quantization 필요 (cosine sim ≥ τ 또는 K-means cluster 동일). continuous SupCon (InfoLOOB 등) 은 mature 도 ↓, hyperparam ↑.
- ⚠️ 10K sample → contrastive batch 256~512 → epoch 당 20~40 batch = small-data regime. Encoder 작게 (≤ 100K param) 유지.
- ⚠️ Risk: SupCon latent 자체가 displacement predictor 가 되면 KNN 우회 → 단, latent → KNN = mode, latent → MLP = mean → Hit-rate 에서 retrieval 우월.
- ✅ Implementation 3~5 day. B.1.1 위 *2 단계 (coord-KNN → SupCon-KNN) 점진 진입* 가능.

**Verdict**: **B.1.1 의 직계 후속 = plan-012 main path 의 c4~c5**. 3 신규 후보 중 ROI 1 위.

---

## Batch D — 4차 검토 (Voxel CE / VQ-Trajectory / Trajectory-CLIP)

### D.1 3D Voxelized Classification — ★★★ corrector reformulation

**개념**: 예측 공간을 1cm voxel grid (예: 15×15×15 = 3,375 class) 로 discretize → CV 외삽 중심점 위 softmax 분류. argmax voxel center 가 최종 좌표. Hit-rate 와 1:1 정렬.

**현 task 쓸무**:
- ✅ Hit-rate ↔ argmax voxel 직접 정렬 — MDN mode collapse / σ 학습 불안정 회피.
- ✅ Multimodal softmax 자연 처리 (좌/우 분기 distribution 표현).
- ✅ CE loss 안정 (MDN log-likelihood vs softmax CE).
- ⚠️ **3,375 class × 10K sample = class 당 ~3 sample** → severely data-starved + imbalanced. **soft label trick (Gaussian-around-true-voxel) 필수** — adjacent voxel 의 continuous structure 회복.
- ⚠️ **15cm window 가 hyperparameter** — plan-006 CV 외삽의 99 percentile error 가 7.5cm 초과 시 window 밖 target 누락. plan-012 c1 에서 max_eucl_error distribution 측정 후 window 조정 필요.
- ⚠️ Voxel center ±0.5cm 안전, but 인접 voxel corner √3 ≈ 1.73cm — adjacent voxel mis-pick 시 hit miss.
- ✅ **권장 frame**: selector coarse pick (27 후보) 유지 + voxel head 를 plan-010 corrector_redesign 의 regression head 대체. **5×5×5 = 125 class** (selector best candidate 위 ±2.5cm window) 가 훨씬 tractable.

**Verdict**: **plan-010 corrector head reformulation 으로 ★★★**. selector 전체 replacement (3,375-class)는 ★★ (data starvation).

### D.2 VQ-Trajectory (Vector Quantized 코드북) — ★★ C.3 대비 정보 손실

**개념**: VQ-VAE 구조 — 11-step trajectory → encoder → K=256 codebook 중 nearest 로 quantize. Codebook 이 maneuver micro-motif (좌선회/감속/수직상승 등) 자동 군집. Inference 시 codebook index → train sample 의 +80ms displacement lookup.

**현 task 쓸무**:
- ✅ Codebook 의 maneuver primitive discovery — interpretability ↑.
- ✅ O(1) lookup (Faiss 불필요).
- ✅ Bottleneck 의 noise suppression.
- ⚠️ **VQ-VAE 학습 난이도**: commitment loss + EMA codebook + dead-entry reset 필요. audio domain (RQ-VAE, EnCodec) mature 하나 10K small-data regime 위험.
- ⚠️ **K=256 < 10K = 정보 손실**: 같은 codebook 에 매핑된 sample 들의 displacement 평균 → multimodal collapse. per-codebook GMM 또는 top-k retrieval 필요 = 결국 codebook 위 KNN.
- ⚠️ **C.3/D.3 와 비교**: 연속 latent + Faiss 10K retrieval (C.3/D.3) >> 이산 latent + 256-cell lookup (D.2). **D.2 는 strictly less expressive**.
- ⚠️ Inference latency: 10K Faiss = 마이크로초 → bottleneck 아님. O(1) lookup 의 실질 이득 0.

**Verdict**: 우아하나 **C.3/D.3 의 strict downgrade**. plan-013 ensemble diversity member (이산 prototype vs 연속 retrieval inductive bias 차이) 로만 가치.

### D.3 Trajectory-CLIP (Dual-Encoder InfoNCE) — ★★★ C.3 의 strict upgrade

**개념**: Encoder A (past 11-step) + Encoder B (future +80ms displacement) → InfoNCE: same-sample (past, future) pair 가까이, 다른 sample pair 멀게. 학습 후 Encoder B drop, Encoder A latent 위 KNN retrieval.

**현 task 쓸무**:
- ✅ **InfoNCE > heuristic SupCon**. C.3 의 fatal weakness ("미래 변위 유사 의 quantization") 를 same-sample positive pair 로 우아하게 해결. 모델 스스로 정렬.
- ✅ Encoder A latent = "past 가 *predictive of similar future*" 인 sample 끼리 모임 = retrieval mechanism 본질 정렬.
- ✅ 학습 후 Encoder B drop → 추론 cost = C.3 와 동일.
- ⚠️ **Single-positive InfoNCE 의 multimodality 한계**: 같은 past 가 좌/우 분기 미래 (Lévy) 갖는 경우, *one past → one future* 강제 → Encoder A latent 가 mean displacement 로 collapse 가능.
  - 완화책: 추론 시 k=5~10 nearest train neighbor retrieval → 각 distinct displacement 가 candidate pool 의 element. **D.3 + B.1.1 (KNN-augmented 27-pool) 결합 시** multimodality 회복.
  - 또는 stochastic Encoder B (VAE) 또는 multi-positive InfoNCE.
- ⚠️ Batch 256~512 InfoNCE → 10K sample 위 epoch 당 20~40 batch. 작지만 가능.
- ✅ **C.3 replacement**: plan-012 main path c4~c5 의 SupCon-KNN 을 그대로 D.3 로 교체. pipeline 변경 0.

**Verdict**: **C.3 의 strict upgrade — plan-012 main path c4~c5 에 직접 대체**.

---

## 종합 비교 (12 후보 ranking 갱신)

| Candidate | Expected ΔOOF | Cost | ROI |
|---|---|---|---|
| (기존) CNN/transformer encoder 확장 | +0.005~0.015 | 중 | ★★★ |
| (B.1.1) KNN-augmented 27-pool | +0.005~0.015 | 소 | ★★★ |
| **(D.3) Trajectory-CLIP ★ C.3 upgrade** | **+0.010~0.020** | 중 | **★★★** |
| **(D.1) Voxel CE corrector head ★ plan-010 reformulation** | **+0.005~0.012** | 소~중 | **★★★** |
| (C.1') variance-aware 27-cand MDN | +0.005~0.010 (F3/F4 fix 후) | 소~중 | ★★★ 조건부 |
| ~~(C.3) SupCon-KNN~~ (D.3 로 superseded) | — | — | deprecated |
| (기존) F3/F4 formula parity fix | +0.002~0.010 | 소 | ★★ |
| (C.2) Path Signatures (In axis sub-slot) | +0.002~0.008 | 소 | ★★ |
| (A.1) IMM-KF (ensemble member) | +0.003~0.008 | 중 | ★★ |
| (B.2) SE(3) (plan-010 corrector 보조) | +0.002~0.005 | 소~중 | ★★ |
| **(D.2) VQ-Trajectory (plan-013 ensemble diversity)** | +0.002~0.005 | 중 | ★★ |
| (기존) KNN paradigm 3B/3C (Diffusion/Hybrid) | +0.01~0.05 불확실 | 대 | ★★ |
| (A.2/A.3/B.3) FNO / Neural ODE / Koopman | ~0 | 대 | ★ |

---

## 권장 진행 (plan-012~013 roadmap 최신)

**plan-012 main path** (D.3 가 C.3 대체):
```
c1: train pool 구축 + displacement clustering pre-check + CV max_error distribution (D.1 window 조정용)
c2: coord-KNN candidates → 27-pool 확장 (B.1.1 baseline)
c3: selector 재학습 + G_KNN 합격 측정
c4: Trajectory-CLIP encoder 학습 (D.3) — InfoNCE on (past, future displacement)
c5: latent-KNN k=5~10 candidates → 27-pool 재확장 + selector 재학습 (multimodality 회복)
c6: CNN encoder 확장 (기존 후보 1) — In axis 병렬
c7: 5×5×5 Voxel CE corrector head (D.1) — plan-010 regression corrector 대체
```

**plan-012 alt path** (직교 axis):
```
c8~c9: F3/F4 parity fix (plan-011.1 carry-over)
c10: variance-aware 27-cand MDN (C.1') — L axis 진정 측정
c11: Path Signatures sub-exp (C.2) — In axis alt
```

**plan-013 backup / ensemble diversity**:
```
- A.1 IMM-KF (physics ensemble member)
- B.2 SE(3) corrector module
- D.2 VQ-Trajectory (discrete prototype ensemble member)
- KNN paradigm 3B/3C (Diffusion / Hybrid)
```

**Skip 확정**: A.2 FNO / A.3 Neural ODE / B.3 Koopman (N=11 + regular grid + 2-step horizon 위에서 본질 advantage 사라짐).

---

**핵심 근거**: plan-011 4-axis ablation 이 corrector path 의 *candidate pool 자체가 Frenet family 27 variant 에 갇힘* 구조적 ceiling 입증.

**2 단계 paradigm shift** (plan-012 main path):
1. **Pool 다양화** = B.1.1 coord-KNN candidates 추가
2. **Retrieval 품질 고도화** = D.3 Trajectory-CLIP latent metric 학습 → latent-KNN

**Corrector reformulation** (plan-012 c7):
- D.1 Voxel CE = regression → classification → Hit-rate 직접 정렬

기존 selector framework 보존하면서 corrector path 의 구조적 ceiling 돌파.
