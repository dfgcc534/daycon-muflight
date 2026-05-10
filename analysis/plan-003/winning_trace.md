# plan-003 winning trace (c12 / G3.5)

## §1. Ablation paired Δ vs R001

R001 reference cv_mean_eucl = **0.013383**

| exp_id | cv_mean_eucl | Δ vs R001 | winning? |
|---|---:|---:|:-:|
| R002_physics-features | 0.015157 | +0.001775 | no |
| R003_ema-extrapolate | 0.014038 | +0.000656 | no |
| R004_wingbeat-oscillation | 0.013476 | +0.000093 | no |
| R005_loss-mse | 0.013388 | +0.000005 | no |

## §2. R006 config 자동 생성

- winning components = **0** → R006 = R001 직접 복제 (학습 skip).
- decision-note: spec-default — winning 0개, R006 ckpt + summary 모두 R001 의 비트 동일 사본.

## §3. R006 결과 + fallback verdict

- R006 cv_mean_eucl = **0.013383**
- vs R001 Δ = +0.000000
- **fallback = False** (R006.cv ≤ R001.cv + 0.001)
- LB submission = R006 의 csv.
