# train_faap_wgan_GD_8th 변경 요약 (vs 7th - 최고 성능 버전 기반)

## 분석: 7th의 성공 요인

### 7th 테스트 결과 (Epoch 23)
```json
{
  "perturbed": {
    "male": {"AP": 0.514, "AR": 0.836},
    "female": {"AP": 0.408, "AR": 0.833}
  },
  "deltas": {
    "male": {"AP": +0.0029, "AR": +0.0021},
    "female": {"AP": +0.0034, "AR": +0.0070}
  },
  "gaps": {
    "AP": {"baseline": 0.1063, "perturbed": 0.1059},
    "AR": {"baseline": 0.0081, "perturbed": 0.0032}
  }
}
```

### 7th 성공 핵심 요소
1. **비대칭 fairness 스케일링**: `fair_m_scale=0.5` → 남성 성능 보존
2. **Epsilon cooldown**: 0.10 → 0.09 후반부 노이즈 감소
3. **적당한 Beta 스케줄**: 0.5 → 0.6 (0.7까지 가지 않음)
4. **모든 지표 baseline 대비 향상**: 드문 성과!

### 4th vs 7th 비교
| 지표 | 4th | 7th | 승자 |
|------|-----|-----|------|
| Male AP Delta | -0.0083 | **+0.0029** | 7th |
| Female AP Delta | -0.0070 | **+0.0034** | 7th |
| Male AR Delta | -0.0010 | **+0.0021** | 7th |
| Female AR Delta | +0.0036 | **+0.0070** | 7th |
| AP Gap | 0.106 | **0.1059** | 7th |
| AR Gap | 0.008 | **0.0032** | 7th ✨ |

---

## 8th 설계 철학

> **"7th의 성공 요소를 유지하면서, AR Gap 개선을 더 강화"**

### 7th 핵심 유지 사항 ✅
1. ✅ **비대칭 fairness**: `fair_f_scale=1.0`, `fair_m_scale=0.5`
2. ✅ **Epsilon 스케줄**: warmup → hold → cooldown (0.05 → 0.10 → 0.09)
3. ✅ **Beta 스케줄**: 0.5 → 0.6 (적당한 증가)
4. ✅ **단방향 Wasserstein**: `ReLU(male - female)`

### 8th 신규 개선: 후반부 Wasserstein 강화

#### 개선: `lambda_w_boost` (후반부 Wasserstein 강화)
```python
# 20 epoch 이후 lambda_w 증가
lambda_w: 0.2 → 0.3 (epoch >= 20)
```
- **동기**: 7th에서 AR Gap이 60% 감소 (0.0081 → 0.0032)했으나 AP Gap 개선은 미미 (0.1063 → 0.1059)
- **메커니즘**: 후반부에 score alignment 압력을 강화하여 AP Gap 추가 개선
- **왜 후반부만?**: 초반부는 detection/fairness 학습에 집중해야 안정적

| epoch | lambda_w (7th) | lambda_w (8th) |
|-------|----------------|----------------|
| 0-19 | 0.2 (고정) | 0.2 (동일) |
| 20-23 | 0.2 (고정) | **0.3** (boost) |

---

## 수학적 정리: 8th Loss Function

### Generator Loss (L_G)
```
L_G = λ_fair × L_fair + β(t) × L_det + λ_w(t) × L_W

where:
  L_fair = fair_f_scale × L_f + fair_m_scale × L_m   # 비대칭 (1.0:0.5)
  L_det  = det_loss_f + det_loss_m                    # detection
  L_W    = mean(ReLU(sorted_m - sorted_f))            # 단방향
  
  β(t)   = 0.5 + 0.1 × (t/T)                          # 0.5 → 0.6
  λ_w(t) = 0.2 (if t < 20) else 0.3                   # 후반부 boost
  ε(t)   = warmup → hold → cooldown (0.05→0.10→0.09)
```

### 하이퍼파라미터 비교

| 파라미터 | 7th | 8th | 변경 이유 |
|----------|-----|-----|-----------|
| `epochs` | 24 | 24 | 동일 유지 |
| `epsilon` warmup | 8 epochs | 8 epochs | 동일 |
| `epsilon` hold | 8 epochs | 8 epochs | 동일 |
| `epsilon` cooldown | 8 epochs | 8 epochs | 동일 |
| `epsilon_min` | 0.09 | 0.09 | 동일 |
| `beta` | 0.5 → 0.6 | 0.5 → 0.6 | 동일 |
| `fair_f_scale` | 1.0 | 1.0 | 동일 |
| `fair_m_scale` | 0.5 | 0.5 | 동일 |
| `lambda_w` | 0.2 (고정) | 0.2 → 0.3 | **boost** |
| `lambda_w_boost` | - | 0.3 | **신규** |
| `lambda_w_boost_epoch` | - | 20 | **신규** |

---

## 기대 효과

### 정량적 목표
| 지표 | Baseline | 7th | 8th 목표 |
|------|----------|-----|----------|
| Female AP | 0.404 | 0.408 (+0.0034) | **0.410+** |
| Female AR | 0.826 | 0.833 (+0.0070) | **0.835+** |
| Male AP | 0.511 | 0.514 (+0.0029) | 0.513+ |
| Male AR | 0.834 | 0.836 (+0.0021) | 0.835+ |
| AP Gap | 0.1063 | 0.1059 | **< 0.103** |
| AR Gap | 0.0081 | 0.0032 | **< 0.003** |

### 핵심 개선 목표
1. **AP Gap 추가 개선**: 후반부 lambda_w 강화로 score alignment 압력 증가
2. **7th 성과 유지**: 모든 지표 baseline 대비 향상 유지
3. **안정성**: 검증된 7th 구조 기반으로 리스크 최소화

---

## 실행 방법

### 기본 실행 (모든 설정이 default로 적용)
```bash
cd /home/dohyeong/Desktop/faap_gan
python train_faap_wgan_GD_8th.py
```

### 단일 GPU 지정 (batch_size 조정)
```bash
cd /home/dohyeong/Desktop/faap_gan
CUDA_VISIBLE_DEVICES=2 python train_faap_wgan_GD_8th.py --batch_size 8
```

### 분산 학습 (DDP)
```bash
cd /home/dohyeong/Desktop
torchrun --nproc_per_node=2 --master_port=29500 \
  -m faap_gan.train_faap_wgan_GD_8th \
  --distributed
```

### 하이퍼파라미터 변경 예시
```bash
# Lambda_w boost 시점 변경
python train_faap_wgan_GD_8th.py --lambda_w_boost_epoch 18

# Lambda_w boost 값 변경
python train_faap_wgan_GD_8th.py --lambda_w_boost 0.35

# 복합 변경
python train_faap_wgan_GD_8th.py \
  --lambda_w_boost_epoch 18 \
  --lambda_w_boost 0.35 \
  --batch_size 8
```

### 평가
```bash
python eval_faap.py \
  --dataset_root /home/dohyeong/Desktop/faap_dataset \
  --detr_checkpoint /home/dohyeong/Desktop/detr/detr-r50-e632da11.pth \
  --generator_checkpoint faap_outputs/faap_outputs_gd_8th/checkpoints/epoch_0023.pth \
  --epsilon 0.09 \
  --split test \
  --results_path faap_outputs/faap_outputs_gd_8th/test_metrics_epoch_0023.json
```

---

## 로그 분석 가이드

### 핵심 모니터링 지표
```jsonl
{
  "epoch": 23,
  "epsilon": 0.09,           # cooldown 완료
  "beta": 0.6,               # schedule 완료
  "lambda_w": 0.3,           # boost 적용 확인 (epoch >= 20)
  "fair_f_scale": 1.0,       # 여성 fairness 유지
  "fair_m_scale": 0.5,       # 남성 fairness 절반
  "obj_score_f": 0.15,       # 여성 detection score
  "obj_score_m": 0.15,       # 남성 detection score
  "g_w": 0.02                # Wasserstein loss (낮을수록 score 정렬됨)
}
```

### 성공 신호
- `g_w` (Wasserstein loss)가 epoch 20 이후 감소 추세
- `obj_score_f` ≈ `obj_score_m` (score 차이 감소)
- Test에서 모든 AP/AR이 baseline 대비 양수 delta

### 문제 신호
- `g_w`가 epoch 20 이후에도 감소하지 않음 → `lambda_w_boost` 0.4로 증가 고려
- `obj_score_f/m` 모두 하락 → boost가 너무 강함, `lambda_w_boost` 0.25로 감소
- Male AP delta가 음수로 전환 → `fair_m_scale` 0.6으로 증가 고려

---

## 7th → 8th 변경 핵심 요약

| 항목 | 7th | 8th |
|------|-----|-----|
| 기반 | 6th 개선 | **7th 기반** |
| 핵심 변경 | 비대칭 fairness | **후반부 λ_w boost** |
| 복잡도 | 낮음 | 매우 낮음 (1개 파라미터 추가) |
| 리스크 | 검증됨 | 낮음 (검증된 7th 유지) |
| 목표 | 전체 성능 개선 | **AP Gap 추가 개선** |

---

## 요약: 8th가 4th를 개선하는 핵심 메커니즘

```
┌─────────────────────────────────────────────────────────────┐
│                    8th vs 4th 핵심 차이                     │
├─────────────────────────────────────────────────────────────┤
│ 1. Female Detection 보정                                    │
│    4th: β × (det_f + det_m)                                │
│    8th: β × det_m + (β + 0.15) × det_f                     │
│    → 여성 검출에 15% 추가 학습 신호                          │
├─────────────────────────────────────────────────────────────┤
│ 2. Wasserstein 점진적 강화                                  │
│    4th: λ_w = 0.2 (고정)                                   │
│    8th: λ_w = 0.1 → 0.3 (12 epoch warmup)                  │
│    → 안정적 기초 학습 후 fairness 강화                       │
├─────────────────────────────────────────────────────────────┤
│ 3. Gap 적응형 스케일링                                      │
│    4th: 고정 가중치                                        │
│    8th: gap 크면 더 강하게, 작으면 유지                      │
│    → 상황 적응적 효율 학습                                  │
└─────────────────────────────────────────────────────────────┘
```

**결론**: 5th/6th/7th의 실패 요인(데이터 제한, 과도한 스케줄링, 비대칭 fairness)을 
회피하면서, 4th의 성공 요소를 유지하고 targeted한 개선만 추가하여 
Female AP/AR 향상과 성별 간 gap 축소를 목표로 합니다.
