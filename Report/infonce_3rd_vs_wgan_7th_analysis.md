# FAAP 실험 분석 보고서

## 1. 파이프라인 비교 분석

### 1.1 train_faap_simclr_infonce_3rd.py (Gender-Aware Score-Based Contrastive)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        3rd Version 파이프라인                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Input Image                                                                │
│       ↓                                                                     │
│  PerturbationGenerator (ε=0.10 고정)                                        │
│       ↓                                                                     │
│  SimCLR Augmentation (ColorJitter)                                          │
│       ↓                                                                     │
│  Frozen DETR → outputs + features                                           │
│       ↓                                                                     │
│  ┌────────────────┐     ┌────────────────────────────────────────┐         │
│  │ Detection Score│     │ SimCLRProjectionHead                   │         │
│  │ (Top-K=10)     │     │ features → MLP → L2 normalize         │         │
│  └───────┬────────┘     └──────────────────┬─────────────────────┘         │
│          ↓                                  ↓                               │
│  ┌───────────────────────────────────────────────────────────────┐         │
│  │ Gender-Aware Score-Based Contrastive Loss                     │         │
│  │ • Anchor: 여성 이미지 (Female)                                 │         │
│  │ • Positive: 남성 이미지 (Male)                                 │         │
│  │ • Negative: 다른 여성 이미지                                   │         │
│  │ • Adaptive Weighting: w = 0.5 + σ(5*(score_m - score_f))      │         │
│  │ • Loss = 1.5 * L(F→M) + 0.5 * L(M→F)                          │         │
│  └───────────────────────────────────────────────────────────────┘         │
│                                                                             │
│  Total Loss = λ_c * L_contrastive + λ_w * L_wasserstein + β * L_det        │
│             = 1.0 * L_contrastive + 0.2 * L_wasserstein + β * L_det        │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 train_faap_wgan_GD_7th.py (WGAN-GD Adversarial)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        7th Version 파이프라인                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Input Image (성별 분리: Female/Male)                                        │
│       ↓                                                                     │
│  PerturbationGenerator (ε: 0.05→0.10→0.09 스케줄링)                          │
│       ↓                                                                     │
│  Frozen DETR → outputs + features                                           │
│       ↓                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐         │
│  │ GenderDiscriminator (4회 반복 업데이트)                        │         │
│  │ • D_loss = CE(D(feat_f), 1) + CE(D(feat_m), 0)                │         │
│  └───────────────────────────────────────────────────────────────┘         │
│       ↓                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐         │
│  │ Generator Fairness Loss (Adversarial)                         │         │
│  │ • fairness_f = -(CE + α*Entropy) for Female                   │         │
│  │ • fairness_m = -(CE + α*Entropy) for Male                     │         │
│  │ • L_fair = 1.0 * fair_f + 0.5 * fair_m                        │         │
│  └───────────────────────────────────────────────────────────────┘         │
│       ↓                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐         │
│  │ 1D Wasserstein Loss (단방향)                                   │         │
│  │ • Female score → Male score 방향으로만 끌어올림                 │         │
│  │ • L_w = ReLU(sorted_m - sorted_f).mean()                      │         │
│  └───────────────────────────────────────────────────────────────┘         │
│                                                                             │
│  Total Loss = λ_fair * L_fair + β * L_det + λ_w * L_w                       │
│             = 2.0 * L_fair + β * L_det + 0.2 * L_w                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 핵심 차이점 비교

| 구분 | 7th (WGAN-GD) | 3rd (Gender-Aware Contrastive) |
|------|---------------|-------------------------------|
| **학습 방식** | Adversarial (GAN) | Contrastive Learning |
| **Discriminator** | GenderDiscriminator (학습) | 없음 |
| **Projection Head** | 없음 | SimCLRProjectionHead (학습) |
| **공정성 손실** | Adversarial + Entropy | InfoNCE (F→M, M→F) |
| **Anchor 정의** | 성별 기반 분리 | 명시적: Female=Anchor, Male=Positive |
| **Epsilon** | 0.05→0.10→0.09 (스케줄링) | 0.10 (고정) |
| **Score 활용** | Wasserstein 정렬만 | Adaptive Weighting에 통합 |
| **비대칭 가중치** | fair_f=1.0, fair_m=0.5 | L(F→M)=1.5, L(M→F)=0.5 |
| **안정성** | GAN 학습 불안정 가능 | Contrastive 학습 안정적 |

### 2.1 핵심 설계 철학 차이

**7th (Adversarial)**
- Discriminator가 성별 구분 → Generator가 이를 혼란시키는 feature 생성
- GAN 학습의 min-max 게임 구조
- 암묵적 가정: "성별 구분 불가능 = 공정함"

**3rd (Contrastive)**
- **명시적 연결**: 여성 feature → 남성 feature 방향으로 직접 이동
- Score 차이에 비례한 adaptive weighting
- 직관적 가정: "저성능(여성) feature가 고성능(남성) feature와 유사해지면 공정해짐"

---

## 3. 실험 결과 비교

### 3.1 최종 에폭 (Epoch 23) 결과 비교

| Metric | Baseline | 7th (WGAN-GD) | 3rd (InfoNCE) | 변화 방향 |
|--------|----------|---------------|---------------|-----------|
| **Male AP** | 0.511 | 0.514 (+0.003) | 0.518 (+0.007) | ↑ 개선 |
| **Female AP** | 0.404 | 0.408 (+0.004) | 0.408 (+0.004) | → 동일 |
| **AP Gap** | **0.107** | **0.106** (-0.5%) | **0.109** (+2.8%) | ⚠️ 악화 |
| **Male AR** | 0.834 | 0.836 (+0.002) | 0.838 (+0.004) | ↑ 개선 |
| **Female AR** | 0.826 | 0.833 (+0.007) | 0.831 (+0.005) | ↑ 개선 |
| **AR Gap** | **0.008** | **0.003** (-62%) | **0.008** (0%) | 7th 우수 |

### 3.2 3rd (InfoNCE) 에폭별 상세 결과

| Epoch | Male AP | Female AP | AP Gap | AR Gap | 비고 |
|-------|---------|-----------|--------|--------|------|
| Baseline | 0.511 | 0.404 | 0.107 | 0.008 | - |
| **Epoch 10** | **0.517** | **0.413** | **0.105** | **0.004** | **Best** |
| Epoch 23 | 0.518 | 0.408 | 0.109 | 0.008 | Final |

### 3.3 핵심 발견

**7th (WGAN-GD) 결과**
- AR Gap: 0.008 → 0.003 (**62% 개선** ✓)
- AP Gap: 0.107 → 0.106 (미미한 개선)
- 전체 성능 유지하면서 AR 공정성 크게 개선

**3rd (InfoNCE) 결과**
- **Epoch 10이 최적**: AP Gap 0.105 (1.5% 개선), AR Gap 0.004 (50% 개선)
- 과학습 문제: Epoch 23에서 AP Gap이 baseline보다 악화
- Male AP 향상 폭이 Female AP보다 큼 → 역효과

---

## 4. 실패 분석 (3rd Version)

### 4.1 주요 원인

1. **Score Gap 역전 현상**
   - 학습 로그 분석: score_gap (M-F)가 음수 (-0.007 ~ -0.014)
   - 의미: 남성 score < 여성 score (기대와 반대)
   - Contrastive Loss가 잘못된 방향으로 작용

2. **Adaptive Weighting의 역효과**
   - score_m - score_f가 음수 → weight가 낮아짐
   - 저성능 남성 샘플이 positive로 작용 → 역효과

3. **과학습 (Overfitting)**
   - Epoch 10 이후 성능 하락
   - Projection Head가 training set에 과적합

### 4.2 2nd → 3rd 핵심 변경의 문제점

```
2nd Version: Score만으로 Low/High 분리 (성별 무관)
  → 실패 원인: obj_score_f ≈ obj_score_m으로 성별 구분 불가

3rd Version: Gender + Score 명시적 결합
  → 새로운 문제: Score가 기대와 반대 방향으로 분포
```

---

## 5. 결론 및 향후 방향

### 5.1 현재 결과 요약

| 실험 | AP Gap 개선 | AR Gap 개선 | 권장 |
|------|-------------|-------------|------|
| 7th (WGAN-GD) | △ (0.5%) | ✓ (62%) | **현재 최선** |
| 3rd (InfoNCE) | ✗ (악화) | △ (조건부) | 추가 연구 필요 |

### 5.2 권장 사항

1. **7th 버전 채택**: AR Gap 개선이 명확, 안정적
2. **3rd 버전 개선 방향**:
   - Early Stopping (Epoch 10 사용)
   - Score normalization 추가
   - Positive/Negative 샘플링 전략 재설계

### 5.3 추가 실험 제안

- **4th Version 설계**: Score 기반 + Gender-Aware의 하이브리드 접근
- Temperature 조정 실험 (0.07 → 0.1 또는 0.05)
- Projection Head 용량 축소 (과적합 방지)

---

**작성일**: 2026-01-27
**실험 환경**: DETR-R50, FAAP Dataset, 24 Epochs
