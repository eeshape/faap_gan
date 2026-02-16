# FAAP 실험 보고서: 공정성 개선을 위한 Perturbation 학습

**작성일**: 2026-01-30
**연구 목표**: 입력 이미지에 작은 perturbation(ε=0.05~0.10)을 추가하여 DETR Object Detector의 성별 간 검출 성능 격차(AP Gap)를 줄이는 것

---

## 1. Baseline 성능

| 성별 | AP | AR |
|------|-----|-----|
| Male | 0.511 | 0.834 |
| Female | 0.404 | 0.826 |
| **Gap (M-F)** | **0.106** | **0.008** |

---

## 2. 실험 요약

### 2.1 버전별 핵심 아이디어 및 결과

| 버전 | 접근법 | Best Epoch | AP Gap | Δ AP Gap | AR Gap | Δ AR Gap | 결과 |
|------|--------|------------|--------|----------|--------|----------|------|
| **7th** | WGAN-GD | 23 | 0.1059 | -0.04pp | 0.0032 | -0.49pp | 미미 |
| **3rd** | Gender-Aware InfoNCE | 3 | 0.1044 | **-0.19pp** | 0.0026 | **-0.55pp** | 가장 양호 |
| **3rd_fix1** | Fair Centroid | 9 | 0.1072 | +0.09pp | 0.0074 | -0.07pp | 실패 |
| **3rd_fix2** | Male-Anchored | 29 | 0.1050 | -0.13pp | 0.0057 | -0.24pp | 미미 |
| **fix3** | Direct Boost | 13 | 0.1119 | +0.56pp | 0.0063 | -0.18pp | 악화 |

---

## 3. 데이터 증강 (Augmentation)

### 3.1 사용된 증강 기법

**3rd, 3rd_fix1, 3rd_fix2 버전에서 SimCLR-style 증강 적용:**

```python
class SimCLRAugmentation(nn.Module):
    # strength = "medium" (기본값)
    self.transform = T.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.1
    )
```

| 버전 | 증강 강도 | 적용 시점 |
|------|----------|----------|
| 3rd | medium | 학습 시 perturbed 이미지에 적용 |
| 3rd_fix1 | weak | brightness/contrast/saturation=0.2, hue=0.05 |
| 3rd_fix2 | medium | 3rd와 동일 |
| 7th, fix3 | 없음 | 증강 미적용 |

### 3.2 증강 목적

- Contrastive learning에서 positive/negative pair의 다양성 확보
- 공정성 개선 목적이 아닌, **representation learning 품질 향상** 목적
- 평가(eval) 시에는 증강 미적용

---

## 4. 각 버전 상세 분석

### 4.1 7th 버전: WGAN-GD (Wasserstein GAN)

**핵심 아이디어**
- GAN 기반 적대적 학습
- Discriminator가 성별 분류, Generator는 이를 속이도록 학습
- Wasserstein distance 기반 분포 정렬

**구현 특징**
```python
# Epsilon Scheduling
epsilon: 0.05 → 0.10 (warmup 8 epochs)
       → 0.10 (hold 6 epochs)
       → 0.09 (cooldown 10 epochs)

# Loss Weights
lambda_fair = 2.0      # 공정성 손실
lambda_w = 0.2         # Wasserstein 가중치
beta = 0.5 → 0.6       # Detection 보존 손실
```

**결과**
| Epoch | Male AP | Female AP | AP Gap | Δ AP Gap |
|-------|---------|-----------|--------|----------|
| 23 | 0.514 | 0.408 | 0.1059 | -0.0004 |

**분석**
- 가장 간단한 접근법
- 증강 없이 적대적 학습만 수행
- 개선폭 매우 작음 (약 0.4%)

---

### 4.2 3rd 버전: Gender-Aware Score-Based Contrastive

**핵심 아이디어**
```
L = -log(exp(sim(z_f, z_m)/τ) / [exp(sim(z_f, z_m)/τ) + Σ exp(sim(z_f, z_f')/τ)])
```

- Anchor: Female 이미지
- Positive: Male 이미지 (고성능)
- Negative: 다른 Female 이미지
- Score 차이에 비례한 Adaptive Weighting

**구현 특징**
- Temperature: 0.07
- Projection Head: 256 → 256 → 128 (L2 normalized)
- 비대칭 가중치: F→M (1.5), M→F (0.5)
- **증강: ColorJitter (medium)**

**결과**
| Epoch | Male AP | Female AP | AP Gap | Δ AP Gap |
|-------|---------|-----------|--------|----------|
| 3 | 0.511 | 0.406 | 0.1044 | **-0.0019** |
| 10 | 0.517 | 0.412 | 0.1048 | -0.0015 |

**분석**
- 가장 좋은 결과를 보였으나, 개선폭이 매우 작음 (약 1.8% 개선)
- Epoch 3 이후 빠른 overfitting 발생
- Score Gap (train)과 실제 AP Gap (eval) 불일치 관측

---

### 4.3 3rd_fix1 버전: Fair Centroid Contrastive

**3rd 버전 실패 분석**
1. Female → Male 당김이 오히려 Male 편향 강화
2. Score-AP 불일치: Train score_gap ≈ -0.01 (F > M), 실제 AP는 M > F
3. Epoch 3에서 최고 → 빠른 overfitting

**핵심 변경**
```
Fair Centroid = 0.7 × Centroid_F + 0.3 × Centroid_M
L_fair = ||z_f - FairCentroid||² + ||z_m - FairCentroid||²
```

- EMA 기반 Centroid 업데이트 (momentum=0.9)
- 대칭적 가중치: F→M (1.0), M→F (1.0)
- Dropout (0.1) 추가 regularization
- Cosine Annealing LR Scheduler
- **증강: ColorJitter (weak)**

**결과**
| Epoch | Male AP | Female AP | AP Gap | Δ AP Gap |
|-------|---------|-----------|--------|----------|
| 9 | 0.519 | 0.412 | 0.1072 | +0.0009 |

**실패 원인 분석**
1. **Loss Saturation**: loss_f_align, loss_m_align이 -10에 즉시 수렴
2. **Representation Collapse**: 모든 샘플이 동일 점으로 수렴
3. **음수 Loss**: total_loss ≈ -24 → 최소화 의미 없음

---

### 4.4 3rd_fix2 버전: Male-Anchored Asymmetric Contrastive

**3rd_fix1 실패 분석**
- Fair Centroid alignment 포화 문제
- Representation collapse
- 음수 loss로 학습 불가

**핵심 변경**
```python
# Male representation 고정 (gradient 차단)
proj_m_detached = proj_m.detach()

# Female만 학습
sim_f2m = torch.mm(proj_f, proj_m_detached.t()) / temperature
```

- Fair Centroid 제거
- Male을 Anchor(고정)로 사용
- Female만 학습 (완전 비대칭)
- 3rd 버전 hyperparameter 복원
- **증강: ColorJitter (medium)**

**결과**
| Epoch | Male AP | Female AP | AP Gap | Δ AP Gap |
|-------|---------|-----------|--------|----------|
| 9 | 0.520 | 0.415 | 0.1054 | -0.0009 |
| 29 | 0.520 | 0.415 | 0.1050 | -0.0013 |

**분석**
- 3rd 대비 Female AP 향상 (+0.011)
- 하지만 Male AP도 동반 상승하여 Gap 개선 미미
- Loss가 정상 범위로 복귀

---

### 4.5 fix3 버전: Direct Confidence Boosting

**이전 접근법들의 근본적 한계 분석**
1. **Representation-Performance Gap**
   - InfoNCE는 cosine similarity 최적화 (표현 공간)
   - Detection confidence는 absolute logit magnitude에 의존
   - 표현이 비슷해져도 detection score는 개선되지 않음

2. **Information Bottleneck**
   - 100개 object query → mean pooling → 1개 vector
   - Per-object confidence 정보 손실

3. **Detection Score에 직접 gradient 없음**

**핵심 변경**
```python
# Contrastive loss 제거
# Female detection confidence를 직접 높이는 loss
L_boost = -log(female_conf + eps) * weight
weight = softmax((threshold - conf) * beta)  # Hard sample mining

# Gap reduction
L_gap = max(0, male_conf - female_conf - margin)
```

- Per-object confidence 사용 (mean pooling 우회)
- Hard sample mining: 낮은 confidence에 더 높은 가중치
- Top-K objects 선택
- **증강: 없음**

**결과**
| Epoch | Male AP | Female AP | AP Gap | Δ AP Gap |
|-------|---------|-----------|--------|----------|
| 13 | 0.520 | 0.408 | 0.1119 | **+0.0056** |

**실패 원인 분석**
- Male AP가 Female AP보다 더 많이 증가 (+0.0095 vs +0.0039)
- Perturbation이 양 성별 모두에 영향 → 이미 성능 좋은 Male이 더 수혜
- Gap이 오히려 악화됨

---

## 5. 근본적 한계 분석

### 5.1 이론적 한계

```
문제 구조:
┌─────────────────────────────────────────────────┐
│  Input Image + Perturbation  →  Frozen DETR  →  Detection  │
│        (변경 가능)                (변경 불가)        (결과)     │
└─────────────────────────────────────────────────┘
```

- DETR의 bias는 **모델 가중치**에 인코딩됨
- 우리는 **입력만** 변경 가능 (모델은 frozen)
- 이는 **편향된 판사를 바꾸지 않고, 증거만 수정**하려는 것과 유사

### 5.2 Perturbation의 구조적 문제

1. **이미지 전체에 적용**: 같은 이미지에 Male/Female 공존 시 둘 다 영향
2. **ε=0.05~0.10의 제약**: 픽셀 값 범위 [0,1]에서 5~10%만 변경 가능
3. **Gender-specific perturbation 불가**: 객체별 차등 적용 어려움

### 5.3 Contrastive Learning의 한계

```
Contrastive Learning:  Representation Space (cosine similarity)
Detection:             Logit Space (absolute magnitude)
                       ↓
              두 공간의 불일치
```

- Representation이 유사해져도 detection confidence는 개선 안됨
- Mean pooling으로 per-object 정보 손실

### 5.4 증강의 한계

- ColorJitter 증강은 **contrastive learning 품질 향상** 목적
- 성별 균형 또는 공정성 개선과 직접 관련 없음
- 증강 여부와 관계없이 결과 유사 (7th vs 3rd)

---

## 6. 실험 결과 종합 비교

| 버전 | 접근법 | 증강 | AP Gap | Δ AP Gap | Δ F_AP | Δ M_AP | Gap 개선율 |
|------|--------|------|--------|----------|--------|--------|-----------|
| Baseline | - | - | 0.1063 | - | - | - | - |
| 7th | WGAN-GD | 없음 | 0.1059 | -0.0004 | +0.003 | +0.003 | 0.4% |
| **3rd (ep3)** | InfoNCE | ColorJitter | **0.1044** | **-0.0019** | +0.002 | +0.001 | **1.8%** |
| 3rd_fix1 | Fair Centroid | ColorJitter | 0.1072 | +0.0009 | +0.008 | +0.009 | -0.8% |
| 3rd_fix2 | Male-Anchored | ColorJitter | 0.1050 | -0.0013 | +0.011 | +0.009 | 1.2% |
| fix3 | Direct Boost | 없음 | 0.1119 | +0.0056 | +0.004 | +0.010 | -5.3% |

---

## 7. 결론 및 향후 방향

### 7.1 현재 접근법의 한계

1. **Input perturbation만으로 model bias를 의미있게 줄이기 어려움**
   - 최선의 결과도 약 1.8% 개선에 불과
   - 다양한 loss function 시도에도 큰 변화 없음

2. **Perturbation이 양 성별에 비차별적 영향**
   - 이미 성능 좋은 Male이 더 많은 수혜
   - Gender-specific 적용 방법 필요

3. **표현 학습과 검출 성능 간의 Gap**
   - Contrastive learning으로 표현은 유사해져도
   - 실제 detection confidence 개선 미미

4. **증강은 공정성 개선에 직접 기여하지 않음**
   - ColorJitter는 representation learning 품질 향상 용도
   - 증강 유무와 관계없이 결과 유사

### 7.2 대안적 접근 방향

1. **DETR Fine-tuning**: Detector 가중치 직접 수정
2. **Post-processing Calibration**: 성별별 confidence 보정
3. **Training Data Augmentation**: Female 이미지 수 증강/불균형 해소
4. **Per-object Perturbation**: 객체별 차등 적용
5. **더 큰 Epsilon**: perturbation 예산 증가 (시각적 품질 trade-off)

---

## 8. 파일 구조

```
faap_gan/
├── train_faap_wgan_GD_7th.py             # 7th WGAN 버전
├── train_faap_simclr_infonce_3rd.py      # 3rd 버전
├── train_faap_simclr_infonce_3rd_fix1.py # fix1 버전
├── train_faap_simclr_infonce_3rd_fix2.py # fix2 버전
├── train_faap_direct_boost_fix3.py       # fix3 버전
├── eval_faap.py                          # 평가 스크립트
└── faap_outputs/
    ├── faap_outputs_gd_7th/              # 7th WGAN 결과
    ├── faap_outputs_infonce_3rd/         # 3rd 결과
    ├── faap_outputs_infonce_3rd_fix1/    # fix1 결과
    ├── faap_outputs_infonce_3rd_fix2/    # fix2 결과
    └── faap_outputs_fix3/                # fix3 결과
```

---

## 부록: 주요 Hyperparameters

| Parameter | 7th | 3rd | fix1 | fix2 | fix3 |
|-----------|-----|-----|------|------|------|
| 접근법 | WGAN-GD | InfoNCE | Fair Centroid | Male-Anchored | Direct Boost |
| Temperature | - | 0.07 | 0.10 | 0.07 | - |
| Learning Rate | 1e-4 | 1e-4 | 5e-5 | 1e-4 | 5e-5 |
| Epsilon | 0.05→0.10→0.09 | 0.10 | 0.10 | 0.10 | 0.10 |
| Epochs | 24 | 24 | 15 | 30 | 20 |
| **증강** | **없음** | **ColorJitter (medium)** | **ColorJitter (weak)** | **ColorJitter (medium)** | **없음** |
| F→M Weight | - | 1.5 | 1.0 | 1.0 | - |
| M→F Weight | - | 0.5 | 1.0 | 0.0 | - |
| Projection Dim | - | 128 | 128 | 128 | - |
| Beta (Det Loss) | 0.5→0.6 | 0.5→0.6 | 0.6→0.7 | 0.5→0.6 | 0.3→0.5 |
| lambda_fair | 2.0 | - | - | - | - |
| lambda_w | 0.2 | 0.2 | 0.5 | 0.2 | - |
