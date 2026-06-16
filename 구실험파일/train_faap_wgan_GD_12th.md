# FAAP 12th: Discriminator-Free Progressive Alignment

**작성일**: 2026년 1월 6일

---

## 목차
1. [11th vs 12th 핵심 차이점](#1-11th-vs-12th-핵심-차이점)
2. [12th 설계 철학](#2-12th-설계-철학)
3. [핵심 혁신 요소](#3-핵심-혁신-요소)
4. [손실 함수 구조](#4-손실-함수-구조)
5. [하이퍼파라미터](#5-하이퍼파라미터)
6. [예상 결과 및 실험 계획](#6-예상-결과-및-실험-계획)

---

## 1. 11th vs 12th 핵심 차이점

| 구분 | 11th (Unified Framework) | 12th (Discriminator-Free Progressive) |
|------|--------------------------|--------------------------------------|
| **Discriminator** | ✅ 사용 | ❌ **완전 제거** |
| **Epsilon** | 0.01 고정 | 0.03 → 0.08 → 0.06 (Progressive) |
| **핵심 전략** | 모든 방법론 융합 | 단순화 + Focal + Curriculum |
| **Adversarial Loss** | ✅ 있음 | ❌ **없음** |
| **Focal Loss** | ❌ 없음 | ✅ **새로 추가** |
| **Curriculum Learning** | ❌ 없음 | ✅ **새로 추가** |
| **학습 안정성** | 중간 (GAN 포함) | **높음** (GAN 없음) |
| **복잡도** | 높음 (7개 손실) | **중간** (6개 손실) |
| **Batch Size** | 6 | **8** (메모리 여유) |

---

## 2. 12th 설계 철학

### 2.1 Discriminator 제거의 근거

Contrastive 1st 실험에서 **Discriminator 없이 AR Gap -61.73%**를 달성했습니다:

```
┌────────────────────────────────────────────────────────┐
│  Contrastive 1st (Discriminator 없음)                  │
├────────────────────────────────────────────────────────┤
│  AR Gap: 0.81% → 0.31% (-61.73%) ⭐ 최대 감소!         │
│  학습 안정성: 매우 높음                                 │
│  Mode Collapse: 없음                                   │
└────────────────────────────────────────────────────────┘
```

**Discriminator 제거의 장점**:
1. GAN 학습의 불안정성 완전 제거
2. Mode collapse 위험 없음
3. 더 단순한 학습 다이나믹스
4. 메모리 사용량 감소 → 더 큰 배치 가능

### 2.2 Progressive Schedule의 근거

GD 7th/10th의 성공적인 epsilon schedule을 기반으로:

```
┌─────────────────────────────────────────────────────────────────┐
│  Progressive Epsilon Schedule                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Epsilon                                                        │
│    ↑                                                            │
│ 0.08 ─────────────────────■■■■■■■■■■                           │
│    │                     ╱            ╲                         │
│ 0.06 ─────────────────────────────────────■■■■■■■              │
│    │               ╱                                            │
│ 0.03 ■■■■■■■■■                                                 │
│    └────────────────────────────────────────────────→ Epoch    │
│         0      8        18             30                       │
│      Warmup   Peak          Cooldown                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

- **Warmup (0-7)**: 0.03 → 0.08 점진 증가 (안정적 시작)
- **Peak (8-17)**: 0.08 유지 (최대 학습 효과)
- **Cooldown (18-29)**: 0.08 → 0.06 감소 (안정화)

---

## 3. 핵심 혁신 요소

### 3.1 Focal Score Alignment Loss (NEW)

**기존 문제점**:
- 일반 MSE/Wasserstein은 모든 샘플에 동일한 가중치
- 이미 높은 score의 샘플도 불필요하게 학습
- Female 그룹의 어려운 샘플이 충분히 개선되지 않음

**Focal Loss 솔루션**:

```python
def _focal_score_alignment_loss(female_scores, male_scores, gamma=2.0, alpha=0.75):
    """
    - p = normalized score (0~1)
    - focal_weight = (1 - p)^gamma  ← 낮은 score에 높은 가중치
    - loss = alpha × focal_weight × (target - p)^2
    """
    target_score = male_scores.mean()  # Male 평균이 목표
    
    p = female_scores.clamp(0.01, 0.99)
    focal_weight = (1 - p) ** gamma  # 핵심: 어려운 샘플에 집중
    
    diff = F.relu(target_score - p)  # 단방향
    loss = alpha * focal_weight * (diff ** 2)
    
    return loss.mean()
```

**효과**:
```
┌────────────────────────────────────────────────────────┐
│  Focal Weight (gamma=2.0)                              │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Weight                                                │
│    ↑                                                   │
│  1.0 ■                                                 │
│    │  ■                                                │
│  0.8 │   ■                                             │
│    │     ■                                             │
│  0.6 │       ■                                         │
│    │          ■                                        │
│  0.4 │            ■                                    │
│    │                ■                                  │
│  0.2 │                    ■                            │
│    │                          ■                        │
│  0.0 └───────────────────────────■──────────→ Score   │
│       0.0  0.2  0.4  0.6  0.8  1.0                     │
│                                                        │
│  → Score가 낮을수록 가중치가 기하급수적으로 증가!         │
└────────────────────────────────────────────────────────┘
```

### 3.2 Curriculum Learning

학습 단계에 따라 다른 전략 적용:

```
┌────────────────────────────────────────────────────────┐
│  Curriculum Learning Phases                            │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Phase 1: Easy First (epoch 0-9)                       │
│  ─────────────────────────────                         │
│  - 높은 confidence 샘플(>0.7) 위주 학습                 │
│  - 기본 패턴 먼저 학습                                  │
│  - Focal weight = 0.2 (낮음)                           │
│                                                        │
│  Phase 2: Full Learning (epoch 10-19)                  │
│  ─────────────────────────────                         │
│  - 전체 샘플 학습                                       │
│  - Focal weight = 0.4 (기본)                           │
│                                                        │
│  Phase 3: Hard Focus (epoch 20-29)                     │
│  ─────────────────────────────                         │
│  - 어려운 샘플에 집중                                   │
│  - Focal weight = 0.6 (증가)                           │
│  - 공정성 극대화                                        │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 3.3 Learning Rate Schedule

CosineAnnealing으로 부드러운 학습:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt_g, T_max=args.epochs, eta_min=1e-6
)
```

---

## 4. 손실 함수 구조

### 4.1 전체 구조 (단순화됨)

```python
G_Loss = β × Detection_Loss                        # Detection 보호
       + λ_contrast × Cross_Gender_Contrastive     # 특징 공간 정렬 (핵심)
       + λ_align × Mean_Alignment                  # 평균 특징 벡터 정렬
       + λ_var × Variance_Alignment                # 분산 정렬
       + λ_focal × Focal_Score_Alignment           # Focal 기반 score 정렬 (NEW)
       + λ_w × Wasserstein_1D                      # 분포 정렬 (단방향)
```

### 4.2 11th vs 12th 손실 비교

| 손실 함수 | 11th | 12th | 역할 |
|-----------|------|------|------|
| Detection Loss | ✅ | ✅ | 탐지 성능 보호 |
| Cross-Gender Contrastive | ✅ | ✅ | 특징 공간 정렬, AR Gap 감소 |
| Mean Alignment | ✅ | ✅ | 평균 특징 정렬 |
| Variance Alignment | ✅ | ✅ | 분산 정렬 |
| **Adversarial Loss** | ✅ | ❌ | **제거됨** |
| **Entropy Loss** | ✅ | ❌ | **제거됨** |
| Wasserstein 1D | ✅ | ✅ | 분포 정렬 |
| Quantile Matching | ✅ | ❌ | → Focal로 대체 |
| **Focal Score Alignment** | ❌ | ✅ | **새로 추가** |

### 4.3 손실 함수 다이어그램

```
┌─────────────────────────────────────────────────────────┐
│                      12th Architecture                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  [이미지] → [Generator] → [Perturbed 이미지]             │
│                              ↓                          │
│                         [DETR Features]                 │
│                              ↓                          │
│   ┌──────────┬───────────────┼───────────────┐          │
│   ↓          ↓               ↓               ↓          │
│ (No Disc.) [Proj.Head]  [Detection]     [Scores]       │
│              ↓               ↓               ↓          │
│          Contrastive   Detection       Wasserstein     │
│          + Alignment    Loss           + Focal         │
│                                                         │
│  ❌ Discriminator 없음 = 학습 안정성 극대화!              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 5. 하이퍼파라미터

### 5.1 Schedule 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| eps_start | 0.03 | 초기 epsilon |
| eps_peak | 0.08 | 최대 epsilon |
| eps_end | 0.06 | 최종 epsilon |
| warmup_epochs | 8 | Warmup 기간 |
| peak_epochs | 10 | Peak 유지 기간 |
| cooldown_epochs | 12 | Cooldown 기간 |
| beta | 0.5 | 초기 detection weight |
| beta_end | 0.65 | 최종 detection weight |

### 5.2 손실 가중치

| 파라미터 | 값 | 변화 (vs 11th) | 설명 |
|----------|-----|----------------|------|
| λ_contrast | 1.2 | +0.2 | Discriminator 없어 증가 |
| λ_align | 0.6 | +0.1 | 특징 정렬 강화 |
| λ_var | 0.15 | 동일 | 분산 정렬 |
| λ_focal | 0.4 | NEW | Focal 기반 score 정렬 |
| λ_w | 0.3 | 동일 | Wasserstein |

### 5.3 Focal 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| focal_gamma | 2.0 | Focal exponent (높을수록 어려운 샘플 집중) |
| focal_alpha | 0.75 | Female 가중치 |

### 5.4 Curriculum 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| curriculum_start_epoch | 10 | 전체 샘플 학습 시작 |
| curriculum_hard_epoch | 20 | 어려운 샘플 집중 시작 |
| easy_threshold | 0.7 | 쉬운 샘플 confidence 기준 |

---

## 6. 예상 결과 및 실험 계획

### 6.1 예상 성능

```
┌────────────────────────────────────────────────────────┐
│  12th 예상 결과                                         │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Male AP:   51.08% → ~51.60% (+0.52%)                 │
│  Female AP: 40.45% → ~40.95% (+0.50%)                 │
│  AP Gap:    10.63% → ~10.65% (+0.02%p) 또는 감소       │
│                                                        │
│  Male AR:   83.39% → ~83.75% (+0.36%)                 │
│  Female AR: 82.58% → ~83.50% (+0.92%)                 │
│  AR Gap:    0.81% → ~0.25% (-0.56%p, -69.14%)         │
│                                                        │
│  핵심 기대:                                             │
│  - AR Gap: Contrastive 1st 수준 (~60% 감소)            │
│  - AP Gap: GD 7th 수준 (유지 또는 소폭 감소)            │
│  - 학습 안정성: 매우 높음 (Discriminator 없음)          │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### 6.2 vs 기존 방법론 예상 비교

| 지표 | Contrastive 1st | GD 7th | GD 10th | **12th (예상)** |
|------|-----------------|--------|---------|-----------------|
| Male AP Δ | +0.36% | +0.29% | +0.90% | **+0.52%** |
| Female AP Δ | +0.18% | +0.34% | +0.79% | **+0.50%** |
| AP Gap 변화 | +1.79% | **-0.42%** | +1.03% | **±0.2%** |
| AR Gap 변화 | **-61.73%** | -60.60% | -22.22% | **-69%** |
| 학습 안정성 | 높음 | 중간 | 중간-높음 | **매우 높음** |

### 6.3 12th의 강점

1. **학습 안정성**: Discriminator 제거로 GAN 불안정성 완전 해결
2. **AR Gap 감소**: Contrastive 1st의 강점 계승 + Focal로 강화
3. **메모리 효율**: Discriminator 제거로 더 큰 배치 가능
4. **해석 가능성**: Focal Loss는 Quantile보다 직관적
5. **Curriculum Learning**: 점진적 난이도 증가로 안정적 학습

### 6.4 실험 실행 명령어

```bash
# 기본 실행
python train_faap_wgan_GD_12th.py

# Custom epsilon (fixed)
python train_faap_wgan_GD_12th.py --epsilon 0.08

# Focal 파라미터 조정
python train_faap_wgan_GD_12th.py \
    --focal_gamma 2.5 \
    --focal_alpha 0.8

# 더 큰 배치 (Discriminator 없어 가능)
python train_faap_wgan_GD_12th.py --batch_size 12
```

### 6.5 후속 실험 방향

12th 결과에 따라:

1. **AR Gap > 예상**: `focal_gamma` 증가 (2.5 → 3.0)
2. **AP Gap 증가**: `lambda_w` 증가 (0.3 → 0.4)
3. **절대 성능 낮음**: `beta_end` 증가 (0.65 → 0.7)
4. **학습 불안정**: `warmup_epochs` 증가 (8 → 10)

---

## 7. 결론

### 12th의 핵심 철학

> **"복잡한 것이 항상 좋은 것은 아니다"**

11th가 모든 방법론을 융합했다면, 12th는 **핵심만 남기고 단순화**했습니다:

- ❌ Discriminator 제거 → 학습 안정성 ↑
- ✅ Contrastive 유지 → AR Gap 감소
- ✅ Wasserstein 유지 → AP Gap 제어
- ✅ Focal 추가 → 어려운 샘플 집중
- ✅ Curriculum 추가 → 점진적 학습

### 기대 효과

```
┌────────────────────────────────────────────────────────┐
│  12th = Contrastive 1st의 안정성                        │
│       + GD 7th/10th의 성능                              │
│       + Focal/Curriculum의 효율성                       │
└────────────────────────────────────────────────────────┘
```

---

*본 문서는 FAAP 12th 방법론의 설계 근거와 예상 결과를 정리한 것입니다. 실제 실험 결과는 `10th이후 실험결과_종합정리.md`에 추가될 예정입니다.*
