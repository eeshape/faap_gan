# FAAP 13th: Adaptive Multi-Scale Contrastive Fairness

**작성일**: 2026년 1월 12일

---

## 목차
1. [핵심 분석 요약](#1-핵심-분석-요약)
2. [13th 설계 철학](#2-13th-설계-철학)
3. [7th + 11th 융합 전략](#3-7th--11th-융합-전략)
4. [핵심 혁신 요소](#4-핵심-혁신-요소)
5. [손실 함수 구조](#5-손실-함수-구조)
6. [하이퍼파라미터](#6-하이퍼파라미터)
7. [예상 결과](#7-예상-결과)

---

## 1. 핵심 분석 요약

### 1.1 이전 버전 성과 분석

| 버전 | 핵심 기여 | AP Gap | AR Gap | 특징 |
|------|----------|--------|--------|------|
| **GD 7th** | 비대칭 스케일링, 단방향 Wasserstein | **-0.38%** | -60.5% | AP Gap 감소 유일 |
| **GD 11th** | Contrastive + GAN + Quantile 융합 | - | - | 복잡한 통합 |
| **Contrastive 1st** | InfoNCE, Discriminator 제거 | - | **-61.73%** | AR Gap 최대 감소 |
| **GD 12th** | Discriminator-Free, Focal Loss | - | - | 안정적 학습 |

### 1.2 성공 요소 분석

```
┌────────────────────────────────────────────────────────────────────────┐
│                      각 버전의 핵심 성공 요소                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [GD 7th] AP Gap -0.38% 달성                                           │
│  ├── 비대칭 Fairness: Female 1.0, Male 0.5                             │
│  ├── 단방향 Wasserstein: Female→Male만 정렬                             │
│  ├── Epsilon Schedule: warmup → hold → cooldown                        │
│  └── 긴 학습 (24 epochs)                                               │
│                                                                        │
│  [Contrastive 1st] AR Gap -61.73% 달성                                 │
│  ├── Discriminator 제거 → 안정적 학습                                   │
│  ├── InfoNCE Loss → 성별 정보 제거                                     │
│  └── Projection Head → 특징 공간 정렬                                   │
│                                                                        │
│  [GD 11th] 종합적 접근                                                  │
│  ├── Multi-method 융합                                                 │
│  ├── Quantile Matching                                                 │
│  └── Feature Alignment (Mean + Variance)                               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. 13th 설계 철학

### 2.1 핵심 원칙

```
┌────────────────────────────────────────────────────────────────────────┐
│                        13th 설계 원칙                                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  1. Discriminator-Free (Contrastive 1st 계승)                          │
│     └── GAN 불안정성 제거, 더 큰 배치 가능                               │
│                                                                        │
│  2. Multi-Scale Contrastive (NEW)                                      │
│     └── Query / Image / Statistical 세 레벨 동시 정렬                   │
│                                                                        │
│  3. Asymmetric Alignment (7th 확장)                                    │
│     └── Female→Male 강하게, Male→Female 약하게                         │
│                                                                        │
│  4. Hard Negative Mining (NEW)                                         │
│     └── 가장 다른 샘플 쌍에 집중 학습                                    │
│                                                                        │
│  5. Adaptive Loss Weighting (NEW)                                      │
│     └── Detection 성능 기반 자동 가중치 조정                             │
│                                                                        │
│  6. Curriculum Schedule (7th 계승)                                     │
│     └── Epsilon: warmup → hold → cooldown                              │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 왜 이 조합인가?

```
AP Gap 감소를 위해:
├── 7th의 비대칭 스케일링 + 단방향 Wasserstein (검증됨)
├── NEW: Asymmetric Alignment Loss (비대칭 개념 확장)
└── NEW: Hard Negative Mining (어려운 경우 집중)

AR Gap 감소를 위해:
├── Contrastive 1st의 InfoNCE 기반 접근
├── NEW: Multi-Scale Contrastive (세밀한 정렬)
└── Feature Alignment (Mean + Variance)

Detection 보호를 위해:
├── 7th의 Curriculum Epsilon Schedule
├── Beta 선형 증가 (후반부 Detection 강화)
└── NEW: Adaptive Loss Weighting (성능 기반 조정)
```

---

## 3. 7th + 11th 융합 전략

### 3.1 7th에서 가져온 요소

| 요소 | 원래 설정 | 13th 적용 | 역할 |
|------|----------|----------|------|
| 비대칭 Fairness | Female 1.0, Male 0.5 | asym_f_weight=1.0, asym_m_weight=0.3 | Male 보호 |
| 단방향 Wasserstein | Female→Male | 그대로 유지 | 분포 정렬 |
| Epsilon Schedule | warmup→hold→cooldown | 0.03→0.08→0.06 | 점진적 학습 |
| Beta Schedule | 0.5→0.6 | 0.4→0.6 | Detection 보호 |

### 3.2 11th에서 가져온 요소

| 요소 | 원래 설정 | 13th 적용 | 역할 |
|------|----------|----------|------|
| Projection Head | 2-layer MLP | Multi-Scale (3 branches) | 특징 투영 |
| Contrastive Loss | Cross-Gender | Multi-Scale Contrastive | 특징 정렬 |
| Feature Alignment | Mean + Variance | Asymmetric Alignment | 통계 정렬 |

### 3.3 새로운 요소 (13th 혁신)

| 요소 | 설명 | 기대 효과 |
|------|------|----------|
| **Multi-Scale Contrastive** | Query/Image/Stat 세 레벨 | 더 세밀한 특징 정렬 |
| **Hard Negative Mining** | 가장 다른 샘플 쌍 집중 | 어려운 경우 해결 |
| **Adaptive Weighting** | Detection 성능 기반 조정 | 자동 균형 유지 |

---

## 4. 핵심 혁신 요소

### 4.1 Multi-Scale Projection Head

```python
class MultiScaleProjectionHead(nn.Module):
    """세 가지 스케일의 특징 추출"""
    
    def forward(self, x):  # x: (batch, num_queries, feat_dim)
        
        # 1. Query-level: 각 query 개별 투영
        query_feat = self.query_proj(x)  # (batch, num_queries, proj_dim)
        
        # 2. Image-level: query 평균 후 투영
        pooled = x.mean(dim=1)
        image_feat = self.image_proj(pooled)  # (batch, proj_dim)
        
        # 3. Statistical-level: 평균+표준편차 연결
        stat_input = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=-1)
        stat_feat = self.stat_proj(stat_input)  # (batch, proj_dim)
        
        return query_feat, image_feat, stat_feat  # 모두 L2 정규화됨
```

**왜 세 가지 스케일인가?**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Multi-Scale의 상보적 역할                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Image-level (가중치 0.5) - 핵심                                        │
│  └── 이미지 전체의 성별 편향 제거                                        │
│  └── AR Gap 감소에 직접적 기여                                          │
│                                                                        │
│  Query-level (가중치 0.3)                                               │
│  └── 개별 객체 탐지 query의 편향 제거                                    │
│  └── 세밀한 탐지 성능 향상                                              │
│                                                                        │
│  Statistical-level (가중치 0.2)                                         │
│  └── 분포 형태 (평균 + 분산) 정렬                                        │
│  └── 전체적인 특징 분포 유사화                                          │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Asymmetric Alignment Loss

```python
def _asymmetric_alignment_loss(feat_f, feat_m, f_weight=1.0, m_weight=0.3):
    """비대칭 정렬: Female은 강하게, Male은 약하게"""
    
    mean_f = feat_f.mean(dim=(0,1))  # Female 그룹 평균
    mean_m = feat_m.mean(dim=(0,1))  # Male 그룹 평균
    
    # Female → Male (강하게, Male은 detach)
    f_to_m_loss = ((mean_f - mean_m.detach()) ** 2).mean()
    
    # Male → Female (약하게, Female은 detach)
    m_to_f_loss = ((mean_m - mean_f.detach()) ** 2).mean()
    
    return f_weight * f_to_m_loss + m_weight * m_to_f_loss
```

**비대칭의 핵심 원리:**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    비대칭 정렬의 그래디언트 흐름                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  Female → Male (f_weight = 1.0)                                        │
│  ├── Male은 detach → Male 특징 변화 없음                                │
│  ├── Female만 Male 방향으로 이동                                        │
│  └── 효과: Female 성능 향상, Male 성능 유지                              │
│                                                                        │
│  Male → Female (m_weight = 0.3)                                        │
│  ├── Female은 detach → Female 특징 변화 없음                            │
│  ├── Male이 약간만 Female 방향으로 이동                                  │
│  └── 효과: 전체 정렬 보조, Male 성능 크게 희생 안함                       │
│                                                                        │
│  ⇒ 결과: Male 보호 + Female 집중 개선 (7th 성공 요소)                    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Hard Negative Mining

```python
def _hard_negative_mining_loss(image_f, image_m, hard_ratio=0.3, temperature=0.05):
    """가장 다른 샘플 쌍에 집중"""
    
    # 유사도 계산
    sim = torch.mm(image_f, image_m.t())  # (N_f, N_m)
    
    # 각 Female에 대해 가장 다른 Male k개 선택
    k_m = int(n_m * hard_ratio)
    _, hard_m_idx = sim.topk(k_m, dim=1, largest=False)  # 최저 유사도
    
    # Hard pairs에 대해 contrastive loss 적용
    hard_sim = torch.gather(sim, 1, hard_m_idx) / temperature
    loss = -torch.logsumexp(hard_sim, dim=1).mean()
    
    return loss
```

**왜 Hard Negative Mining인가?**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Hard Negative Mining의 효과                          │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  문제: 일반 Contrastive Loss는 모든 쌍에 동일한 중요도                    │
│        → 이미 유사한 쌍에 불필요한 학습 리소스 소비                        │
│                                                                        │
│  해결: 가장 다른(어려운) 쌍에 집중                                        │
│        → 학습 효율 극대화                                               │
│        → 극단적 편향 케이스 우선 해결                                    │
│                                                                        │
│  예시:                                                                  │
│  ┌─────────────────────────────────────────────────────┐               │
│  │ Female 샘플 A                                       │               │
│  │ ├── Male B: 유사도 0.8 (쉬움) → 학습 제외           │               │
│  │ ├── Male C: 유사도 0.6 (쉬움) → 학습 제외           │               │
│  │ └── Male D: 유사도 0.2 (어려움) → 집중 학습 ⭐       │               │
│  └─────────────────────────────────────────────────────┘               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.4 Adaptive Loss Weighting

```python
def _adaptive_weight(current_score, threshold, base_weight, scale=1.5):
    """Detection 성능 기반 자동 가중치 조정"""
    
    if current_score >= threshold:
        return base_weight  # 성능 좋으면 기본 가중치
    
    # 성능이 떨어지면 Detection 가중치 증가
    ratio = (threshold - current_score) / threshold
    return base_weight * (1 + ratio * (scale - 1))
```

**Adaptive Weighting 시나리오:**

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Adaptive Weighting 동작                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [시나리오 1] Detection 성능 양호 (obj_score >= 0.45)                    │
│  ├── adaptive_beta = 기본값 (current_beta)                             │
│  └── Fairness 손실에 더 많은 가중치                                     │
│                                                                        │
│  [시나리오 2] Detection 성능 저하 (obj_score < 0.45)                     │
│  ├── adaptive_beta = 기본값 × 1.5 (최대)                               │
│  └── Detection 보호 강화 → 성능 회복                                    │
│                                                                        │
│  효과: 수동 튜닝 없이 자동으로 Detection/Fairness 균형 유지               │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. 손실 함수 구조

### 5.1 전체 손실 함수

```python
# Multi-Scale Contrastive (세 레벨 가중 합)
contrast_total = 0.5 * image_contrast   # 핵심
              + 0.3 * query_contrast    # 세밀
              + 0.2 * stat_contrast     # 분포

# 최종 손실
G_Loss = adaptive_beta × Detection_Loss           # Detection 보호
       + λ_contrast × contrast_total              # Multi-Scale Contrastive
       + λ_asym × Asymmetric_Alignment           # 비대칭 정렬
       + λ_hard × Hard_Negative_Mining           # 어려운 샘플
       + λ_w × Wasserstein_1D                    # 분포 정렬 (단방향)
```

### 5.2 손실 비교표

| 손실 | 7th | 11th | 12th | **13th** |
|------|-----|------|------|----------|
| Detection | ✅ β schedule | ✅ 고정 | ✅ 고정 | ✅ **Adaptive** |
| Adversarial | ✅ | ✅ | ❌ | ❌ |
| Contrastive | ❌ | ✅ | ✅ | ✅ **Multi-Scale** |
| Wasserstein | ✅ 단방향 | ✅ 단방향 | ✅ 단방향 | ✅ 단방향 |
| Asymmetric | 스케일만 | ❌ | ❌ | ✅ **Loss로 확장** |
| Hard Mining | ❌ | ❌ | ❌ | ✅ **NEW** |
| Focal | ❌ | ❌ | ✅ | ❌ |
| Quantile | ❌ | ✅ | ❌ | ❌ |

---

## 6. 하이퍼파라미터

### 6.1 스케줄 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| epsilon_start | 0.03 | Warmup 시작 epsilon |
| epsilon_peak | 0.08 | Peak epsilon |
| epsilon_final | 0.06 | Cooldown 후 최종 epsilon |
| warmup_epochs | 8 | Warmup 기간 |
| hold_epochs | 8 | Peak 유지 기간 |
| cooldown_epochs | 14 | Cooldown 기간 |
| beta_start | 0.4 | Detection 가중치 시작 |
| beta_final | 0.6 | Detection 가중치 최종 |

### 6.2 손실 가중치

| 파라미터 | 값 | 역할 |
|----------|-----|------|
| λ_contrast | 1.2 | Multi-Scale Contrastive |
| λ_asym | 0.8 | Asymmetric Alignment |
| λ_hard | 0.4 | Hard Negative Mining |
| λ_w | 0.3 | Wasserstein |

### 6.3 기타 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| temperature | 0.05 | Contrastive (더 sharp) |
| proj_dim | 128 | Projection 차원 |
| asym_f_weight | 1.0 | Female→Male 가중치 |
| asym_m_weight | 0.3 | Male→Female 가중치 |
| hard_ratio | 0.3 | Hard sample 비율 |
| batch_size | 8 | 배치 크기 (D 없어 여유) |

---

## 7. 예상 결과

### 7.1 목표

| 지표 | 7th 성과 | 13th 목표 |
|------|----------|-----------|
| AP Gap 감소율 | -0.38% | **-1% ~ -2%** |
| AR Gap 감소율 | -60.5% | **-65% ~ -70%** |
| Female AP Δ | +0.83% | **+1% ~ +1.5%** |
| Male AP Δ | +0.57% | **+0.5%** (유지) |
| Detection 유지 | ✅ | ✅ |

### 7.2 개선 근거

```
┌────────────────────────────────────────────────────────────────────────┐
│                    13th 개선 예상 근거                                   │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  1. Multi-Scale Contrastive                                            │
│     └── 세 레벨 동시 정렬 → Contrastive 1st보다 세밀한 정렬              │
│     └── AR Gap 추가 감소 기대 (-5% ~ -10%)                              │
│                                                                        │
│  2. Asymmetric Loss 확장                                               │
│     └── 7th의 스케일링을 Loss로 공식화                                   │
│     └── 더 명확한 비대칭 학습 → AP Gap 추가 감소                         │
│                                                                        │
│  3. Hard Negative Mining                                               │
│     └── 극단적 편향 케이스 우선 해결                                     │
│     └── 전체적인 공정성 향상                                            │
│                                                                        │
│  4. Adaptive Weighting                                                 │
│     └── Detection 성능 자동 보호                                        │
│     └── Fairness/Detection 균형 자동화                                  │
│                                                                        │
│  5. Discriminator 제거                                                  │
│     └── Contrastive 1st 성공 요소 계승                                  │
│     └── 안정적 학습 + 더 큰 배치                                        │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### 7.3 실험 실행

```bash
# 기본 실행
python -m faap_gan.train_faap_wgan_GD_13th

# 커스텀 설정
python -m faap_gan.train_faap_wgan_GD_13th \
    --epochs 30 \
    --batch_size 8 \
    --lambda_contrast 1.2 \
    --lambda_asym 0.8 \
    --lambda_hard 0.4
```

---

## 8. 결론

13th 버전은 7th의 **AP Gap 감소 성공 요소**와 Contrastive 1st의 **AR Gap 감소 성공 요소**를 융합하면서, 세 가지 핵심 혁신을 추가했습니다:

1. **Multi-Scale Contrastive**: 더 세밀한 특징 공간 정렬
2. **Asymmetric Alignment Loss**: 7th의 비대칭 개념을 Loss로 공식화
3. **Hard Negative Mining**: 어려운 케이스 집중 학습

Discriminator를 제거하여 학습 안정성을 확보하고, Adaptive Weighting으로 Detection/Fairness 균형을 자동화했습니다.

---

*이 문서는 train_faap_wgan_GD_13th.py의 설계 문서입니다.*
