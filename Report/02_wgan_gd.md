# 2. WGAN-GD 계열 (Wasserstein GAN with Gender-Dual Perturbation)

**파일**: `train_faap_wgan.py`, `train_faap_wgan_GD.py` ~ `train_faap_wgan_GD_14th.py`
**기간**: 2025-11-24 ~ 2026-01-20
**총 버전**: 17개 (wgan, GD base, 2nd~14th)
**핵심 성과**: 7th — 국내학회 제출 최종 버전

---

## 2.1 파이프라인 (7th 기준 — 최종 논문 버전)

```
┌───────────────────────────────────────────────────────────────┐
│               WGAN-GD 7th Pipeline (논문 제출 버전)              │
│                                                               │
│  ┌──────────┐        ┌──────────┐                             │
│  │ Female   │        │  Male    │                             │
│  │ Image    │        │  Image   │                             │
│  └────┬─────┘        └────┬─────┘                             │
│       │                   │                                   │
│       ▼                   ▼                                   │
│  Generator(G)        Generator(G)    ← 양성 교란 (GD 이후)     │
│       │                   │                                   │
│  Perturbed_f         Perturbed_m                              │
│       │                   │                                   │
│       ▼                   ▼                                   │
│  ┌──────────────────────────────┐                             │
│  │         Frozen DETR          │                             │
│  └─────┬──────────┬─────┬──────┘                              │
│        │          │     │                                     │
│   Detection    Features  Matched                              │
│   Outputs    (B×100×256) Scores                               │
│        │          │      │                                    │
│   L_det(f,m)  D(feat)   Wasserstein                           │
│        │       │   │     │                                    │
│        │    CE  Entropy  L_w = ReLU(sorted_m - sorted_f)      │
│        │       └─┬─┘                                          │
│        │    L_fairness                                        │
│        │         │                                            │
│        └────┬────┘                                            │
│             ▼                                                 │
│  L_total = λ_fair·L_fairness + β(t)·L_det + λ_w·L_w         │
└───────────────────────────────────────────────────────────────┘
```

---

## 2.2 Loss 수식 진화

### Phase 1: WGAN (2025-11-24)

$$L_G = L_{fairness} + \beta \cdot L_{det} + \lambda_w \cdot W(s_f, s_m)$$

여기서 Wasserstein loss:
$$W(s_f, s_m) = \frac{1}{K}\sum_{k=1}^{K} |sorted\_f_k - sorted\_m_k|$$
- 양방향: 남녀 모두 이동

### Phase 1→2: GD 전환 (2025-11-28)
- 남성도 G 적용 (양성 교란)
- D도 양쪽 perturbed image로 학습

### Phase 2: 3rd → 4th 전환점 (단방향 Wasserstein)

**3rd (실패 — 양방향)**:
$$W_{bi}(s_f, s_m) = \frac{1}{K}\sum_{k=1}^{K} |sorted\_f_k - sorted\_m_k|$$

**4th (성공 — 단방향)**:
$$W_{uni}(s_f, s_m) = \frac{1}{K}\sum_{k=1}^{K} \text{ReLU}(sorted\_m_k - sorted\_f_k)$$

핵심: `male_scores.detach()` — 남성 score를 타겟으로 고정, 여성만 끌어올림

### 7th (최종 논문 버전) 전체 Loss

**Generator Total Loss**:
$$L_G = \lambda_{fair} \cdot L_{fairness} + \beta(t) \cdot L_{det} + \lambda_w \cdot L_w$$

**Fairness Loss** (비대칭 가중치):
$$L_{fairness} = w_f \cdot L_{fair,f} + w_m \cdot L_{fair,m}$$

$$L_{fair,f} = -(CE(\hat{y}_f, 1) + \alpha \cdot H(\hat{y}_f))$$
$$L_{fair,m} = -(CE(\hat{y}_m, 0) + \alpha \cdot H(\hat{y}_m))$$

- $w_f = 1.0$ (여성 가중치)
- $w_m = 0.5$ (남성 가중치 — 억제)

**Detection Loss**:
$$L_{det} = L_{det,f} + L_{det,m}$$
$$L_{det,g} = \sum_{k} w_k \cdot l_k^{(g)} \quad (k \in \{ce, bbox, giou\})$$

**Wasserstein Loss** (단방향):
$$L_w = \frac{1}{K}\sum_{k=1}^{K} \text{ReLU}(\hat{s}_{m,k} - s_{f,k})$$
- $\hat{s}_m$: detach된 남성 score (타겟)
- $s_f$: 여성 score (학습 대상)
- 선형 보간으로 크기 맞춤

**Discriminator Loss**:
$$L_D = \frac{1}{2}[CE(D(feat_f), 1) + CE(D(feat_m), 0)]$$

---

## 2.3 하이퍼파라미터 스케줄링 (7th)

### Epsilon: 3-Phase Schedule (Warmup → Hold → Cooldown)
```
    ε
0.10 ┤          ┌──────────┐
     │         /│          │╲
0.09 ┤        / │          │ ╲────────
     │       /  │          │
0.05 ┤──────/   │          │
     └──────┴───┴──────────┴─────────→ epoch
     0      8   14         24
       Warmup  Hold    Cooldown
```

### Beta: 선형 증가
```
    β
0.6 ┤                           ╱
    │                        ╱
0.5 ┤──────────────────────╱
    └──────────────────────────→ epoch
    0                        23
```

---

## 2.4 버전별 상세 비교

### 초기 (wgan ~ GD 2nd): Loss 구조 확립

| 버전 | epochs | batch | k_d | beta | lambda_w | 핵심 변경 |
|------|--------|-------|-----|------|----------|-----------|
| wgan | 6 | 8 | 2 | 0.7 | 0.05 | +Wasserstein 도입 |
| GD base | 12 | 4 | 2 | 0.7 | 0.05 | +남녀 모두 G 적용 |
| GD 2nd | 12 | 4 | 2 | 0.3 | 0.1 | lambda_fair=5.0 강화 |

### 핵심 진화 (3rd ~ 8th): 최적 구성 탐색

| 버전 | Wasserstein 방향 | fair_m_scale | Epsilon Schedule | Beta Schedule | 결과 |
|------|:---:|:---:|---|---|:---:|
| **3rd** | 양방향 | 1.0 | Warmup only | 고정 0.5 | **실패** |
| **4th** | **단방향** | 1.0 | Warmup only | 고정 0.5 | 전환점 |
| 5th | 단방향 | 1.0 | Warmup only | 고정 0.5 | 보통 |
| 6th | 단방향 | 1.0 | 3-phase | 0.5→0.7 | 보통 |
| 6th_2 | 단방향 | 1.0 | Stress window | 가변 | **실패** |
| **7th** | **단방향** | **0.5** | **3-phase** | **0.5→0.6** | **최고** |
| 8th | 단방향 | 0.5 | 3-phase | 0.5→0.6 | 7th 미세변형 |

### 후기 (9th ~ 14th): AP Gap 해결 시도

| 버전 | epochs | 신규 Loss/모듈 | Discriminator | 핵심 아이디어 |
|------|--------|----------------|:---:|---------------|
| 9th | 28 | quantile_matching, score_gap_penalty | O | Quantile 정렬 |
| 10th | 28 | gap_scale, detection_guard, score_uplift | O | Adaptive Alignment |
| **11th** | 30 | ProjectionHead, cross_gender_contrastive | **X** | ★ Contrastive 도입 |
| 12th | 30 | focal_score_alignment | **X** | D 제거 확정 |
| 13th | 30 | 3-branch ProjectionHead, hard_negative_mining | X | Multi-Scale |
| 14th | 30 | quantile_weighted_wasserstein, confidence_margin | X | AP 직접 최적화 |

---

## 2.5 7th 실험 결과 (국내학회 제출 버전)

**Test Set, Epoch 23, ε=0.09**

| 지표 | 성별 | Baseline | Perturbed | Delta |
|------|------|----------|-----------|-------|
| **AP** | Male | 0.5108 | 0.5137 | **+0.0029** |
| | Female | 0.4045 | 0.4078 | **+0.0034** |
| **AR** | Male | 0.8339 | 0.8359 | **+0.0021** |
| | Female | 0.8258 | 0.8328 | **+0.0070** |

| Gap 지표 | Baseline | Perturbed | 개선율 |
|----------|----------|-----------|--------|
| **AP Gap** (M-F) | 0.1063 | 0.1059 | -0.4% |
| **AR Gap** (M-F) | 0.0081 | **0.0032** | **-60.5%** |

### 7th 성공 요인 분석

1. **단방향 Wasserstein** (4th에서 도입): Female만 끌어올리고 Male 고정
2. **비대칭 fairness** (7th 신규): `fair_m_scale=0.5`로 남성 perturbation 억제
3. **3-phase Epsilon**: Warmup→Hold→Cooldown으로 안정적 수렴
4. **충분한 학습**: 24 epoch (이전 12 대비 2배)
5. **완만한 Beta 증가**: 0.5→0.6으로 점진적 detection 보존 강화

---

## 2.6 7th 핵심 하이퍼파라미터

| 카테고리 | 파라미터 | 값 | 설명 |
|---------|---------|-----|------|
| **학습** | `epochs` | 24 | 총 에폭 |
| | `batch_size` | 4 | 배치 크기 |
| | `lr_g` | 1e-4 | Generator LR |
| | `lr_d` | 1e-4 | Discriminator LR |
| | `k_d` | 4 | D 업데이트 횟수 |
| **Epsilon** | `epsilon` | 0.05 | 시작값 |
| | `epsilon_final` | 0.10 | 최대값 |
| | `epsilon_min` | 0.09 | 쿨다운 후 최소값 |
| | `warmup_epochs` | 8 | Warmup 기간 |
| | `hold_epochs` | 6 | Hold 기간 |
| | `cooldown_epochs` | 10 | Cooldown 기간 |
| **Loss 가중치** | `lambda_fair` | 2.0 | Fairness 전체 가중치 |
| | `lambda_w` | 0.2 | Wasserstein 가중치 |
| | `beta` | 0.5→0.6 | Detection (선형 증가) |
| | `alpha` | 0.2 | Entropy 가중치 |
| **비대칭** | `fair_f_scale` | 1.0 | 여성 fairness 가중치 |
| | `fair_m_scale` | 0.5 | 남성 fairness 가중치 |
| **기타** | `max_norm` | 0.1 | Gradient clipping |

---

## 2.7 Phase 3 전환 (11th~14th): GAN-Free로의 진화

11th에서 Contrastive loss 도입 → 12th에서 Discriminator 완전 제거.
이후 WGAN-GD 계열 내에서도 GAN-free 구조가 탐색됨.

```
WGAN-GD 진화 경로:

GAN 기반:        wgan → GD → 2nd → 3rd → 4th → ... → 7th(최고) → 8th → 9th → 10th
                                                                            │
GAN-Free 전환:                                                    11th ─→ 12th → 13th → 14th
```

| 변화 | GAN 기반 (7th) | GAN-Free (12th) |
|------|:---:|:---:|
| Discriminator | O | X |
| Fairness Loss | Adversarial CE | Contrastive InfoNCE |
| Score 정렬 | Wasserstein | Wasserstein + Focal |
| 학습 안정성 | D/G 균형 필요 | 안정적 |
| 결과 | AR Gap -60% | 7th 미달 |
