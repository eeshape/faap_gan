# Score-Based Contrastive Learning for Fair Adversarial Perturbation

**FAAP Training - 2nd Version Technical Report**

---

## 목차

1. [연구 개요](#1-연구-개요)
2. [연구 배경 및 동기](#2-연구-배경-및-동기)
3. [이론적 배경](#3-이론적-배경)
4. [제안 방법론](#4-제안-방법론)
5. [구현 세부사항](#5-구현-세부사항)
6. [손실 함수 분석](#6-손실-함수-분석)
7. [하이퍼파라미터 설계](#7-하이퍼파라미터-설계)
8. [학습 파이프라인](#8-학습-파이프라인)
9. [기대 효과 및 가설](#9-기대-효과-및-가설)
10. [실험 설계](#10-실험-설계)

---

## 1. 연구 개요

### 1.1 핵심 기여

본 연구는 **Detection Score 기반 대조학습(Score-Based Contrastive Learning)**을 통해 Object Detection 모델의 성별 간 공정성(Fairness)을 개선하는 새로운 방법론을 제안한다.

**핵심 아이디어:**
- 기존 방식: 성별(Gender)을 기준으로 고성능/저성능 그룹을 암묵적으로 가정
- 제안 방식: **실제 Detection Score**를 기준으로 동적으로 고성능/저성능 그룹을 분리

### 1.2 1st 버전 대비 개선점

| 구분 | 1st Version | 2nd Version (본 연구) |
|------|-------------|----------------------|
| 분리 기준 | 성별 (Female/Male) | Detection Score (Low/High) |
| 가정 | 여성=저성능, 남성=고성능 (암묵적) | Score 기반 동적 분리 (명시적) |
| Score 활용 | Wasserstein에만 사용 | Contrastive Learning 핵심 |
| 학습 목표 | 성별 feature 혼합 | 저성능 feature → 고성능 방향 이동 |

### 1.3 Wasserstein Distance와의 차별점

```
┌─────────────────────────────────────────────────────────────────┐
│                    Wasserstein Distance                         │
│  - 작동 레벨: Score 분포 (output level)                          │
│  - 변화 대상: Detection score 값의 분포 정렬                      │
│  - 한계: Feature representation 자체는 변하지 않음                │
└─────────────────────────────────────────────────────────────────┘
                              vs
┌─────────────────────────────────────────────────────────────────┐
│              Score-Based Contrastive Learning                    │
│  - 작동 레벨: Feature representation (embedding level)          │
│  - 변화 대상: 저성능 이미지의 feature 자체가 이동                  │
│  - 장점: 근본적인 representation 변화로 일반화 가능                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 연구 배경 및 동기

### 2.1 문제 정의

Object Detection 모델(DETR)에서 관찰되는 성별 기반 성능 격차:

```
Baseline DETR Performance:
- Male AP:   0.45
- Female AP: 0.38
- AP Gap:    0.07 (남성 우위)
```

이 격차는 학습 데이터의 불균형, 외모 특성의 차이, 모델의 암묵적 편향 등에서 기인한다.

### 2.2 기존 접근법의 한계

**Adversarial Approach (7th version):**
```python
# GenderDiscriminator로 성별 구분 후 adversarial 학습
d_loss = cross_entropy(discriminator(features), gender_labels)
fairness_loss = -(ce + α * entropy)
```

**한계점:**
1. 성별 레이블에 의존 → 레이블 노이즈에 취약
2. Discriminator 학습 불안정 → GAN mode collapse 위험
3. AP Gap 미개선 (AR Gap만 60% 개선)

**1st InfoNCE (성별 기반):**
```python
# 성별로만 분리
loss = infonce(proj_female, proj_male)  # Female ↔ Male
```

**한계점:**
1. "여성=저성능" 가정이 항상 성립하지 않음
2. 실제 Detection Score를 활용하지 않음
3. 남성 중 저성능 샘플, 여성 중 고성능 샘플 무시

### 2.3 핵심 통찰

> **Detection Score 자체가 성능의 직접적 지표이다.**
>
> 성별이 아닌 실제 score를 기준으로 분리하면:
> - 저성능 이미지 = 자연스럽게 여성 이미지 다수 포함
> - 고성능 이미지 = 자연스럽게 남성 이미지 다수 포함
> - 그러나 성별과 무관한 순수 성능 기반 학습 가능

---

## 3. 이론적 배경

### 3.1 InfoNCE Loss

InfoNCE (Noise Contrastive Estimation)는 representation learning의 핵심 손실 함수다:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_{k=1}^{N} \exp(\text{sim}(z_i, z_k) / \tau)}$$

**구성 요소:**
- $z_i$: Anchor sample의 projection
- $z_j^+$: Positive sample의 projection
- $z_k$: 모든 샘플 (positive + negatives)
- $\tau$: Temperature (default: 0.07)
- $\text{sim}(\cdot, \cdot)$: Cosine similarity

### 3.2 SimCLR Framework

본 연구는 SimCLR의 구조를 차용하되, **positive/negative 정의를 재설계**한다:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Encoder    │ -> │  Projection  │ -> │  Contrastive │
│   (DETR)     │    │    Head      │    │     Loss     │
│   frozen     │    │   trainable  │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

**SimCLR 원본:**
- Positive: 같은 이미지의 다른 augmentation view
- Negative: 다른 이미지

**본 연구 (Score-Based):**
- Positive: 고성능 이미지 (Detection Score 상위)
- Negative: 다른 저성능 이미지 (Detection Score 하위)

### 3.3 Score 기반 분리의 수학적 정당성

배치 내 $N$개 이미지에 대해 Detection Score $s_i$를 계산:

$$s_i = \frac{1}{K} \sum_{k=1}^{K} \max_{c} P(c | q_k^i)$$

여기서:
- $K$: Top-k queries (default: 10)
- $q_k^i$: 이미지 $i$의 $k$번째 query
- $P(c | q)$: Query $q$가 class $c$일 확률 (softmax output)

**Median 기준 분리:**
$$\text{Low} = \{i : s_i \leq \text{median}(s)\}$$
$$\text{High} = \{i : s_i > \text{median}(s)\}$$

---

## 4. 제안 방법론

### 4.1 전체 아키텍처

```
Input Image
     │
     ▼
┌─────────────────────┐
│ PerturbationGenerator│  ← trainable
│ (U-Net style)        │
└─────────────────────┘
     │ δ (perturbation)
     ▼
┌─────────────────────┐
│ x' = clamp(x + δ)   │
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ SimCLR Augmentation │  ← ColorJitter (optional)
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│    Frozen DETR      │  ← frozen
│  (backbone + decoder)│
└─────────────────────┘
     │
     ├─── outputs (pred_logits, pred_boxes)
     │         │
     │         ▼
     │    ┌────────────────┐
     │    │ Detection Score│ → Score-based 분리 기준
     │    │   Calculation  │
     │    └────────────────┘
     │
     └─── features (decoder output)
               │
               ▼
         ┌───────────────┐
         │ Projection    │  ← trainable
         │ Head (MLP)    │
         └───────────────┘
               │
               ▼
         ┌───────────────┐
         │ Score-Based   │
         │ Contrastive   │
         │ Loss          │
         └───────────────┘
```

### 4.2 Score-Based Contrastive Loss

**핵심 수식:**

$$\mathcal{L}_{\text{low} \to \text{high}} = -\frac{1}{N_{\text{low}}} \sum_{i \in \text{Low}} \log \frac{\sum_{j \in \text{High}} \exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{j \in \text{High}} \exp(\text{sim}(z_i, z_j) / \tau) + \sum_{k \in \text{Low}, k \neq i} \exp(\text{sim}(z_i, z_k) / \tau)}$$

**해석:**
- **Numerator**: 저성능 anchor와 모든 고성능 샘플 간의 similarity
- **Denominator**: 저성능 anchor와 모든 샘플 (고성능 + 다른 저성능) 간의 similarity
- **목표**: 저성능 feature가 고성능 feature 방향으로 이동

### 4.3 양방향 학습

단방향 학습의 한계를 극복하기 위해 **양방향 InfoNCE**를 적용:

```python
# (a) Low → High (강하게: 1.5)
loss_low2high = InfoNCE(anchor=proj_low, positive=proj_high, negative=proj_low)

# (b) High → Low (약하게: 0.5)
loss_high2low = InfoNCE(anchor=proj_high, positive=proj_low, negative=proj_high)

# 비대칭 결합
loss = 1.5 * loss_low2high + 0.5 * loss_high2low
```

**비대칭 가중치 설계 근거:**
- **Low→High (1.5)**: 주요 목표. 저성능 feature를 고성능 방향으로 적극 이동
- **High→Low (0.5)**: 보조 목표. Projection head가 고성능 샘플에서도 학습되도록 균형 유지

### 4.4 Multi-Loss 통합

최종 손실 함수:

$$\mathcal{L}_{\text{total}} = \lambda_c \cdot \mathcal{L}_{\text{contrastive}} + \lambda_w \cdot \mathcal{L}_{\text{wasserstein}} + \beta \cdot \mathcal{L}_{\text{detection}}$$

| Loss | Weight | 역할 |
|------|--------|------|
| $\mathcal{L}_{\text{contrastive}}$ | $\lambda_c = 1.0$ | Feature-level 공정성 (핵심) |
| $\mathcal{L}_{\text{wasserstein}}$ | $\lambda_w = 0.2$ | Score-level 분포 정렬 (보조) |
| $\mathcal{L}_{\text{detection}}$ | $\beta = 0.5 \to 0.6$ | Detection 성능 유지 |

---

## 5. 구현 세부사항

### 5.1 Detection Score 계산

```python
def _image_level_detection_score(outputs: dict, top_k: int = 10) -> torch.Tensor:
    """
    DETR logits에서 직접 이미지 단위 score 계산.

    Args:
        outputs: DETR output (pred_logits: [B, 100, num_classes+1])
        top_k: 상위 k개 query만 사용

    Returns:
        (B,) 각 이미지의 detection score
    """
    # Softmax 후 no-object class 제외
    probs = outputs["pred_logits"].softmax(dim=-1)[..., :-1]  # (B, 100, num_classes)

    # 각 query의 max class probability
    max_probs = probs.max(dim=-1).values  # (B, 100)

    # Top-k query의 평균
    topk_probs = max_probs.topk(top_k, dim=1).values  # (B, top_k)
    return topk_probs.mean(dim=1)  # (B,)
```

**설계 근거:**
- **Top-k 사용 이유**: 100개 query 중 대부분은 no-object. 상위 k개만 의미있는 detection
- **GT Matching 제거**: Hungarian matching 없이 순수 모델 출력만 사용 → 단순화 및 효율성

### 5.2 Projection Head

```python
class SimCLRProjectionHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x: (B, 100, 256) DETR decoder features
        pooled = x.mean(dim=1)  # (B, 256) image-level pooling
        proj = self.net(pooled)  # (B, 128)
        return F.normalize(proj, dim=-1, p=2)  # L2 normalize
```

**구조적 특징:**
- **2-layer MLP**: SimCLR 표준 구조
- **Query Pooling**: 100개 query를 평균하여 image-level representation
- **L2 Normalization**: Cosine similarity 계산에 최적화

### 5.3 SimCLR Augmentation

```python
class SimCLRAugmentation(nn.Module):
    """Detection 친화적 augmentation"""

    STRENGTHS = {
        "none": None,
        "weak": T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        "medium": T.ColorJitter(0.3, 0.3, 0.3, 0.1),  # 권장
        "strong": T.Compose([
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
        ]),
    }
```

**Detection 친화적 설계:**
- **Geometric Transform 제외**: Crop, Flip 등은 bounding box 변형 → Detection에 부적합
- **Color Transform만 사용**: ColorJitter로 조명/색상 불변성 학습
- **Grayscale 제한적 사용**: Strong 모드에서만 (detection 성능 저하 위험)

### 5.4 안전장치

```python
# 1. 최소 샘플 수 체크
if N < 4:
    return loss=0.0  # 최소 4개 필요 (low 2개 + high 2개)

# 2. 각 그룹 최소 2개 체크
if n_low < 2 or n_high < 2:
    return loss=0.0  # Negative 샘플 필요

# 3. Score detach
scores = scores.detach()  # 분리 기준에 gradient 불필요

# 4. 자기 자신 마스킹
mask_self = torch.eye(n_low, device=device, dtype=torch.bool)
sim_low2low = sim_low2low.masked_fill(mask_self, float('-inf'))
```

---

## 6. 손실 함수 분석

### 6.1 Score-Based Contrastive Loss 상세

**Forward Pass:**

```
Input: projections (N, D), scores (N,)
       │
       ▼
┌─────────────────────────┐
│ 1. Score-based 분리     │
│    median = scores.median()
│    low_mask = scores <= median
│    high_mask = scores > median
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ 2. Similarity 계산      │
│    sim_low2high: (N_low, N_high)
│    sim_low2low:  (N_low, N_low)
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ 3. InfoNCE (Low→High)   │
│    numerator = logsumexp(sim_low2high)
│    denominator = logsumexp([sim_low2high, sim_low2low])
│    loss_l2h = -(numerator - denominator).mean()
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ 4. InfoNCE (High→Low)   │
│    (동일 로직, 방향 반대)
└─────────────────────────┘
       │
       ▼
┌─────────────────────────┐
│ 5. 비대칭 결합          │
│    loss = 1.5 * loss_l2h + 0.5 * loss_h2l
└─────────────────────────┘
```

### 6.2 Wasserstein Loss (보조)

```python
def _wasserstein_1d_asymmetric(female_scores, male_scores):
    """단방향 Wasserstein: Female → Male"""
    sorted_f = female_scores.sort().values
    sorted_m = male_scores.detach().sort().values  # Male은 target

    # 분위수 정렬 후 단방향 거리
    return F.relu(sorted_m - sorted_f).mean()
```

**역할:**
- Score 분포 레벨에서 추가 정렬
- Contrastive loss와 상호 보완
- 여성 score < 남성 score일 때만 패널티

### 6.3 Detection Loss (정규화)

```python
loss_det, _ = detr.detection_loss(outputs, targets)
```

**역할:**
- Perturbation이 detection 성능을 유지하도록 제약
- β scheduling: 0.5 → 0.6 (점진적 강화)

---

## 7. 하이퍼파라미터 설계

### 7.1 핵심 하이퍼파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| `temperature` | 0.07 | [0.05, 0.1] | InfoNCE temperature (SimCLR 표준) |
| `score_top_k` | 10 | [5, 20] | Score 계산 시 top-k queries |
| `score_margin` | 0.0 | [0.0, 0.1] | High/Low 분리 마진 |
| `proj_dim` | 128 | [64, 256] | Projection 출력 차원 |
| `lambda_contrastive` | 1.0 | [0.5, 2.0] | Contrastive loss 가중치 |
| `lambda_wass` | 0.2 | [0.1, 0.5] | Wasserstein loss 가중치 |

### 7.2 Epsilon Scheduling

```
Epoch:  0  1  2  3  4  5  6  7  8  9  10 11 12 13 ... 23
        │─────── Warmup ───────│─ Hold ─│── Cooldown ──│
Epsilon:0.05 ────────────> 0.10 ─────> 0.10 ────> 0.09
```

**설계 근거:**
- **Warmup (0→7)**: Perturbation 강도를 점진적으로 증가
- **Hold (8→13)**: 최대 강도 유지하여 충분히 학습
- **Cooldown (14→23)**: 미세 조정으로 안정화

### 7.3 Beta Scheduling

```
β = 0.5 + (0.6 - 0.5) × (epoch / (epochs - 1))
```

- 초기: Detection loss 약하게 → Fairness 학습에 집중
- 후기: Detection loss 강하게 → 성능 유지 강화

---

## 8. 학습 파이프라인

### 8.1 전체 흐름

```python
for epoch in range(epochs):
    for samples, targets, genders in dataloader:
        # 1. Forward
        perturbed = generator(samples)
        perturbed = simclr_aug(perturbed)  # optional
        outputs, features = detr(perturbed)

        # 2. Score 계산 (DETR logits 직접)
        image_scores = _image_level_detection_score(outputs, top_k=10)

        # 3. Projection
        projections = proj_head(features)

        # 4. Losses
        loss_contrastive = contrastive_loss_fn(projections, image_scores)
        loss_wasserstein = wasserstein_loss(female_scores, male_scores)
        loss_det = detr.detection_loss(outputs, targets)

        # 5. Total
        total = λ_c * loss_contrastive + λ_w * loss_wasserstein + β * loss_det

        # 6. Backward
        total.backward()
        optimizer.step()
```

### 8.2 학습 가능 파라미터

| 모듈 | 학습 가능 | 파라미터 수 (추정) |
|------|----------|-------------------|
| DETR (backbone + decoder) | ❌ frozen | - |
| PerturbationGenerator | ✅ | ~2M |
| SimCLRProjectionHead | ✅ | ~50K |

### 8.3 메모리 및 연산 최적화

```python
# 1. Gradient 차단 (score 계산)
scores = scores.detach()

# 2. No-grad 영역 (metrics)
with torch.no_grad():
    delta_linf = ...
    obj_mean = ...

# 3. Tensor 타입 필터링 (outputs indexing)
female_outputs = {k: v[idx] for k, v in outputs.items() if isinstance(v, torch.Tensor)}
```

---

## 9. 기대 효과 및 가설

### 9.1 핵심 가설

**H1: Feature-level 학습이 Score-level 학습보다 효과적이다**
- Wasserstein: Score 분포만 정렬 → 표면적 개선
- Contrastive: Feature 자체 이동 → 근본적 개선

**H2: Score 기반 분리가 성별 기반 분리보다 정확하다**
- 성별 기반: "모든 여성 = 저성능" 가정 (부정확)
- Score 기반: 실제 성능으로 분리 (정확)

**H3: 양방향 학습이 단방향보다 안정적이다**
- 단방향: proj_high가 학습되지 않음 → 불균형
- 양방향: 모든 샘플이 학습에 참여 → 균형

### 9.2 예상 결과

| 메트릭 | Baseline | 7th (Adv) | 1st (InfoNCE) | 2nd (Score-Based) |
|--------|----------|-----------|---------------|-------------------|
| Male AP | 0.45 | 0.44 | 0.44 | 0.44 |
| Female AP | 0.38 | 0.39 | 0.40 | **0.42+** |
| AP Gap | 0.07 | 0.05 | 0.04 | **<0.03** |
| AR Gap | 0.10 | 0.04 | 0.05 | **<0.04** |

### 9.3 성공 기준

1. **AP Gap < 0.09** (15% 이상 개선)
2. **Female AP > 0.41** (baseline 0.38 대비 +0.03)
3. **Male AP 유지** (0.44 이상)

---

## 10. 실험 설계

### 10.1 실행 명령어

```bash
# 기본 실행
python train_faap_simclr_infonce_2nd.py --batch_size 8

# 전체 옵션
python train_faap_simclr_infonce_2nd.py \
    --batch_size 8 \
    --epochs 24 \
    --temperature 0.07 \
    --score_top_k 10 \
    --lambda_contrastive 1.0 \
    --lambda_wass 0.2 \
    --aug_strength medium
```

### 10.2 Ablation Study 설계

| 실험 | 설명 | 명령어 |
|------|------|--------|
| A1 | Contrastive만 (Wass 제거) | `--lambda_wass 0.0` |
| A2 | Temperature 변화 | `--temperature 0.05/0.1` |
| A3 | Top-k 변화 | `--score_top_k 5/20` |
| A4 | Augmentation 제거 | `--aug_strength none` |
| A5 | 단방향만 (High→Low 제거) | 코드 수정 필요 |

### 10.3 모니터링 메트릭

```
[Epoch X] Summary:
  Contrastive Loss: X.XXXX (핵심)
  Wasserstein Loss: X.XXXX
  Detection Loss: X.XXXX
  Total: X.XXXX
  Score-based split: low=X.X, high=X.X
  Score gap (high-low): X.XXXX
  Obj Score (F/M): X.XXXX / X.XXXX
  Obj Score Gap (M-F): X.XXXX
```

**주요 관찰 포인트:**
1. `Contrastive Loss`: 감소 추세 확인
2. `Score gap`: 학습 진행에 따른 변화
3. `Obj Score Gap (M-F)`: 성별 간 격차 감소 확인

### 10.4 출력 파일

```
faap_outputs/faap_outputs_infonce_2nd/
├── config.json              # 실험 설정
├── dataset_layout.json      # 데이터셋 구조
├── train_log.jsonl          # 학습 로그 (epoch별)
└── checkpoints/
    ├── epoch_0000.pth
    ├── epoch_0001.pth
    └── ...
```

---

## 부록: 코드 구조

```
train_faap_simclr_infonce_2nd.py
├── SimCLRAugmentation (65-126)
│   └── Detection 친화적 ColorJitter
├── SimCLRProjectionHead (133-164)
│   └── 2-layer MLP + L2 normalize
├── ScoreBasedContrastiveLoss (171-286)
│   ├── Score 기반 분리
│   └── 양방향 InfoNCE
├── _image_level_detection_score (356-382)
│   └── Top-k query confidence 평균
├── _wasserstein_1d_asymmetric (308-329)
│   └── 단방향 Wasserstein
├── Scheduling Functions (501-533)
│   ├── _scheduled_epsilon
│   └── _scheduled_beta
└── main() (540-906)
    └── Training loop
```

---

**Author:** FAAP Research Team
**Version:** 2nd (Score-Based Contrastive Learning)
**Last Updated:** 2025-01-24
