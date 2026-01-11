# FAAP Contrastive Fairness Training: 기술 분석 보고서

## 목차
1. [개요](#1-개요)
2. [문제 정의 및 동기](#2-문제-정의-및-동기)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [핵심 모듈 상세 분석](#4-핵심-모듈-상세-분석)
5. [손실 함수 설계](#5-손실-함수-설계)
6. [학습 파이프라인](#6-학습-파이프라인)
7. [수학적 기반](#7-수학적-기반)
8. [하이퍼파라미터 분석](#8-하이퍼파라미터-분석)
9. [설계 결정 및 트레이드오프](#9-설계-결정-및-트레이드오프)
10. [결론 및 확장 가능성](#10-결론-및-확장-가능성)

---

## 1. 개요

### 1.1 프로젝트 목적

본 시스템은 **FAAP (Fair Adversarial Attack for Person detection)** 프레임워크의 10번째 버전으로, 객체 탐지 모델(DETR)에서 발생하는 **성별 기반 편향(Gender Bias)**을 완화하기 위한 학습 시스템입니다.

### 1.2 핵심 혁신점

| 버전 | 접근 방식 | 문제점 |
|------|-----------|--------|
| 이전 버전 (WGAN) | Discriminator를 통한 적대적 학습 | 모드 붕괴, 불안정한 학습 |
| **10th 버전** | **Contrastive Learning 기반 Fairness** | 안정적 학습, GAN 없음 |

### 1.3 철학적 기반

전통적인 Contrastive Learning과 본 시스템의 Fairness Contrastive의 차이:

```
┌─────────────────────────────────────────────────────────────────┐
│           전통적 Contrastive Learning                            │
│   같은 클래스 = Positive (가깝게)                                 │
│   다른 클래스 = Negative (멀리)                                   │
├─────────────────────────────────────────────────────────────────┤
│           Fairness Contrastive (본 시스템)                       │
│   다른 성별 = Positive (가깝게) ← 핵심 역전!                      │
│   같은 성별 = Neutral                                            │
│   → 결과: 특징 공간에서 성별 정보 제거                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 문제 정의 및 동기

### 2.1 편향 문제의 정의

객체 탐지 모델은 학습 데이터의 불균형으로 인해 특정 인구 집단에 대해 차별적인 성능을 보일 수 있습니다:

- **문제**: 여성 이미지에서의 탐지 정확도가 남성 이미지보다 낮음
- **원인**: 학습 데이터에서 성별 관련 시각적 특징과 탐지 품질 간의 spurious correlation
- **목표**: 성별에 관계없이 동등한 탐지 성능 달성

### 2.2 WGAN 방식의 한계

이전 버전에서 사용한 WGAN (Wasserstein GAN) 방식의 문제점:

1. **모드 붕괴 (Mode Collapse)**: Generator가 다양한 perturbation을 생성하지 못함
2. **학습 불안정성**: Discriminator와 Generator 간의 균형 유지 어려움
3. **하이퍼파라미터 민감성**: critic_iter, gradient penalty 계수 등 조정 복잡

### 2.3 Contrastive 접근의 동기

Contrastive Learning은 다음과 같은 이점을 제공합니다:

- **안정성**: 적대적 학습 없이 representation 학습 가능
- **이론적 기반**: InfoNCE loss의 mutual information 최대화 특성
- **유연성**: 다양한 fairness constraint 통합 용이

---

## 3. 시스템 아키텍처

### 3.1 전체 시스템 구조

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FAAP Contrastive System                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────┐      ┌───────────────────┐      ┌─────────────────┐  │
│   │   Input      │      │   Perturbation    │      │   Perturbed     │  │
│   │   Image      │ ──▶  │   Generator (G)   │ ──▶  │   Image         │  │
│   │   x          │      │   δ = G(x)        │      │   x + δ         │  │
│   └──────────────┘      └───────────────────┘      └────────┬────────┘  │
│                                                              │           │
│                         ┌────────────────────────────────────▼───────┐  │
│                         │         Frozen DETR                         │  │
│                         │  ┌───────────┐  ┌──────────┐  ┌──────────┐ │  │
│                         │  │ Backbone  │─▶│Transformer│─▶│ Decoder  │ │  │
│                         │  │ (ResNet)  │  │ Encoder   │  │ Features │ │  │
│                         │  └───────────┘  └──────────┘  └────┬─────┘ │  │
│                         └──────────────────────────────────────│──────┘  │
│                                                                 │         │
│   ┌─────────────────────────────────────────────────────────────▼──────┐ │
│   │                    Two Output Branches                              │ │
│   │                                                                     │ │
│   │   ┌─────────────────────┐          ┌─────────────────────────────┐ │ │
│   │   │  Detection Head     │          │  Projection Head (New!)      │ │ │
│   │   │  - pred_logits      │          │  - LayerNorm                 │ │ │
│   │   │  - pred_boxes       │          │  - Linear(256→256)           │ │ │
│   │   │                     │          │  - ReLU                      │ │ │
│   │   │  → Detection Loss   │          │  - Linear(256→128)           │ │ │
│   │   └─────────────────────┘          │  - L2 Normalize              │ │ │
│   │                                     │                             │ │ │
│   │                                     │  → Contrastive Loss         │ │ │
│   │                                     └─────────────────────────────┘ │ │
│   └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 데이터 흐름

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Training Data Flow                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Dataset                                                                    │
│   ┌─────────────────┐    ┌─────────────────┐                                │
│   │  Female Images  │    │   Male Images   │                                │
│   │    (women)      │    │     (men)       │                                │
│   └────────┬────────┘    └────────┬────────┘                                │
│            │                      │                                          │
│            ▼                      ▼                                          │
│   ┌─────────────────────────────────────────┐                               │
│   │     Gender-Balanced Batch Sampler       │                               │
│   │     (WeightedRandomSampler)             │                               │
│   └────────────────────┬────────────────────┘                               │
│                        │                                                     │
│                        ▼                                                     │
│   ┌─────────────────────────────────────────┐                               │
│   │        Batch Split by Gender            │                               │
│   │   female_batch    │    male_batch       │                               │
│   └─────────┬─────────┴──────────┬──────────┘                               │
│             │                    │                                           │
│             ▼                    ▼                                           │
│   ┌─────────────────┐  ┌─────────────────┐                                  │
│   │ G(female_batch) │  │ G(male_batch)   │                                  │
│   │ perturbed_f     │  │ perturbed_m     │                                  │
│   └────────┬────────┘  └────────┬────────┘                                  │
│            │                    │                                            │
│            ▼                    ▼                                            │
│   ┌─────────────────┐  ┌─────────────────┐                                  │
│   │ DETR(pert_f)    │  │ DETR(pert_m)    │                                  │
│   │ → feat_f        │  │ → feat_m        │                                  │
│   │ → outputs_f     │  │ → outputs_m     │                                  │
│   └────────┬────────┘  └────────┬────────┘                                  │
│            │                    │                                            │
│            ▼                    ▼                                            │
│   ┌─────────────────┐  ┌─────────────────┐                                  │
│   │ ProjHead(feat_f)│  │ ProjHead(feat_m)│                                  │
│   │ → proj_f        │  │ → proj_m        │                                  │
│   └────────┬────────┘  └────────┬────────┘                                  │
│            │                    │                                            │
│            └──────────┬─────────┘                                            │
│                       ▼                                                      │
│   ┌─────────────────────────────────────────┐                               │
│   │         Loss Computation                 │                               │
│   │   - Cross-Gender Contrastive Loss       │                               │
│   │   - Feature Alignment Loss              │                               │
│   │   - Score Distribution Loss             │                               │
│   │   - Detection Loss                      │                               │
│   └────────────────────┬────────────────────┘                               │
│                        │                                                     │
│                        ▼                                                     │
│   ┌─────────────────────────────────────────┐                               │
│   │    Backpropagation (G, ProjHead만)      │                               │
│   └─────────────────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 핵심 모듈 상세 분석

### 4.1 PerturbationGenerator

**목적**: 입력 이미지에 대해 bounded perturbation δ를 생성

**아키텍처**: U-Net 스타일의 경량 Encoder-Decoder

```python
class PerturbationGenerator(nn.Module):
    def __init__(self, base_channels: int = 32, epsilon: float = 0.05):
        # Encoder (Downsampling path)
        self.down1 = ConvBlock(3, 32, stride=1)       # H×W → H×W
        self.down2 = ConvBlock(32, 64, stride=2)      # H×W → H/2×W/2
        self.down3 = ConvBlock(64, 128, stride=2)     # H/2×W/2 → H/4×W/4
        
        # Bottleneck
        self.bottleneck = ConvBlock(128, 128, stride=1)
        
        # Decoder (Upsampling path with skip connections)
        self.up2 = UpBlock(128, 64)                   # H/4×W/4 → H/2×W/2
        self.up1 = UpBlock(64, 32)                    # H/2×W/2 → H×W
        
        # Output
        self.out_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
```

**핵심 메커니즘**:

1. **Skip Connections**: Encoder 특징을 Decoder에 직접 연결 (`u2 = up2(h3) + h2`)
2. **Bounded Output**: `tanh` 활성화 후 ε 스케일링으로 perturbation 크기 제한
3. **이미지 정규화 호환**: ImageNet 정규화된 입력에 대해 동작

```python
def forward(self, x):
    # ... encoder/decoder ...
    delta = torch.tanh(self.out_conv(u1))  # [-1, 1] 범위
    return self.epsilon * delta             # [-ε, ε] 범위로 제한
```

**수학적 표현**:

$$\delta = \epsilon \cdot \tanh(G_\theta(x))$$

$$\tilde{x} = \text{clamp}(x + \delta, \text{valid\_range})$$

### 4.2 FrozenDETR

**목적**: 고정된 DETR 모델로 특징 추출 및 탐지 수행

**핵심 특징**:

1. **가중치 동결**: 모든 파라미터가 학습되지 않음 (`requires_grad=False`)
2. **이중 출력**: Detection 결과와 Transformer 특징을 동시 반환

```python
def forward_with_features(self, samples) -> Tuple[dict, torch.Tensor]:
    # Backbone: 이미지 → CNN 특징
    features, pos = self.model.backbone(samples)
    src, mask = features[-1].decompose()
    
    # Transformer: 특징 → Object Queries 처리
    hs = self.model.transformer(
        self.model.input_proj(src),  # (B, 256, H/32, W/32)
        mask,
        self.model.query_embed.weight,  # (100, 256) learned queries
        pos[-1]
    )[0]
    
    # Detection Head
    outputs_class = self.model.class_embed(hs)   # Classification
    outputs_coord = self.model.bbox_embed(hs)    # Bounding Box
    
    # 반환: (detection_outputs, decoder_features)
    return outputs, hs[-1]  # hs[-1]: (B, 100, 256)
```

**DETR 아키텍처 내부 구조**:

```
Input Image (B, 3, H, W)
        │
        ▼
┌───────────────────────────────────────┐
│           ResNet-50 Backbone          │
│   Conv layers → Feature maps          │
│   Output: (B, 2048, H/32, W/32)       │
└───────────────────────────────────────┘
        │
        ▼ input_proj: 2048 → 256
┌───────────────────────────────────────┐
│        Transformer Encoder            │
│   6 layers of self-attention          │
│   + positional encoding               │
│   Output: (H/32×W/32, B, 256)         │
└───────────────────────────────────────┘
        │
        │  + Query Embeddings (100, 256)
        ▼
┌───────────────────────────────────────┐
│        Transformer Decoder            │
│   6 layers of cross-attention         │
│   Output: hs (B, 100, 256)            │ ← 이 특징을 Fairness에 활용
└───────────────────────────────────────┘
        │
        ▼
   ┌────┴────┐
   │         │
   ▼         ▼
┌──────┐  ┌──────┐
│Class │  │ Box  │
│Embed │  │Embed │
│91cls │  │ 4dim │
└──────┘  └──────┘
```

### 4.3 ProjectionHead

**목적**: Contrastive Learning을 위한 특징 투영

**설계 원리** (SimCLR 영감):

```python
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=128):
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),      # 안정적인 학습
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        # x: (B, 100, 256) - DETR decoder features
        pooled = x.mean(dim=1)            # (B, 256) - Query averaging
        projected = self.net(pooled)      # (B, 128)
        return F.normalize(projected, dim=-1)  # Unit hypersphere
```

**Query Pooling 전략**:

DETR의 100개 object query 특징을 평균하여 이미지 수준 representation 생성:

$$z = \frac{1}{N_q} \sum_{i=1}^{N_q} h_i$$

이는 이미지 전체의 "탐지 패턴"을 요약하며, 성별에 따른 탐지 행동 차이를 포착합니다.

### 4.4 GenderCocoDataset

**목적**: 성별 라벨이 포함된 COCO 형식 데이터셋

**디렉토리 구조**:

```
faap_dataset/
├── women_split/
│   ├── train/                          # 이미지 폴더
│   ├── val/
│   ├── test/
│   ├── gender_women_train.json         # COCO 형식 어노테이션
│   ├── gender_women_val.json
│   └── gender_women_test.json
└── men_split/
    ├── train/
    ├── val/
    ├── test/
    ├── gender_men_train.json
    ├── gender_men_val.json
    └── gender_men_test.json
```

**데이터 반환 형식**:

```python
def __getitem__(self, idx):
    image, target = super().__getitem__(idx)
    # target: {"boxes": Tensor, "labels": Tensor, ...}
    if self.include_gender:
        return image, target, self.gender  # "female" or "male"
    return image, target
```

---

## 5. 손실 함수 설계

### 5.1 전체 손실 함수 구조

$$\mathcal{L}_{\text{total}} = \lambda_{\text{contrast}} \cdot \mathcal{L}_{\text{contrast}} + \lambda_{\text{align}} \cdot \mathcal{L}_{\text{align}} + \lambda_{\text{var}} \cdot \mathcal{L}_{\text{var}} + \lambda_{\text{score}} \cdot \mathcal{L}_{\text{score}} + \beta \cdot \mathcal{L}_{\text{det}}$$

| 손실 함수 | 기본 가중치 | 역할 |
|-----------|-------------|------|
| $\mathcal{L}_{\text{contrast}}$ | 1.0 | 성별 간 특징 혼합 |
| $\mathcal{L}_{\text{align}}$ | 0.5 | 평균 특징 정렬 |
| $\mathcal{L}_{\text{var}}$ | 0.1 | 분산 정렬 |
| $\mathcal{L}_{\text{score}}$ | 0.3 | 탐지 점수 균등화 |
| $\mathcal{L}_{\text{det}}$ | 0.6 | 탐지 성능 유지 |

### 5.2 Cross-Gender Contrastive Loss

**핵심 아이디어**: 다른 성별 샘플을 positive pair로 취급

**수학적 정의**:

여성 샘플 $z_f^{(i)}$와 남성 샘플 $z_m^{(j)}$의 유사도:

$$\text{sim}(z_f^{(i)}, z_m^{(j)}) = \frac{z_f^{(i)} \cdot z_m^{(j)}}{\tau}$$

여기서 $\tau$는 temperature 파라미터 (기본값: 0.1)

**손실 계산**:

$$\mathcal{L}_{f \to m} = -\frac{1}{N_f} \sum_{i=1}^{N_f} \log \frac{\sum_{j=1}^{N_m} \exp(\text{sim}(z_f^{(i)}, z_m^{(j)}))}{N_m}$$

$$\mathcal{L}_{m \to f} = -\frac{1}{N_m} \sum_{j=1}^{N_m} \log \frac{\sum_{i=1}^{N_f} \exp(\text{sim}(z_m^{(j)}, z_f^{(i)}))}{N_f}$$

$$\mathcal{L}_{\text{contrast}} = \frac{\mathcal{L}_{f \to m} + \mathcal{L}_{m \to f}}{2}$$

**코드 구현**:

```python
def _cross_gender_contrastive_loss(proj_f, proj_m, temperature=0.1):
    # 유사도 행렬 계산
    sim_f_to_m = torch.mm(proj_f, proj_m.t()) / temperature  # (N_f, N_m)
    
    n_f, n_m = proj_f.size(0), proj_m.size(0)
    
    # 여성→남성: 각 여성이 남성 전체와 유사해지도록
    loss_f_to_m = -torch.logsumexp(sim_f_to_m, dim=1).mean() + log(n_m)
    
    # 남성→여성: 각 남성이 여성 전체와 유사해지도록
    loss_m_to_f = -torch.logsumexp(sim_f_to_m.t(), dim=1).mean() + log(n_f)
    
    return (loss_f_to_m + loss_m_to_f) / 2
```

**기하학적 해석**:

```
훈련 전:                        훈련 후:
    ●●●  Female cluster            ● ○ ●
   ●   ●                          ○ ● ○ ●
    ● ●                           ● ○ ● ○
                                    ● ○
    ○○○  Male cluster             Gender-agnostic
   ○   ○                          representation
    ○ ○
```

### 5.3 Feature Alignment Loss

**목적**: 성별 그룹의 특징 분포 통계량 정렬

#### 5.3.1 Mean Alignment

$$\mu_f = \frac{1}{N_f} \sum_{i=1}^{N_f} z_f^{(i)}, \quad \mu_m = \frac{1}{N_m} \sum_{j=1}^{N_m} z_m^{(j)}$$

$$\mathcal{L}_{\text{align}} = \|\mu_f - \mu_m\|_2^2$$

#### 5.3.2 Variance Alignment

$$\sigma_f^2 = \frac{1}{N_f} \sum_{i=1}^{N_f} (z_f^{(i)} - \mu_f)^2$$

$$\sigma_m^2 = \frac{1}{N_m} \sum_{j=1}^{N_m} (z_m^{(j)} - \mu_m)^2$$

$$\mathcal{L}_{\text{var}} = \|\sigma_f^2 - \sigma_m^2\|_2^2$$

**코드 구현**:

```python
def _feature_alignment_loss(feat_f, feat_m):
    # Query pooling: (N, 100, 256) → (N, 256)
    pooled_f = feat_f.mean(dim=1)
    pooled_m = feat_m.mean(dim=1)
    
    # 그룹 평균
    mean_f = pooled_f.mean(dim=0)
    mean_m = pooled_m.mean(dim=0)
    mean_loss = F.mse_loss(mean_f, mean_m)
    
    # 그룹 분산 (최소 2개 샘플 필요)
    if pooled_f.size(0) >= 2 and pooled_m.size(0) >= 2:
        var_f = pooled_f.var(dim=0)
        var_m = pooled_m.var(dim=0)
        var_loss = F.mse_loss(var_f, var_m)
    else:
        var_loss = 0.0
    
    return mean_loss, var_loss
```

### 5.4 Score Distribution Loss

**목적**: 탐지 신뢰도 점수의 성별 간 격차 해소

**핵심 인사이트**: 편향된 모델은 여성 이미지에서 더 낮은 탐지 신뢰도를 보임

**단방향 패널티**: 여성 점수가 남성보다 낮을 때만 손실 발생

$$\mathcal{L}_{\text{gap}} = \text{ReLU}(\bar{s}_m - \bar{s}_f)$$

**분위수 정렬** (선택적):

$$\mathcal{L}_{\text{quantile}} = \frac{1}{3} \sum_{q \in \{0.25, 0.5, 0.75\}} \text{ReLU}(Q_q^m - Q_q^f)$$

$$\mathcal{L}_{\text{score}} = \mathcal{L}_{\text{gap}} + \mathcal{L}_{\text{quantile}}$$

**코드 구현**:

```python
def _score_distribution_loss(female_scores, male_scores):
    # 평균 점수 차이 (단방향)
    mean_f = female_scores.mean()
    mean_m = male_scores.detach().mean()  # 남성 기준 고정
    gap_loss = F.relu(mean_m - mean_f)    # 여성이 낮을 때만 패널티
    
    # 분위수 정렬
    if female_scores.numel() >= 3 and male_scores.numel() >= 3:
        q_levels = torch.tensor([0.25, 0.5, 0.75])
        q_f = torch.quantile(female_scores, q_levels)
        q_m = torch.quantile(male_scores.detach(), q_levels)
        quantile_loss = F.relu(q_m - q_f).mean()
        return gap_loss + quantile_loss
    
    return gap_loss
```

### 5.5 Detection Loss

**목적**: Perturbation 추가 후에도 탐지 성능 유지

**DETR 원본 손실 함수 사용**:

$$\mathcal{L}_{\text{det}} = \lambda_{\text{ce}} \mathcal{L}_{\text{ce}} + \lambda_{\text{bbox}} \mathcal{L}_{\text{bbox}} + \lambda_{\text{giou}} \mathcal{L}_{\text{giou}}$$

- $\mathcal{L}_{\text{ce}}$: Cross-entropy for classification
- $\mathcal{L}_{\text{bbox}}$: L1 loss for bounding box regression
- $\mathcal{L}_{\text{giou}}$: Generalized IoU loss

**Hungarian Matching**:

DETR의 bipartite matching을 사용하여 예측-GT 쌍 결정:

```python
def detection_loss(self, outputs, targets):
    loss_dict = self.criterion(outputs, targets)  # Hungarian matching 포함
    weight_dict = self.criterion.weight_dict
    total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys())
    return total, loss_dict
```

---

## 6. 학습 파이프라인

### 6.1 초기화 단계

```
1. 분산 학습 환경 설정 (optional)
2. DETR 모델 로드 및 동결
3. Generator, ProjectionHead 초기화
4. Optimizer 설정 (Adam, lr=1e-4)
5. 데이터로더 생성 (gender-balanced)
```

### 6.2 에폭별 학습 루프

```python
for epoch in range(epochs):
    for samples, targets, genders in train_loader:
        # 1. 성별별 배치 분리
        female_batch, female_targets = split_by_gender(samples, targets, "female")
        male_batch, male_targets = split_by_gender(samples, targets, "male")
        
        # 2. Perturbation 생성 및 적용
        female_perturbed = apply_generator(generator, female_batch)
        male_perturbed = apply_generator(generator, male_batch)
        
        # 3. DETR Forward (특징 + 탐지 결과)
        outputs_f, feat_f = detr.forward_with_features(female_perturbed)
        outputs_m, feat_m = detr.forward_with_features(male_perturbed)
        
        # 4. Projection
        proj_f = proj_head(feat_f)
        proj_m = proj_head(feat_m)
        
        # 5. 손실 계산
        contrast_loss = cross_gender_contrastive(proj_f, proj_m)
        align_loss, var_loss = feature_alignment(feat_f, feat_m)
        score_loss = score_distribution(female_scores, male_scores)
        det_loss = detection_loss(outputs_f, outputs_m, targets)
        
        total_loss = (λ_contrast * contrast_loss 
                    + λ_align * align_loss
                    + λ_var * var_loss
                    + λ_score * score_loss
                    + β * det_loss)
        
        # 6. Backpropagation (Generator + ProjHead만)
        total_loss.backward()
        clip_grad_norm_(...)
        optimizer.step()
```

### 6.3 유효 타겟 필터링

빈 bounding box를 가진 타겟 제외:

```python
# 유효한 타겟만 필터링 (빈 박스 제외)
valid_f_idx = [i for i, t in enumerate(female_targets) if t["boxes"].numel() > 0]
valid_f_targets = [female_targets[i] for i in valid_f_idx]

# outputs도 해당 인덱스만 추출
valid_outputs_f = {
    "pred_logits": outputs_f["pred_logits"][valid_f_idx],
    "pred_boxes": outputs_f["pred_boxes"][valid_f_idx],
}
```

### 6.4 메트릭 로깅

| 메트릭 | 설명 |
|--------|------|
| `g_contrast` | Contrastive fairness loss |
| `g_align` | Mean alignment loss |
| `g_var` | Variance alignment loss |
| `g_score` | Score distribution loss |
| `g_det` | Detection loss |
| `g_total` | 총 손실 |
| `delta_linf` | Perturbation L∞ norm |
| `delta_l2` | Perturbation L2 norm |
| `obj_score` | 평균 탐지 신뢰도 |
| `obj_score_f` | 여성 평균 탐지 신뢰도 |
| `obj_score_m` | 남성 평균 탐지 신뢰도 |

---

## 7. 수학적 기반

### 7.1 Contrastive Learning과 Mutual Information

InfoNCE loss는 하한 바운드로서 mutual information을 추정:

$$\mathcal{L}_{\text{InfoNCE}} \geq \log(N) - I(z_f; z_m)$$

본 시스템에서 cross-gender contrastive loss를 최소화하면:
- 여성 representation과 남성 representation 간의 mutual information 증가
- 결과적으로 성별 정보가 representation에서 제거됨

### 7.2 Domain Adaptation 관점

Feature alignment는 Maximum Mean Discrepancy (MMD)와 유사:

$$\text{MMD}(P_f, P_m) = \|\mathbb{E}_{x \sim P_f}[\phi(x)] - \mathbb{E}_{x \sim P_m}[\phi(x)]\|_\mathcal{H}$$

Mean alignment loss는 이의 finite-sample 추정치:

$$\mathcal{L}_{\text{align}} \approx \|\hat{\mu}_f - \hat{\mu}_m\|_2^2$$

### 7.3 Adversarial Perturbation 이론

Perturbation bound ε는 adversarial robustness 관점에서 중요:

$$\|δ\|_\infty \leq \epsilon$$

- 너무 작음 (ε < 0.01): 모델 행동 변화 불충분
- 너무 큼 (ε > 0.2): 이미지 왜곡이 심해 비현실적
- 최적 범위 (ε ≈ 0.05-0.1): 미묘하지만 효과적인 변화

---

## 8. 하이퍼파라미터 분석

### 8.1 주요 하이퍼파라미터

| 파라미터 | 기본값 | 역할 | 권장 범위 |
|----------|--------|------|-----------|
| `epsilon` | 0.08 | Perturbation 크기 제한 | [0.03, 0.15] |
| `temperature` | 0.1 | Contrastive softmax sharpness | [0.05, 0.5] |
| `lambda_contrast` | 1.0 | Contrastive loss 가중치 | [0.5, 2.0] |
| `lambda_align` | 0.5 | Mean alignment 가중치 | [0.1, 1.0] |
| `lambda_var` | 0.1 | Variance alignment 가중치 | [0.01, 0.5] |
| `lambda_score` | 0.3 | Score alignment 가중치 | [0.1, 0.5] |
| `beta` | 0.6 | Detection loss 가중치 | [0.3, 1.0] |
| `lr_g` | 1e-4 | Generator 학습률 | [1e-5, 5e-4] |
| `proj_dim` | 128 | Projection 차원 | [64, 256] |

### 8.2 Temperature의 효과

$$\text{softmax}(z_i / \tau)$$

- **낮은 τ (0.05)**: Sharp distribution → Hard assignment, 학습 초기 불안정
- **높은 τ (0.5)**: Smooth distribution → Soft assignment, 학습 느림
- **최적 τ (0.1)**: 균형점, 대부분의 contrastive learning에서 효과적

### 8.3 손실 가중치 균형

```
Detection 중심 설정:        Fairness 중심 설정:
β = 0.8                     β = 0.3
λ_contrast = 0.5            λ_contrast = 1.5
→ 탐지 성능 우선             → 공정성 우선

균형 설정 (권장):
β = 0.6
λ_contrast = 1.0
λ_align = 0.5
λ_score = 0.3
```

---

## 9. 설계 결정 및 트레이드오프

### 9.1 GAN vs Contrastive: 왜 Contrastive인가?

| 측면 | GAN (WGAN) | Contrastive |
|------|------------|-------------|
| 학습 안정성 | 낮음 (모드 붕괴 위험) | 높음 |
| 하이퍼파라미터 민감도 | 높음 | 중간 |
| 수렴 속도 | 느림/불안정 | 빠름/안정 |
| 이론적 해석 | 어려움 | Mutual Information |
| 구현 복잡도 | 높음 (D/G 번갈아 학습) | 낮음 |

### 9.2 Query Pooling 전략

DETR의 100개 query를 처리하는 방법:

1. **Mean Pooling (채택)**: 간단하고 효과적
2. **Max Pooling**: High-confidence query에 편향
3. **Attention Pooling**: 추가 파라미터 필요
4. **Top-K Selection**: 하이퍼파라미터 추가

Mean pooling은 모든 query의 정보를 균등하게 반영하여 전체 탐지 패턴을 포착합니다.

### 9.3 단방향 Score Loss

양방향 대신 단방향 패널티를 사용하는 이유:

```
양방향: |score_f - score_m|
→ 남성 점수도 낮출 수 있음 (전체 성능 저하)

단방향: ReLU(score_m - score_f)
→ 여성 점수를 남성 수준으로 올림 (바람직한 방향)
```

### 9.4 DETR 동결의 의미

**동결하는 이유**:
1. Pre-trained 지식 보존
2. 학습 안정성 향상
3. Perturbation의 효과만 분리하여 분석 가능

**동결하지 않으면**:
- DETR 자체가 변형되어 원래 용도와 달라짐
- 학습 불안정
- 많은 GPU 메모리 필요

---

## 10. 결론 및 확장 가능성

### 10.1 주요 기여

1. **GAN-free Fairness Learning**: Discriminator 없이 안정적인 fairness 학습
2. **Cross-Gender Contrastive**: 전통적 contrastive의 역전을 통한 성별 정보 제거
3. **다중 손실 함수 설계**: 탐지 성능과 공정성의 균형

### 10.2 확장 가능성

#### 다른 보호 속성으로 확장

```python
# 인종, 연령 등으로 확장 가능
for attr_a, attr_b in protected_attribute_pairs:
    contrast_loss += cross_group_contrastive(proj_a, proj_b)
    align_loss += feature_alignment(feat_a, feat_b)
```

#### 다른 탐지 모델에 적용

```python
# YOLO, Faster R-CNN 등에 적용 가능
class FrozenYOLO:
    def forward_with_features(self, x):
        # 유사한 인터페이스 구현
        return outputs, features
```

#### Self-Supervised Pre-training

```python
# 라벨 없는 데이터로 fairness-aware representation 학습
class FairSSL:
    def __init__(self):
        self.proj_head = ProjectionHead(...)
        # SimCLR 스타일 augmentation
```

### 10.3 한계점

1. **성별 이진 분류**: Non-binary 성별 고려 안 함
2. **Intersectionality**: 성별 외 다른 속성과의 교차 효과 미고려
3. **데이터셋 의존성**: FAAP 데이터셋 구조에 특화

### 10.4 향후 연구 방향

1. **Multi-attribute Fairness**: 여러 보호 속성 동시 처리
2. **Interpretability**: Perturbation이 어떤 시각적 특징을 변경하는지 분석
3. **Real-world Deployment**: 실제 환경에서의 효과 검증

---

## 부록: 코드 실행 예시

### A.1 학습 실행

```bash
python train_faap_contrastive_1st.py \
    --dataset_root /path/to/faap_dataset \
    --epochs 30 \
    --batch_size 4 \
    --epsilon 0.08 \
    --lambda_contrast 1.0 \
    --temperature 0.1
```

### A.2 분산 학습

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_faap_contrastive_1st.py \
    --distributed \
    --batch_size 2
```

### A.3 체크포인트에서 재개

```bash
python train_faap_contrastive_1st.py \
    --resume faap_outputs/faap_outputs_1st/checkpoints/epoch_0010.pth
```

---

## 11. 7th WGAN-GD 버전과의 비교 분석

본 섹션에서는 최고 성능을 기록한 `train_faap_wgan_GD_7th.py`와 현재 Contrastive 버전의 핵심 차이점을 분석합니다.

### 11.1 전체 아키텍처 비교

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    7th WGAN-GD vs Contrastive 1st 아키텍처 비교                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   7th WGAN-GD:                          Contrastive 1st:                        │
│   ┌─────────────────────┐               ┌─────────────────────┐                 │
│   │    Generator (G)    │               │    Generator (G)    │                 │
│   └──────────┬──────────┘               └──────────┬──────────┘                 │
│              │                                     │                             │
│              ▼                                     ▼                             │
│   ┌─────────────────────┐               ┌─────────────────────┐                 │
│   │    Frozen DETR      │               │    Frozen DETR      │                 │
│   └──────────┬──────────┘               └──────────┬──────────┘                 │
│              │                                     │                             │
│        ┌─────┴─────┐                         ┌─────┴─────┐                       │
│        │           │                         │           │                       │
│        ▼           ▼                         ▼           ▼                       │
│   ┌─────────┐ ┌─────────┐               ┌─────────┐ ┌─────────────┐             │
│   │Detection│ │Gender   │               │Detection│ │ Projection  │             │
│   │  Head   │ │Discrim. │               │  Head   │ │    Head     │             │
│   │         │ │   (D)   │               │         │ │  (No Adv.)  │             │
│   └────┬────┘ └────┬────┘               └────┬────┘ └──────┬──────┘             │
│        │           │                         │             │                     │
│        │      Adversarial                    │        Contrastive                │
│        │        Training                     │         Learning                  │
│        │           │                         │             │                     │
│        ▼           ▼                         ▼             ▼                     │
│   ┌─────────────────────┐               ┌─────────────────────┐                 │
│   │  det_loss + fair    │               │  det_loss + contrast│                 │
│   │  + wasserstein      │               │  + align + score    │                 │
│   └─────────────────────┘               └─────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 11.2 핵심 구성요소 비교표

| 구성요소 | 7th WGAN-GD | Contrastive 1st |
|----------|-------------|-----------------|
| **Fairness 메커니즘** | GenderDiscriminator (적대적 학습) | ProjectionHead (대조 학습) |
| **학습 방식** | 2-player min-max game | 단일 최적화 |
| **Discriminator** | ✅ 존재 (별도 학습) | ❌ 없음 |
| **Projection Head** | ❌ 없음 | ✅ 존재 |
| **Optimizer 수** | 2개 (opt_g, opt_d) | 1개 (opt_g) |
| **학습 안정성** | 상대적 불안정 | 안정적 |
| **모드 붕괴 위험** | 있음 | 없음 |

### 11.3 손실 함수 수학적 비교

#### 11.3.1 7th WGAN-GD 손실 함수

**전체 손실 구조**:

$$\mathcal{L}^{7th}_{\text{total}} = \lambda_{\text{fair}} \cdot \mathcal{L}_{\text{fair}} + \beta \cdot \mathcal{L}_{\text{det}} + \lambda_w \cdot \mathcal{L}_{\text{wasserstein}}$$

**1. Discriminator 손실** (D 업데이트):

$$\mathcal{L}_D = \text{CE}(D(f_{\text{female}}), 1) + \text{CE}(D(f_{\text{male}}), 0)$$

- 여성 → 클래스 1, 남성 → 클래스 0으로 분류 학습
- Cross-entropy 기반 이진 분류

**2. Fairness 손실** (G 업데이트, D를 속이는 방향):

$$\mathcal{L}_{\text{fair}} = s_f \cdot \mathcal{L}_{\text{fair}}^f + s_m \cdot \mathcal{L}_{\text{fair}}^m$$

$$\mathcal{L}_{\text{fair}}^f = -(\text{CE}(D(f_f), 1) + \alpha \cdot H(D(f_f)))$$

$$\mathcal{L}_{\text{fair}}^m = -(\text{CE}(D(f_m), 0) + \alpha \cdot H(D(f_m)))$$

여기서:
- $s_f = 1.0$, $s_m = 0.5$ (여성에 더 높은 가중치)
- $H(\cdot)$ = 엔트로피 정규화
- 음수 부호: D가 틀리도록 G를 학습

**3. Wasserstein 정렬 손실** (단방향):

$$\mathcal{L}_{\text{wasserstein}} = \frac{1}{K} \sum_{k=1}^{K} \text{ReLU}(s_m^{(k)} - s_f^{(k)})$$

- $s_f^{(k)}$, $s_m^{(k)}$: 정렬된 detection score의 k번째 값
- 단방향: 여성 score < 남성 score일 때만 패널티

**4. Epsilon 스케줄링**:

```
Warmup (0→8 epoch):   0.05 → 0.10
Hold (8→14 epoch):    0.10 유지
Cooldown (14→24):     0.10 → 0.09
```

#### 11.3.2 Contrastive 1st 손실 함수

**전체 손실 구조**:

$$\mathcal{L}^{Con}_{\text{total}} = \lambda_c \cdot \mathcal{L}_{\text{contrast}} + \lambda_a \cdot \mathcal{L}_{\text{align}} + \lambda_v \cdot \mathcal{L}_{\text{var}} + \lambda_s \cdot \mathcal{L}_{\text{score}} + \beta \cdot \mathcal{L}_{\text{det}}$$

**1. Cross-Gender Contrastive 손실**:

$$\mathcal{L}_{\text{contrast}} = \frac{1}{2}\left(\mathcal{L}_{f \to m} + \mathcal{L}_{m \to f}\right)$$

$$\mathcal{L}_{f \to m} = -\text{logsumexp}\left(\frac{z_f \cdot z_m^T}{\tau}\right) + \log(N_m)$$

**2. Feature Alignment 손실**:

$$\mathcal{L}_{\text{align}} = \|\mu_f - \mu_m\|_2^2$$

$$\mathcal{L}_{\text{var}} = \|\sigma_f^2 - \sigma_m^2\|_2^2$$

**3. Score Distribution 손실** (7th와 유사하지만 분위수 추가):

$$\mathcal{L}_{\text{score}} = \text{ReLU}(\bar{s}_m - \bar{s}_f) + \frac{1}{3}\sum_{q \in \{0.25, 0.5, 0.75\}} \text{ReLU}(Q_q^m - Q_q^f)$$

### 11.4 학습 동역학 비교

#### 7th WGAN-GD 학습 루프

```python
for epoch in epochs:
    for batch in dataloader:
        # 1. Discriminator 업데이트 (k_d = 4회)
        for _ in range(k_d):
            with torch.no_grad():
                perturbed = generator(images)  # G 고정
                features = detr(perturbed)
            d_loss = CE(D(feat_f), 1) + CE(D(feat_m), 0)
            d_loss.backward()
            opt_d.step()
        
        # 2. Generator 업데이트 (1회)
        perturbed = generator(images)
        features = detr(perturbed)
        g_loss = -CE(D(feat_f), 1) - CE(D(feat_m), 0)  # D를 속이는 방향
        g_loss += detection_loss + wasserstein_loss
        g_loss.backward()
        opt_g.step()
```

**특징**:
- D를 k_d=4회 업데이트 후 G를 1회 업데이트
- D가 충분히 강해진 후 G가 학습해야 효과적
- 불균형 시 모드 붕괴 또는 학습 실패

#### Contrastive 1st 학습 루프

```python
for epoch in epochs:
    for batch in dataloader:
        # 단일 단계 업데이트
        perturbed = generator(images)
        features = detr(perturbed)
        proj_f, proj_m = proj_head(feat_f), proj_head(feat_m)
        
        loss = contrastive(proj_f, proj_m)
        loss += alignment(feat_f, feat_m)
        loss += detection_loss
        loss.backward()
        opt_g.step()  # Generator + ProjHead 동시 학습
```

**특징**:
- 단일 optimizer로 단순한 학습 루프
- 적대적 균형 문제 없음
- 안정적인 수렴

### 11.5 하이퍼파라미터 비교

| 파라미터 | 7th WGAN-GD | Contrastive 1st | 비고 |
|----------|-------------|-----------------|------|
| `epsilon` | 0.05→0.10→0.09 (스케줄) | 0.08 (고정) | 7th: 동적 스케줄링 |
| `beta` | 0.5→0.6 (스케줄) | 0.6 (고정) | 7th: 점진적 증가 |
| `lr_g` | 1e-4 | 1e-4 | 동일 |
| `lr_d` | 1e-4 | N/A | Contrastive: D 없음 |
| `k_d` | 4 | N/A | D 업데이트 횟수 |
| `lambda_fair` | 2.0 | N/A | |
| `lambda_contrast` | N/A | 1.0 | |
| `lambda_w` | 0.2 | N/A | |
| `lambda_score` | N/A | 0.3 | 유사한 역할 |
| `alpha` (entropy) | 0.2 | N/A | |
| `temperature` | N/A | 0.1 | Contrastive용 |
| `fair_f_scale` | 1.0 | N/A | 여성 가중치 |
| `fair_m_scale` | 0.5 | N/A | 남성 가중치 |
| `epochs` | 24 | 30 | |

### 11.6 Fairness 달성 메커니즘 비교

#### 7th WGAN-GD: 적대적 정보 제거

```
┌─────────────────────────────────────────────────────────────────┐
│                    Adversarial Information Removal               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   목표: D가 성별을 구분하지 못하게 만들기                          │
│                                                                  │
│   D의 목표:    P(gender=female|features) 정확히 예측              │
│   G의 목표:    D가 random guess (50%)하게 만들기                  │
│                                                                  │
│   학습 동역학:                                                    │
│   ┌─────────────────────────────────────────────────────┐       │
│   │  D: "이 특징은 여성이다" (확신 증가)                   │       │
│   │  G: "D가 구분 못하게 특징 변형" (perturbation 조정)    │       │
│   │  D: "다시 구분 시도" (적응)                           │       │
│   │  G: "더 강하게 변형" ...                              │       │
│   │  → 평형점: D 정확도 ≈ 50%                            │       │
│   └─────────────────────────────────────────────────────┘       │
│                                                                  │
│   수학적 게임:                                                    │
│   min_G max_D  E[log D(f)] + E[log(1-D(f))]                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Contrastive 1st: 표현 공간 혼합

```
┌─────────────────────────────────────────────────────────────────┐
│                  Contrastive Representation Mixing               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   목표: 성별 간 특징 거리를 최소화                                │
│                                                                  │
│   학습 전:                         학습 후:                       │
│                                                                  │
│   특징 공간:                       특징 공간:                     │
│   ┌───────────────┐               ┌───────────────┐             │
│   │  ● ● ●        │               │    ●○ ●○      │             │
│   │   ● ●  Female │               │  ○● ○●        │             │
│   │               │      →        │    ●○ ○●      │             │
│   │        ○ ○ ○  │               │  ●○ ●○        │             │
│   │         ○ ○   │               │    Mixed!     │             │
│   │          Male │               │               │             │
│   └───────────────┘               └───────────────┘             │
│                                                                  │
│   수학적 목표:                                                    │
│   min  D_KL(P_female || P_male) + D_KL(P_male || P_female)      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 11.7 장단점 비교

#### 7th WGAN-GD

**장점**:
- ✅ **검증된 최고 성능**: 실험에서 가장 좋은 fairness-detection 균형
- ✅ **강력한 adversarial signal**: D가 직접적으로 성별 정보 제거 유도
- ✅ **비대칭 가중치** (`fair_f_scale=1.0`, `fair_m_scale=0.5`): 여성에 집중
- ✅ **동적 스케줄링**: epsilon, beta가 학습 단계에 따라 조정

**단점**:
- ❌ 학습 불안정 (D/G 균형 민감)
- ❌ 하이퍼파라미터 튜닝 복잡 (k_d, lr_d, lr_g 균형)
- ❌ 모드 붕괴 가능성
- ❌ 추가 모델(D) 필요 → 메모리/계산 비용

#### Contrastive 1st

**장점**:
- ✅ **학습 안정성**: GAN 동역학 문제 없음
- ✅ **단순한 구조**: 단일 optimizer
- ✅ **이론적 명확성**: Mutual Information 기반
- ✅ **확장 용이**: 다른 보호 속성 추가 쉬움

**단점**:
- ❌ **성능 열위 가능성**: 적대적 신호보다 약할 수 있음
- ❌ **배치 크기 의존성**: Contrastive learning은 큰 배치에서 효과적
- ❌ **하이퍼파라미터 다수**: 5개 손실 가중치 조정 필요

### 11.8 7th의 성능 우위 분석

7th WGAN-GD가 더 좋은 성능을 보이는 이유:

#### 1. 직접적인 Adversarial Signal

```python
# 7th: D가 직접 "이건 여성이다/남성이다" 판단
fairness_f = -(CE(D(feat_f), 1) + alpha * entropy)
# → G는 D의 명확한 피드백을 받아 학습

# Contrastive: 간접적인 유사도 최적화
contrast_loss = -logsumexp(sim_f_to_m / tau)
# → 더 부드러운 gradient, 덜 직접적
```

#### 2. 비대칭 처리

```python
# 7th: 여성에 2배 가중치
fairness_loss = 1.0 * fairness_f + 0.5 * fairness_m

# Contrastive: 대칭적 처리
contrast_loss = (loss_f_to_m + loss_m_to_f) / 2
```

7th는 편향이 더 심한 여성 그룹에 집중하여 효율적으로 fairness 향상.

#### 3. 동적 스케줄링

```python
# 7th: 학습 단계별 최적화
epsilon: 0.05 → 0.10 → 0.09  # 초기 탐색 → 강화 → 안정화
beta:    0.5  → 0.6          # detection 가중치 점진 증가

# Contrastive: 고정값
epsilon = 0.08  # 단일 값
beta = 0.6      # 단일 값
```

### 11.9 개선 방향 제안

Contrastive 버전의 성능 향상을 위한 제안:

#### 1. 비대칭 Contrastive Loss 도입

```python
# 현재 (대칭):
contrast_loss = (loss_f_to_m + loss_m_to_f) / 2

# 제안 (비대칭):
contrast_loss = 1.5 * loss_f_to_m + 0.5 * loss_m_to_f
# 여성이 남성 방향으로 이동하는 것에 더 집중
```

#### 2. 하이브리드 접근

```python
# Contrastive + 경량 Discriminator
total_loss = (
    lambda_contrast * contrastive_loss
    + lambda_adv * light_adversarial_loss  # 작은 D 추가
    + lambda_align * alignment_loss
    + beta * detection_loss
)
```

#### 3. 동적 스케줄링 적용

```python
# Contrastive에도 스케줄링 적용
epsilon = schedule_epsilon(epoch)  # 7th 방식
lambda_contrast = schedule_lambda(epoch)  # 초기 높게 → 점진 감소
```

### 11.10 요약 비교표

| 측면 | 7th WGAN-GD | Contrastive 1st | 승자 |
|------|-------------|-----------------|------|
| **Fairness 성능** | 높음 | 중간 | 7th |
| **Detection 유지** | 좋음 | 좋음 | 동등 |
| **학습 안정성** | 낮음 | 높음 | Contrastive |
| **구현 복잡도** | 높음 | 낮음 | Contrastive |
| **하이퍼파라미터 수** | 많음 | 많음 | 동등 |
| **메모리 사용량** | 높음 (D 추가) | 낮음 | Contrastive |
| **이론적 기반** | GAN 이론 | InfoNCE/MI | Contrastive |
| **확장성** | 어려움 | 용이함 | Contrastive |

---

*본 문서는 train_faap_contrastive_1st.py의 완전한 기술 분석을 제공합니다.*
*버전: 10th (Contrastive Fairness)*
*최종 수정: 2026-01-12*
