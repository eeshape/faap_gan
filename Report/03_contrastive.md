# 3. Contrastive 계열 (Cross-Gender InfoNCE Fairness)

**파일**: `train_faap_contrastive_1st.py` ~ `3rd.py`, `train_faap_contrastive_iou.py`
**기간**: 2026-01-10 ~ 2026-01-20
**핵심**: Discriminator 없이 Feature 공간에서 직접 성별 정렬

---

## 3.1 파이프라인

```
┌───────────────────────────────────────────────────────────────┐
│             Contrastive Fairness Pipeline                      │
│                                                               │
│  [Female Image]        [Male Image]                           │
│       │                     │                                 │
│       ▼                     ▼                                 │
│  Generator(G)          Generator(G)                           │
│       │                     │                                 │
│  Perturbed_f           Perturbed_m                            │
│       │                     │                                 │
│       ▼                     ▼                                 │
│  ┌──────────────────────────────────┐                         │
│  │           Frozen DETR            │                         │
│  └────┬─────────┬──────┬───────────┘                          │
│       │         │      │                                      │
│  Detection   Features  Scores                                 │
│  Outputs   (B×100×256) (matched)                              │
│       │         │       │                                     │
│  L_det    ProjectionHead  L_score                             │
│       │      │      │     (Wasserstein)                       │
│       │    z_f    z_m                                         │
│       │      │      │                                         │
│       │    InfoNCE Loss                                       │
│       │   (cross-gender                                       │
│       │    positive)                                          │
│       │         │                                             │
│       └────┬────┘                                             │
│            ▼                                                  │
│  L = λ_c·L_contrast + λ_s·L_score + λ_a·L_align              │
│    + λ_v·L_var + β·L_det                                     │
│                                                               │
│  ※ Discriminator 없음 (GAN-Free)                              │
└───────────────────────────────────────────────────────────────┘
```

---

## 3.2 핵심 모듈: ProjectionHead

```python
class ProjectionHead(nn.Module):
    # DETR feature → contrastive embedding
    # LayerNorm(256) → Linear(256→256) → ReLU → Linear(256→128) → L2-normalize
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=128):
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),          # ← 입력 정규화
            nn.Linear(input_dim, hidden_dim),  # 256 → 256
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim), # 256 → 128
        )
    def forward(hs):
        pooled = hs.mean(dim=1)              # (B, 100, 256) → (B, 256)
        return F.normalize(self.net(pooled), dim=-1)  # L2 정규화
```

---

## 3.3 Loss 수식

### Cross-Gender Contrastive Loss (Soft Uniform Assignment)

표준 InfoNCE와 달리, **모든 cross-gender pair를 positive로** 취급하는 logsumexp 방식:

$$L_{f \to m} = -\text{logsumexp}(\text{sim}(z_f, z_m) / \tau, \text{dim}=1).\text{mean}() + \log(N_m)$$

$$L_{m \to f} = -\text{logsumexp}(\text{sim}(z_m, z_f) / \tau, \text{dim}=1).\text{mean}() + \log(N_f)$$

- **모든 cross-gender pair가 positive**: 각 female에 대해 모든 male이 positive (표준 InfoNCE의 1:N 구조와 다름)
- $\log(N)$: 정규화 상수 (sample 수에 따른 보정)
- $\tau$: temperature (낮을수록 sharp)
- 1st, 2nd: 대칭 평균 $L_c = (L_{f \to m} + L_{m \to f}) / 2$
- 3rd: 비대칭 가중치 $L_c = 1.5 \cdot L_{f \to m} + 0.5 \cdot L_{m \to f}$

### Feature Alignment Loss

Raw DETR decoder feature (projected embedding이 아님)에 대해 정렬:

$$L_{align} = \text{MSE}(\bar{f}_f, \bar{f}_m) \quad (\text{mean feature 정렬, } f \in \mathbb{R}^{B \times 100 \times 256})$$
$$L_{var} = \text{MSE}(\text{Var}(f_f), \text{Var}(f_m)) \quad (\text{분산 정렬})$$

- $f_f, f_m$: DETR decoder의 raw feature (B×100×256), ProjectionHead 통과 전
- mean/var는 query 차원(dim=1)에 대해 계산

### Score Distribution Loss (Mean Gap + Quantile Alignment)

WGAN-GD의 sorted Wasserstein과 달리, **mean gap + quantile 정렬** 방식:

$$L_{score} = \text{ReLU}(\bar{s}_m - \bar{s}_f) + \frac{1}{|Q|}\sum_{q \in Q} \text{ReLU}(s_{m}^{(q)} - s_{f}^{(q)})$$

- $\bar{s}$: 평균 score gap (단방향: male > female일 때만 패널티)
- $Q = \{0.25, 0.50, 0.75\}$: quantile 위치
- $s^{(q)}$: 해당 quantile에서의 score 값
- male score는 `detach()` 처리 (타겟으로 고정)

### Total Generator Loss

$$L_G = \lambda_c \cdot L_{InfoNCE} + \lambda_s \cdot L_{score} + \lambda_a \cdot L_{align} + \lambda_v \cdot L_{var} + \beta \cdot L_{det}$$

---

## 3.4 버전별 상세

### Contrastive 1st — GAN-Free 기반 확립

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `epochs` | 30 | |
| `batch_size` | 4 | |
| `epsilon` | 0.08 (고정) | Warmup 없음 |
| `beta` | 0.6 (고정) | |
| `lambda_contrast` | 1.0 | 대칭 contrastive |
| `temperature` | 0.1 | |
| `lambda_align` | 0.5 | Mean 정렬 |
| `lambda_var` | 0.1 | 분산 정렬 |
| `lambda_score` | 0.3 | Score Wasserstein |
| `proj_dim` | 128 | Projection 출력 차원 |

**특징**: 대칭적 contrastive (F→M = M→F 동일 가중치)

**결과**: AR Gap 0.0031 (-61%) — 7th와 동등한 AR Gap, GAN-free로 달성

### Contrastive 2nd — 7th 스케줄링 통합

| 파라미터 | 값 | 1st 대비 변경 |
|----------|-----|---------------|
| `epochs` | 24 | -6 |
| `batch_size` | 5 | +1 |
| `epsilon` | 0.05→0.10→0.09 | **3-phase 도입** |
| `beta` | 0.2→0.6 | **선형 증가** |
| `lambda_contrast` | 1.5 | +0.5 (전체 가중치 강화) |
| `temperature` | 0.07 | -0.03 (sharper) |
| `lambda_align` | 0.4 | -0.1 |
| `lambda_var` | 0.15 | +0.05 |
| `lambda_score` | 0.4 | +0.1 (강화) |

**여전히 대칭**: $L_c = \lambda_c \cdot (L_{f \to m} + L_{m \to f}) / 2$ — `lambda_contrast=1.5`는 전체 가중치이며, 방향별 비대칭이 아님

**결과**: AP Gap 0.115 — **악화** (스케줄링 조합 실패)

### Contrastive 3rd — 비대칭 Contrastive 도입

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `epochs` | 30 | |
| `batch_size` | 5 | |
| `epsilon` | 0.08 (고정) | 단순화 (1st로 회귀) |
| `beta` | 0.6 (고정) | 단순화 |
| `lambda_contrast` | 1.0 | 전체 contrastive 가중치 |
| `contrast_f_scale` | 1.5 | 여성→남성 강조 |
| `contrast_m_scale` | 0.5 | 남성→여성 억제 |
| `temperature` | 0.1 | |

**핵심 변경**: 2nd의 복잡한 스케줄링 제거 + **최초로 방향별 비대칭 가중치 도입**

**Loss**: $L_c = \lambda_c \cdot (1.5 \cdot L_{f \to m} + 0.5 \cdot L_{m \to f})$ (3:1 비율, `_asymmetric_cross_gender_contrastive_loss` 함수 사용)

---

## 3.5 Contrastive IoU — IoU 인지 Contrastive

```
┌────────────────────────────────────────────┐
│       IoU-Aware Contrastive Pipeline        │
│                                            │
│  DETR Detection → Hungarian Matching       │
│       │                                    │
│  IoU 계산 (예측 vs GT)                      │
│       │                                    │
│  ┌────▼─────┐  ┌──────────┐  ┌──────────┐ │
│  │ High IoU │  │ Mid IoU  │  │ Low IoU  │ │
│  │ (>τ)     │  │ (무시)   │  │ (<τ)     │ │
│  │ Positive │  │ Margin   │  │ Anchor   │ │
│  └──────────┘  └──────────┘  └──────────┘ │
│       │                           │        │
│       └──── InfoNCE ──────────────┘        │
│                                            │
│  + EMA Prototype Alignment                 │
│  proto_f ← momentum·proto_f + (1-m)·z_f   │
│  L_proto = MSE(proto_f, proto_m)           │
└────────────────────────────────────────────┘
```

**핵심 혁신**: Localization 품질(IoU)을 contrastive sampling에 반영 → AP Gap 직접 타겟

**특징**:
- High-IoU detection을 positive, Low-IoU를 anchor로 사용
- EMA prototype으로 안정적 타겟 제공
- AP 개선과 직접 연관된 학습 신호

---

## 3.6 비교 요약

| 버전 | Discriminator | 대칭성 | Epsilon | AP Gap | AR Gap | 평가 |
|------|:---:|:---:|---|---|---|:---:|
| 1st | X | 대칭 | 0.08 고정 | 0.108 | **0.0031** | GAN-free 가능 확인 |
| 2nd | X | **대칭** | 3-phase | **0.115** | 0.0069 | **악화** (스케줄링 과다) |
| 3rd | X | **비대칭** (최초) | 0.08 고정 | - | - | 비대칭 가중치 도입 |
| IoU | X | - | - | - | - | AP 직접 타겟 |

### 핵심 교훈
1. GAN-free로도 AR Gap 60%+ 개선 가능
2. 스케줄링 통합이 항상 좋은 것은 아님 (2nd 실패)
3. 단순한 구조가 복잡한 조합보다 나은 경우가 많음
