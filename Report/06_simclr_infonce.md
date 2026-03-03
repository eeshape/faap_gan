# 6. SimCLR InfoNCE 계열

**파일**: `train_faap_simclr_infonce.py` (base) ~ `train_faap_simclr_infonce_9th.py`
**기간**: 2026-01-20 ~ 2026-02-17
**총 버전**: 13개 (base, 2nd, 3rd, fix1, fix2, fix4, 4th~9th)
**핵심 성과**: 3rd — 전체 연구 최고 AP Gap 0.1044 (-1.8%)

---

## 6.1 파이프라인

```
┌─────────────────────────────────────────────────────────────────┐
│            SimCLR InfoNCE Fairness Pipeline                      │
│                                                                  │
│  [Female Image]          [Male Image]                            │
│       │                       │                                  │
│       ▼                       ▼                                  │
│  Generator(G)            Generator(G)                            │
│       │                       │                                  │
│  Perturbed_f             Perturbed_m                             │
│       │                       │                                  │
│       ▼                       ▼                                  │
│  SimCLR Augmentation     SimCLR Augmentation                     │
│  (ColorJitter, 학습 시)   (ColorJitter, 학습 시)                  │
│       │                       │                                  │
│       ▼                       ▼                                  │
│  ┌──────────────────────────────────────────────┐                │
│  │               Frozen DETR                    │                │
│  └──┬──────────────────────────────┬────────────┘                │
│     │ outputs                      │ features (B×100×256)        │
│     │                              │                             │
│  L_det(f,m)               ProjectionHead (2-layer MLP)          │
│     │                         │               │                  │
│     │                       z_f             z_m                  │
│     │                         │               │                  │
│     │             Cross-Gender InfoNCE Loss                      │
│     │         Positive: cross-gender (F↔M)                      │
│     │         Negative: same-gender  (F↔F, M↔M)                 │
│     │         비대칭: 1.5×L(F→M) + 0.5×L(M→F)                   │
│     │                         │                                  │
│     │               (3rd: Adaptive Score Weighting)              │
│     │         w = 0.5 + sigmoid((score_m - score_f) × 5)        │
│     │                         │                                  │
│     │     Wasserstein Score Alignment (단방향)                    │
│     │     ReLU(sorted_m - sorted_f).mean()                       │
│     │                         │                                  │
│     └───────────┬─────────────┘                                  │
│                 ▼                                                 │
│  L = λ_infonce·L_infonce + λ_w·L_w + β·L_det                    │
│                                                                   │
│  ※ Discriminator 없음 (GAN-Free)                                 │
│  ※ Generator만 학습 (D/G 교대 없음)                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6.2 SimCLR Augmentation 테이블

Detection 친화적으로 설계. 공간 변환(Crop, Flip 등)은 제외하고 색상 변환만 적용하여 Bounding Box label을 보존한다.

| 강도 | ColorJitter | RandomGrayscale | 설명 |
|------|-------------|-----------------|------|
| `none` | - | - | Augmentation 없음 (baseline 비교용) |
| `weak` | (0.2, 0.2, 0.2, 0.05) | - | Detection 안전, 보수적 시작점 |
| `medium` | (0.3, 0.3, 0.3, 0.1) | - | **기본값 (추천)**, Detection 친화적 |
| `strong` | (0.4, 0.4, 0.4, 0.1) | p=0.2 | 표준 SimCLR 세팅, Detection 성능 저하 위험 |

구현 방식: ImageNet 정규화 해제 → ColorJitter 적용 (이미지별 독립) → 재정규화.

---

## 6.3 Loss 수식

### Cross-Gender InfoNCE Loss (Multi-Positive Logsumexp)

모든 cross-gender sample을 positive로 취급하는 logsumexp 기반 loss:

$$L_{F \to M} = -\left(\text{logsumexp}(\text{sim}(z_f, z_m) / \tau) - \text{logsumexp}([\text{sim}(z_f, z_m); \text{sim}(z_f, z'_f)] / \tau)\right).\text{mean}()$$

- **Positive**: 모든 남성 sample (multi-positive — 표준 InfoNCE의 single-positive과 다름)
- **Negative**: 같은 성별의 다른 sample (자기 자신 제외, 대각선 마스킹)
- $\tau$: temperature (0.07)
- 분자: `logsumexp(sim_f2m, dim=1)` — 모든 cross-gender 유사도
- 분모: `logsumexp([sim_f2m; sim_f2f_masked], dim=1)` — 전체 유사도

비대칭 결합 (base, 3rd):
$$L_{InfoNCE}^{total} = 1.5 \cdot L_{F \to M} + 0.5 \cdot L_{M \to F}$$

### 3rd 전용 — Adaptive Score Weighting

$$w_{ij} = 0.5 + \sigma\!\left(5 \cdot (score_{m,j} - score_{f,i})\right)$$

$$\tilde{s}_{ij} = s_{ij} + \alpha \cdot \log(w_{ij} + \epsilon)$$

여성이 저성능, 남성이 고성능인 쌍에 더 강한 학습 신호를 부여. Score Gap Reversal 현상 발생 시 역효과.

### Wasserstein Score Alignment (단방향)

$$L_w = \frac{1}{K}\sum_{k=1}^{K} \text{ReLU}(\hat{s}_{m,k} - s_{f,k})$$

- $\hat{s}_{m,k}$: 남성 score (detach — 타겟 고정)
- 여성 score < 남성 score일 때만 패널티 부여

### Detection Loss

$$L_{det} = L_{CE} + L_{bbox} + L_{giou} \quad (\text{DETR criterion})$$

### Total Generator Loss

$$L = \lambda_{InfoNCE} \cdot L_{InfoNCE} + \lambda_w \cdot L_w + \beta(t) \cdot L_{det}$$

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| $\lambda_{InfoNCE}$ | 1.0 | InfoNCE 손실 가중치 |
| $\lambda_w$ | 0.2 | Wasserstein 손실 가중치 |
| $\beta$ | 0.5 → 0.6 | Detection 손실 가중치 (선형 증가) |

---

## 6.4 버전 진화 테이블

| 버전 | Epochs | Batch | Epsilon | Temp | 핵심 변경 | 결과 |
|------|:------:|:-----:|---------|:----:|-----------|------|
| **base** | 24 | 4 | 0.05→0.10→0.09 (warmup) | 0.07 | 표준 InfoNCE, Symmetric warmup schedule | 기준선 |
| **2nd** | 24 | 4 | 0.05→0.10→0.09 | 0.07 | **Score-Based Contrastive** (성별 무시, Detection Score 기준 high/low 분리) | - |
| **3rd** | 24 | 8 | **0.10 고정** | 0.07 | **Gender-Aware + Adaptive Score Weighting** | **AP Gap 0.1044 (최고)** |
| **3rd_fix1** | **15** | 8 | 0.10 고정 | 0.10 | Fair Centroid Alignment, 대칭(1:1), Cosine LR | **실패** (과적합 방지 실패) |
| **3rd_fix2** | **10** | 8 | 0.10 고정 | 0.07 | Male-Anchored, **Male Detach**, softmax 가중치 | 안정, AP Gap -1.2% |
| **3rd_fix4** | 24 | 8 | 0.10 고정 | 0.07 | SupCon 정규화 (base_temperature 스케일링) | 미평가 |
| **4th** | 24 | 8 | 0.05→0.10→0.09 | **0.10** | **Male Detach + StabilizedProjectionHead** (LayerNorm+Dropout), Cosine LR, MoCo 미채택 | 미평가 |
| **5th** | 24 | **6** | 0.10 고정 | 0.07 | Controlled Aggression (M→F 가중치 점진적 감소 ep0-8) | 미평가 |
| **6th** | 24 | 6 | 0.10 고정 | 0.07 | Clamped Adaptive Weighting (Score Gap Reversal 원천 차단) | 미평가 |
| **7th** | 24 | 6 | 0.10 고정 | 0.07 | Dual Gap Direct Minimization (AP/AR Gap 직접 공격) | 미평가 |
| **8th** | 24 | 6 | 0.10 고정 | 0.07 | Asymmetric Detection Loss (Female det 강화, Male 억제) | 미평가 |
| **9th** | 24 | 6 | 0.10 고정 | 0.07 | 8th + RecallGapLoss (AR Gap 직접 최소화) | 미평가 |

**공통 변경점 (base → 3rd 이후 전 버전)**: epsilon 0.10 고정, batch 6 이상.

---

## 6.5 3rd 상세 분석 — 전체 연구 최고 AP Gap

### 3rd 핵심 설계

**2nd 실패 원인**: 2nd는 성별 정보를 사용하지 않고 Detection Score의 median 기준으로 high/low를 분리하는 `ScoreBasedContrastiveLoss`를 사용. 그러나 배치 내 `obj_score_f ≈ obj_score_m`으로 high/low 그룹에 남녀가 균등 분포 → 성별 공정성 개선 효과 없음 → AP Gap 오히려 악화.

**3rd 해결책**: Gender + Score 명시적 결합.

```
Anchor  : 여성 이미지 (Female)
Positive: 남성 이미지 (Male)       ← 탐지 잘 되는 영역으로 당김
Negative: 다른 여성 이미지          ← 탐지 안 되는 영역에서 멀리
```

Adaptive Weight 수식:
```python
score_diff = scores_m.unsqueeze(0) - scores_f.unsqueeze(1)  # (N_f, N_m)
weights = 0.5 + torch.sigmoid(score_diff * 5)               # [0.5, 1.5]
sim_f2m_weighted = sim_f2m + alpha * log(weights + 1e-8)
```

### Epoch별 성능 (3rd 실험 결과)

| Epoch | Male AP | Female AP | AP Gap | AR Gap | 비고 |
|-------|---------|-----------|--------|--------|------|
| Baseline | 0.511 | 0.404 | 0.1063 | 0.0081 | DETR 원본 |
| **Epoch 3** | **0.517** | **0.413** | **0.1044** | **0.0026** | **전체 연구 최고** |
| Epoch 10 | 0.517 | 0.413 | 0.1050 | 0.0040 | Early Stop 권장 시점 |
| Epoch 23 | 0.518 | 0.408 | 0.1090 | 0.0080 | 과적합 (Baseline 수준으로 복귀) |

- **Best AP Gap**: 0.1044 (-1.8%) — InfoNCE 계열 및 전체 연구 최고 기록
- **Best AR Gap**: 0.0026 (-67.9%) — WGAN 7th의 0.0032보다도 우수
- **문제**: Epoch 3 이후 급격한 과적합 → Early Stop 없이는 장기 유지 불가

### Score Gap Reversal 현상

3rd의 핵심 실패 원인. 학습 중 배치 내에서 `score_m < score_f`(음수 gap)가 관측됨.

```
기대: score_m > score_f → sigmoid(positive) → weight > 1.0 → 강한 학습
현실: score_m < score_f → sigmoid(negative) → weight < 0.5 → 학습 약화
     → Adaptive Weighting이 역효과 → 과적합 심화
```

학습 로그: `score_gap (M-F)` = -0.007 ~ -0.014 (지속적 음수).

### WGAN 7th vs InfoNCE 3rd 비교

| 지표 | Baseline | WGAN 7th (논문) | InfoNCE 3rd Best (ep3) | InfoNCE 3rd Final (ep23) |
|------|----------|-----------------|------------------------|--------------------------|
| Male AP | 0.511 | 0.514 (+0.3%) | 0.517 (+0.6%) | 0.518 (+0.7%) |
| Female AP | 0.404 | 0.408 (+0.4%) | **0.413 (+0.9%)** | 0.408 (+0.4%) |
| **AP Gap** | 0.1063 | 0.1059 (-0.4%) | **0.1044 (-1.8%)** | 0.1090 (+2.5%) |
| **AR Gap** | 0.0081 | **0.0032 (-60.5%)** | 0.0026 (-67.9%) | 0.0080 (-1.2%) |
| 안정성 | - | 24 epoch 안정 | Epoch 3 이후 불안정 | 과적합 |

**해석**: InfoNCE 3rd는 Epoch 3에서 전체 실험 최고 AP Gap을 기록했으나, 이후 과적합으로 7th에 비해 최종 결과가 열위. 대표 결과(논문)로 7th 채택.

---

## 6.6 핵심 하이퍼파라미터 비교 테이블

| 파라미터 | base | 2nd | **3rd** | fix1 | fix2 | fix4 | 4th | 5th | 6th~9th |
|----------|:----:|:---:|:-------:|:----:|:----:|:----:|:---:|:---:|:-------:|
| `epochs` | 24 | 24 | 24 | **15** | **10** | 24 | 24 | 24 | 24 |
| `batch_size` | 4 | 4 | **8** | 8 | 8 | 8 | 8 | **6** | **6** |
| `epsilon` | 0.05→0.10→0.09 | 0.05→0.10→0.09 | **0.10** | 0.10 | 0.10 | 0.10 | 0.05→0.10→0.09 | 0.10 | 0.10 |
| `temperature` | 0.07 | 0.07 | 0.07 | **0.10** | 0.07 | 0.07 | **0.10** | 0.07 | 0.07 |
| `asymmetric` | 1.5:0.5 | 1.5:0.5 (low→high) | 1.5:0.5 | **1:1** | F→M only | 1.5:0.5 | F→M only | 1.5:감쇠→0 | 1.5:0.5 |
| `male_detach` | X | X | X | X | **O** | X | **O** | X | - |
| `adaptive_w` | X | X | **O** | X | softmax | SupCon | X | X | clamped |
| `aug_strength` | medium | medium | medium | medium | medium | medium | **weak** | medium | medium |
| `lr_schedule` | 고정 | 고정 | 고정 | **Cosine** | 고정 | 고정 | **Cosine** | 고정 | **Cosine/warmup** |

---

## 6.7 fix 변형 상세

### 3rd_fix1 — Fair Centroid (실패)

**아이디어**: 여성과 남성의 가중 평균 centroid를 "공정 centroid"로 정의하고 양쪽 모두를 그 방향으로 당김.

```python
# EMA 업데이트 (momentum=0.9)
centroid_f = 0.9 * centroid_f + 0.1 * current_f
centroid_m = 0.9 * centroid_m + 0.1 * current_m

# Fair Centroid: 저성능 그룹(Female) 쪽으로 치우침 (7:3)
fair_centroid = 0.7 * centroid_f + 0.3 * centroid_m  # ← 단순 평균 아님
```

**변경**: 비대칭(1.5:0.5) → 대칭(1:1), epochs 24 → 15, temperature 0.07 → 0.10, Cosine LR, Dropout(0.1) 추가.

**결과**: 과적합 방지 목적이었으나 실패. Loss가 -10에 즉시 수렴 (similarity → 1.0), Representation Collapse 발생.

### 3rd_fix2 — Male-Anchored Detach (안정적)

**아이디어**: Male projection을 detach하여 gradient 흐름 차단 → Male representation이 Female에 의해 왜곡되는 것을 방지.

```python
proj_m_detached = proj_m.detach()  # gradient 차단
# F→M만 학습, M→F 제거
score_weight_m = F.softmax(scores_m * 5, dim=0)  # softmax 가중치
```

**결과**: Epoch 29까지 안정적. AP Gap -1.2% 수준 유지. 4th 설계의 핵심 검증 근거.

### 3rd_fix4 — SupCon 정규화 (미평가)

**아이디어**: SupCon(Supervised Contrastive Learning) 논문의 base_temperature 정규화 적용.

```python
loss_scaled = loss * (temperature / base_temperature)  # = 1.0 (base=0.07 동일 시)
```

**결과**: 미평가 (실험 진행 전 4th 설계로 전환).

---

## 6.8 4th 설계 의도 (Male Detach + Stabilized, 미평가)

3rd의 3가지 실패 원인을 동시에 해결하려는 시도. 초기에는 MoCo Memory Bank를 고려했으나, 3rd의 실패가 batch size가 아닌 구조적 문제(Male gradient, Adaptive Weighting, 정규화 부재)에서 기인한다고 판단하여 **MoCo를 채택하지 않고** 검증된 fix2 조합을 기반으로 설계:

| 원인 | 3rd | 4th 해결책 |
|------|-----|------------|
| Male gradient 오염 | M→F 0.5 가중치 | **Male Detach** (`proj_m.detach()`, fix2에서 검증) |
| Score Gap Reversal | Adaptive Weighting | Weighting **완전 제거** |
| ProjectionHead 과적합 | 정규화 없음 | **StabilizedProjectionHead** (LayerNorm + Dropout 0.1) |

추가: Feature Mean Alignment (MSE between female/male feature centroids), Cosine LR schedule, 3 epoch contrastive warmup, M→F 방향 완전 제거 (F→M only), temperature 0.07→0.10, augmentation medium→weak.

---

## 6.9 5th~9th 설계 방향

| 버전 | 핵심 아이디어 | 3rd 대비 차이 |
|------|-------------|---------------|
| **5th** | Controlled Aggression — M→F 가중치를 ep0-3 유지 후 ep4-8에 걸쳐 선형 감쇠 | Adaptive Weighting 제거, M→F 점진적 제거 |
| **6th** | Clamped Adaptive — Score Gap Reversal 시 weight를 clamp하여 역효과 차단 | 3rd adaptive 개선판 |
| **7th** | Dual Gap Direct — AP Gap과 AR Gap을 proxy 없이 직접 최소화 | 간접 proxy → 직접 Gap 공격 |
| **8th** | Asymmetric Detection Loss — Female det loss 강화, Male det loss 억제 | Detection loss 비대칭 적용 |
| **9th** | 8th + RecallGapLoss — AR Gap 직접 최소화 term 추가 | 8th + 7th 결합 |

6th~9th 공통: `batch_size=6`, `epsilon=0.10 고정`, `temperature=0.07`.

---

## 6.10 교훈

1. **성별 정보의 명시적 활용이 효과적**: Score만으로 분리(2nd)보다 Gender를 직접 Anchor/Positive로 지정(3rd)하는 방식이 확연히 우수. 성별 정보를 feature 학습에 명시적으로 통합하는 것이 공정성 개선의 핵심.

2. **비대칭 학습이 일관되게 우수**: `L(F→M) > L(M→F)` 비대칭이 전 버전에 걸쳐 대칭 방식보다 좋은 결과를 보임. 여성→남성 방향에 집중하는 것이 단방향 정렬 전략과 일치.

3. **Male Detach가 과적합 방지의 핵심**: fix2에서 확인. Male projection을 detach하면 Male representation이 Female contrastive 학습에 의해 왜곡되지 않아 장기 안정성이 높아짐. Male AP 보호와 Female AP 향상을 동시에 달성.

4. **Representation-Performance Gap 존재**: InfoNCE로 feature space를 정렬해도 Detection AP로 즉각 반영되지 않는 간극이 존재. Feature 유사도 향상이 탐지 성능 향상으로 직결되지 않는 경우가 있음 — 특히 epoch 3 이후의 과적합이 이를 잘 보여줌.

5. **Early Stop이 필수**: 3rd는 Epoch 3에서 최고 성능을 기록하지만 이후 급격히 악화. InfoNCE 계열은 검증 지표 모니터링 기반의 Early Stop 없이는 최적 상태를 유지할 수 없음.

6. **Epsilon 고정이 단순하고 효과적**: base/2nd의 3단계 warmup schedule보다 epsilon=0.10 고정이 3rd 이후 버전에서 일관되게 사용됨. 복잡한 스케줄링이 항상 유리하지 않음.
