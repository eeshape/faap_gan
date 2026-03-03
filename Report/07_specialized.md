# 7. Specialized 접근법: Score Contrastive & Direct Boost

> **Phase 4 후반부** (2026-01-27 ~ 2026-01-31)
> WGAN-GD 및 InfoNCE 3rd에서 확인된 한계를 극복하기 위한 두 가지 특수 목적 접근법.

---

## 7.1 개요

Phase 4의 핵심 접근법인 InfoNCE 3rd가 AP Gap 1.8% 개선이라는 최고 기록을 달성했으나, epoch 3 이후 과적합이 급격히 진행되는 불안정성 문제가 있었다. 이를 해결하고자 두 가지 방향으로 실험이 분기되었다.

| 접근법 | 파일 | 핵심 아이디어 | 결과 |
|--------|------|---------------|------|
| **Score-Based Contrastive v2** | `train_faap_score_contrastive.py` | 고정 threshold 제거, 배치 내 상대적 ranking으로 Positive/Anchor 분리 | AP Gap 0.1049 (-1.3%) |
| **Direct Confidence Boosting (fix3)** | `train_faap_direct_boost_fix3.py` | Contrastive loss 제거, female detection confidence 직접 최적화 | AP Gap 0.1119 (+0.56%) **실패** |

---

## 7.2 Score-Based Contrastive v2

### 7.2.1 v1의 문제점과 v2의 개선 동기

Score-Based Contrastive v1에서 고정 threshold(0.5)를 사용했을 때, 실제 image-level detection score가 배치 내에서 대부분 0.9 이상에 분포하여 Positive anchor가 생성되지 않는 문제가 발생했다. Loss가 작동하지 않으면서 학습 신호가 소실되는 현상이 관찰되었다.

**v2 핵심 해결책**: 고정 threshold를 배치 내 상대적 ranking 기반으로 교체

```
[v1] 고정 threshold = 0.5
     → score 대부분 0.90+ → threshold 초과 → Anchor 생성 안 됨 → Loss ≈ 0

[v2] 배치 내 상위 K% = Positive (고성능 anchor)
     배치 내 하위 K% = Anchor (저성능, 이동 대상)
     → 항상 균형 있는 Positive/Anchor 보장
```

### 7.2.2 Pipeline 구조

```
Input Image
    ↓
PerturbationGenerator (epsilon 0.05→0.10→0.09 schedule)
    ↓
Perturbed Image → FrozenDETR → Detection Outputs + Features (hs[-1]: B×100×256)
    ↓
[이미지 단위 Detection Score 계산]
  matcher(outputs, targets) → 매칭된 detection의 평균 score → image_scores (N,)
    ↓
[Adaptive Percentile Split]
  argsort(descending=True)
  ├── 상위 top_k% (기본 40%) → Positive (고성능 샘플)
  ├── 하위 bottom_k% (기본 40%) → Anchor (저성능 샘플)
  └── 중간 20% → margin region (무시)
    ↓
ProjectionHead (Linear 256→512→128, BN, ReLU, L2 normalize)
    ↓
AdaptiveScoreContrastiveLoss (InfoNCE, temperature=0.1)
    ↓
+ λ_wass × L_wasserstein (성별 기반, 보조)
+ β × L_det
```

### 7.2.3 Adaptive Percentile-Based Sampling 상세

**분리 기준**:

| 그룹 | 조건 | 역할 |
|------|------|------|
| **Positive** | 배치 내 상위 top_k% (기본 40%) | 고성능 anchor, 목표 representation |
| **Margin** | 중간 20% | 무시 (경계 영역 제외) |
| **Anchor** | 배치 내 하위 bottom_k% (기본 40%) | 저성능 샘플, Positive 방향으로 당김 |

**겹침 방지 로직**:
```python
if n_top + n_bottom > n:
    n_top = n // 2
    n_bottom = n - n_top
```

**실제 문서에 기재된 파라미터 (args 기준)**:
- `top_k_percent = 0.4` (40%), `bottom_k_percent = 0.4` (40%) → 중간 20% margin
- 보고서 제목에 25%로 표기된 경우는 v1 설정 기준이며, v2 코드 기본값은 40%/40%

### 7.2.4 InfoNCE Loss 수식

Anchor(저성능)를 Positive(고성능) 방향으로 당기는 단방향 InfoNCE:

$$L_{\text{contrast}} = -\frac{1}{|A|} \sum_{i \in A} \log \frac{\exp(\text{logsumexp}(\text{sim}(z_i, z_j^+) / \tau))}{\exp(\text{logsumexp}(\text{sim}(z_i, z_k) / \tau))}$$

- $A$: Anchor 집합 (하위 K%)
- $z_j^+$: Positive 집합 (상위 K%)의 projection
- $z_k$: 전체 배치 (자기 자신 제외, `-inf` masking 적용)
- $\tau = 0.1$ (temperature)

**양방향 옵션** (`--bidirectional`): Anchor→Positive 손실에 Positive 내 clustering 손실을 비대칭 가중치로 추가
- `anchor_weight = 1.0`, `positive_weight = 0.3`

### 7.2.5 전체 손실 함수

$$L_{\text{total}} = w_c(t) \cdot \lambda_c \cdot L_{\text{contrast}} + \lambda_w \cdot L_{\text{wass}} + \beta(t) \cdot L_{\text{det}}$$

| 항 | 가중치 | 설명 |
|----|--------|------|
| $L_{\text{contrast}}$ | $\lambda_c = 1.0$ | Adaptive percentile InfoNCE |
| $L_{\text{wass}}$ | $\lambda_w = 0.2$ | 단방향 Wasserstein (female→male 정렬, 보조) |
| $L_{\text{det}}$ | $\beta = 0.5 \to 0.6$ | DETR detection loss |
| $w_c(t)$ | `contrastive_warmup_epochs=3` | Warmup 가중치: $t / T_w$ (0→1) |

**Warmup 설계 의도**: 초반 학습 불안정을 방지하기 위해 Contrastive loss를 처음에는 약하게 적용하고 점진적으로 강화.

### 7.2.6 주요 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|----------|----|------|
| `temperature` | 0.1 | v1(0.07) 대비 안정성 향상 |
| `top_k_percent` | 0.4 | 배치 내 상위 40% |
| `bottom_k_percent` | 0.4 | 배치 내 하위 40% |
| `contrastive_warmup_epochs` | 3 | Warmup 에폭 수 |
| `lambda_contrastive` | 1.0 | Contrastive 손실 가중치 |
| `lambda_wass` | 0.2 | Wasserstein 보조 손실 |
| `lr_g` | 1e-4 | Generator + ProjectionHead 공동 학습률 |
| `epochs` | 30 | 총 학습 에폭 |
| `batch_size` | 7 | (balance_genders=True) |
| `epsilon` | 0.05→0.10→0.09 | Warmup 8ep / Hold 6ep / Cooldown 16ep |

### 7.2.7 결과

| Epoch | AP Gap | Δ AP Gap | 비고 |
|-------|--------|----------|------|
| 29 (best) | **0.1049** | **-1.3%** | Score v2 최고 성능 |

- Baseline AP Gap 0.1063 대비 1.3% 개선
- 성별 정보 없이 detection score만으로 분리 → 배치 내 male/female이 고르게 분포 시 효과적
- InfoNCE 3rd (ep3) 기준 AP Gap 0.1044와 비교 시 소폭 열세

---

## 7.3 Direct Confidence Boosting (fix3)

### 7.3.1 설계 동기: 이전 접근법의 구조적 한계 분석

fix3는 InfoNCE 계열 접근법의 근본적 한계에 대한 비판적 분석에서 출발했다.

**한계 1: Representation-Performance Gap**
- InfoNCE는 cosine similarity 최적화, 즉 feature space 정렬 문제를 해결
- DETR의 detection confidence는 logit의 절대적 크기(magnitude)에 의존
- Feature 공간에서 female과 male이 유사해져도 → detection score 개선으로 직결되지 않음

**한계 2: Mean Pooling Information Bottleneck**
- `hs[-1]`: (B, 100, 256) → `mean(dim=1)` → (B, 256)
- 100개 object query 정보를 단일 벡터로 압축
- 개별 object의 confidence 정보, 공간 정보 손실
- Female의 특정 객체에서 낮은 confidence가 평균에 묻혀 학습 신호 약화

**한계 3: Detection Score에 직접 Gradient 없음**
- Contrastive loss: representation level의 간접 최적화
- Female detection confidence를 직접 높이는 gradient 신호 부재

**fix3 해결 전략**: Contrastive loss를 완전히 제거하고, female detection confidence를 per-object 수준에서 직접 최적화

### 7.3.2 Pipeline 구조

```
Input Image (mixed gender batch)
    ↓
PerturbationGenerator (epsilon=0.10, 고정)
    ↓
Perturbed Image → FrozenDETR → pred_logits (B, 100, num_classes+1)
    ↓
[Gender Split]
  female_idx → outputs_f (pred_logits, pred_boxes)
  male_idx   → outputs_m (pred_logits, pred_boxes)
    ↓
[Per-Object Confidence 추출]
  softmax(pred_logits)[..., :-1].max(dim=-1)  # "no object" 클래스 제외
  → conf_f_all (N_f, 100), conf_m_all (N_m, 100)
    ↓
[Top-K Selection]
  topk(k=10, dim=1) → conf_f (N_f, 10), conf_m (N_m, 10)
    ↓
DirectConfidenceBoostLoss
  ├── L_boost: Hard sample mining weighted -log(conf_f)
  ├── L_gap: max(0, mean(conf_m) - mean(conf_f) - margin)
  └── L_threshold: penalty for conf_f < 0.3
    ↓
+ λ_w × L_wasserstein (per-object Wasserstein)
+ β × L_det
```

### 7.3.3 DirectConfidenceBoostLoss 상세

**구성 요소 1: L_boost — Hard Sample Mining Weighted Confidence Boosting**

$$w = \text{softmax}((\theta - c_f) \cdot \beta_{\text{hard}})$$
$$L_{\text{boost}} = \frac{1}{N_f} \sum_i \sum_k \left[ -\log(c_{f,i,k} + \varepsilon) \cdot w_{i,k} \right]$$

- $c_{f,i,k}$: 이미지 $i$의 top-k 번째 female object confidence
- $\theta = 0.7$: target confidence (목표값)
- $\beta_{\text{hard}} = 5.0$: hard mining 강도 (높을수록 낮은 confidence에 집중)
- $w$: $(\theta - c_f)$가 클수록 (즉, confidence가 낮을수록) 높은 가중치

**직관**: 현재 confidence가 목표 confidence와 멀리 떨어진 (어려운) 샘플에 더 강한 학습 신호를 집중시킴.

**구성 요소 2: L_gap — Gender Gap Reduction**

$$L_{\text{gap}} = \max\left(0,\ \overline{c_m} - \overline{c_f} - m\right)$$

- $\overline{c_m}$: male top-k confidence 평균 (`.detach()`, gradient 차단)
- $\overline{c_f}$: female top-k confidence 평균
- $m = 0.0$: 허용 margin
- Male을 타겟으로 고정하고 female gap만 줄임

**구성 요소 3: L_threshold — Threshold Penalty (추가 구성 요소)**

$$L_{\text{threshold}} = \frac{1}{N} \sum_{i,k} \mathbb{1}[c_{f,i,k} < 0.3] \cdot (0.3 - c_{f,i,k})$$

threshold = 0.3 미만인 low-confidence object에 추가 패널티.

**전체 손실**:

$$L_{\text{boost\_total}} = L_{\text{boost}} + 0.5 \cdot L_{\text{gap}} + 0.3 \cdot L_{\text{threshold}}$$

$$L_{\text{total}} = \lambda_{\text{boost}} \cdot L_{\text{boost\_total}} + \lambda_w \cdot L_{\text{wass}} + \beta(t) \cdot L_{\text{det}}$$

### 7.3.4 Per-Object Wasserstein Loss

mean pooling bottleneck 우회를 위해 per-object top-k confidence를 flatten하여 Wasserstein 계산:

```python
flat_f = conf_f.flatten().sort().values   # (N_f × k,)
flat_m = conf_m.flatten().detach().sort().values
# 단방향: female < male일 때만 패널티
return F.relu(flat_m - flat_f).mean()
```

### 7.3.5 주요 하이퍼파라미터

| 파라미터 | 값 | 비고 |
|----------|----|------|
| `target_conf` | 0.7 | Hard mining 목표 confidence |
| `hard_mining_beta` | 5.0 | Softmax 온도 (hard mining 강도) |
| `gap_margin` | 0.0 | Gap loss 허용 margin |
| `top_k` | 10 | Per-image 사용 object 수 |
| `lambda_boost` | 1.0 | Direct boost 손실 가중치 |
| `lambda_wass` | 0.3 | Wasserstein 보조 손실 |
| `beta` | 0.3→0.5 | Detection loss 가중치 (낮게 시작) |
| `lr_g` | 5e-5 | 더 낮은 학습률 (직접 최적화) |
| `epsilon` | 0.10 | 고정 (스케줄링 없음) |
| `epochs` | 20 | 총 학습 에폭 |
| `lr_scheduler` | cosine | CosineAnnealingLR |

### 7.3.6 결과

| Epoch | AP Gap | Δ AP Gap | 비고 |
|-------|--------|----------|------|
| 13 (best) | 0.1119 | **+0.56%** | **실패** |

- Baseline AP Gap 0.1063 대비 오히려 악화
- **실패 원인 분석**:
  - Female confidence를 직접 높이는 gradient가 perturbation generator를 통해 역전파되었으나, 동일한 perturbation이 male image에도 적용되어 male confidence 역시 함께 상승
  - Gender-agnostic perturbation 구조상 female만 선택적으로 향상시키는 것이 불가능
  - L_boost가 배치 전체의 detection confidence를 높이는 방향으로 수렴 → male이 더 큰 수혜 (baseline AP가 높아 absolute gain 더 큼)
  - Loss landscape 불안정: L_boost + L_gap + L_threshold + L_wass + L_det 다섯 가지 손실 간 충돌

---

## 7.4 두 접근법 비교

### 7.4.1 성능 비교표

| 항목 | Score v2 | Direct Boost (fix3) |
|------|----------|---------------------|
| **AP Gap (최고)** | **0.1049** (-1.3%) | 0.1119 (+0.56%) |
| **Δ AP Gap** | **-1.3%** | **+0.56% (악화)** |
| **Best Epoch** | 29 | 13 |
| **학습 안정성** | 보통 (warmup으로 초반 안정) | 불안정 |
| **구조 복잡도** | 중간 (ProjectionHead 추가) | 낮음 (추가 모듈 없음) |
| **Gender 정보 활용** | 없음 (score-only ranking) | 있음 (explicit split) |
| **Gradient 경로** | Representation → Loss | Confidence → Loss (직접) |
| **주요 실패 원인** | - | Male도 함께 상승 |

### 7.4.2 설계 철학 비교

| 관점 | Score v2 | Direct Boost (fix3) |
|------|----------|---------------------|
| **문제 정의** | 낮은 score를 높은 score 방향으로 당김 | Female confidence를 직접 target confidence로 올림 |
| **최적화 대상** | Feature space (representation) | Detection logit (직접) |
| **Gender 정보** | 암묵적 (score ranking에 내포) | 명시적 (gender split 후 분리 처리) |
| **병목 우회** | 부분적 (mean pooling 여전히 사용) | 완전 (per-object confidence 직접 사용) |
| **이론적 근거** | Hard example mining (InfoNCE) | Focal loss 스타일 hard mining |

### 7.4.3 실패에서 얻은 교훈

**Score v2의 교훈**:
- 성별 정보 없이 detection score ranking만으로는 제한적 효과
- 배치 내 low/high score 샘플에 male/female이 고르게 분포할 경우 성별 격차 해소에 간접적으로만 기여
- InfoNCE 3rd (Gender-Aware)와 비교 시, 명시적 gender 정보 활용이 더 효과적임을 재확인

**Direct Boost (fix3)의 교훈**:
- **근본적 한계 확인**: Input perturbation 프레임워크에서 gender-specific gradient를 생성할 수 없음
  - Generator가 이미지 전체에 동일한 perturbation을 적용하므로, female image와 male image에 동일한 변화를 주게 됨
  - Female confidence boost를 위한 gradient가 필연적으로 male에도 영향
- Per-object confidence를 직접 사용해도, perturbation의 실제 효과는 이미지 수준에서 발생
- 이론적으로 합리적인 설계도 프레임워크 구조적 제약에 의해 실패할 수 있음

---

## 7.5 Phase 4 전체 맥락에서의 위치

```
InfoNCE 1st (Cross-Gender) ── 실패
    ↓
Score-Based v1/v2 ── AP Gap -1.3%    ←── 7.2절
    ↓
InfoNCE 2nd (Score 기반) ── 실패
    ↓
★ InfoNCE 3rd (Gender-Aware) ── AP Gap -1.8% (최고)
    ├── fix1 (Fair Centroid) ── loss 포화, 실패
    ├── fix2 (Male-Anchored) ── AP Gap -1.2%, 안정적
    ├── fix3 (Direct Boost) ── AP Gap +0.56%, 실패  ←── 7.3절
    └── fix4 (SupCon) ── 미평가
```

Score v2와 fix3는 각각 서로 다른 방향에서 InfoNCE 3rd의 한계를 극복하려 했으나, 두 접근법 모두 **input perturbation + frozen DETR 프레임워크의 구조적 제약**이라는 동일한 벽에 부딪혔다. 이는 이후 향후 연구 방향에서 프레임워크 자체를 전환해야 한다는 결론의 주요 근거가 되었다.

---

*이전 장: [06. Phase 4 InfoNCE 상세](./06_infonce.md)*
*다음 장: [08. 전체 비교 요약 및 결론](./08_comparison.md)*
