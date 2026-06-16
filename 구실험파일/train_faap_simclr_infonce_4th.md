# Stabilized Gender-Aware Contrastive Learning (4th Version)

`train_faap_simclr_infonce_4th.py` 설계 문서

---

## 핵심 아이디어

**3개 버전의 장점만 결합하여 3rd의 과적합 문제를 근본적으로 해결**

```
4th = 3rd의 AP Gap 개선력 + fix2의 안정성 + 1st의 정규화
```

---

## 3rd 실패 원인 분석 (Deep Dive)

### 원인 1: Male gradient 오염 (Epoch 3 과적합)
```
3rd: loss = 1.5 * L(F→M) + 0.5 * L(M→F)
                                    ↑
                           Male feature가 Female 방향으로 이동
                           → Male AP 하락 → AP Gap 악화
```
- M→F 방향의 gradient가 Male representation을 변형
- Epoch 3까지는 F→M이 지배적이지만, 이후 M→F 효과가 누적

### 원인 2: Score Gap Reversal → Adaptive Weighting 역효과
```
기대: score_m > score_f → weight > 1.0 → 강한 학습
현실: 학습 중 score_m < score_f (역전!)
     → sigmoid(negative * 5) < 0.5
     → weight = 0.5 + 0.3 = 0.8 < 1.0
     → 필요한 쌍의 학습이 약해짐
```

### 원인 3: ProjectionHead 과적합
- 정규화 없는 2-layer MLP
- Small batch (8)에서 빠르게 과적합

---

## 4th 설계 결정

### 변경 1: Male Detach (fix2에서 검증)

```python
# 3rd: Male gradient 통과 (양방향)
proj_m = proj_all[male_idx]  # gradient 흐름
loss = 1.5 * F→M + 0.5 * M→F

# 4th: Male gradient 차단 (단방향)
proj_m_detached = proj_m.detach()  # gradient 차단
loss = F→M only  # M→F 제거
```

**효과**: fix2는 이 방식으로 epoch 29까지 안정적이었음

### 변경 2: Adaptive Weighting 완전 제거

```python
# 3rd: Score 차이 기반 가중치 (Score Gap Reversal로 역효과)
score_diff = scores_m - scores_f
weights = 0.5 + sigmoid(score_diff * 5)
sim_weighted = sim + alpha * log(weights)

# 4th: 균일 가중치 (단순하고 안정적)
# → Adaptive Weighting 코드 완전 제거
# → 모든 F→M 쌍을 동등하게 취급
```

### 변경 3: StabilizedProjectionHead

```python
# 3rd: 정규화 없음
nn.Linear(256, 256) → ReLU → nn.Linear(256, 128) → L2

# 4th: LayerNorm + Dropout
nn.LayerNorm(256) → nn.Linear(256, 256) → ReLU → Dropout(0.1) → nn.Linear(256, 128) → L2
```

### 변경 4: Feature Mean Alignment (1st에서 채택)

```python
# Projection space (contrastive loss)와 별도로
# 원본 DETR feature space에서도 분포 정렬
mean_f = features_f.mean(dim=1).mean(dim=0)      # Female 중심
mean_m = features_m.mean(dim=1).mean(dim=0).detach()  # Male 중심 (고정)
loss_align = MSE(mean_f, mean_m)  # Female 중심 → Male 중심
```

### 변경 5: Schedule 개선

| Schedule | 3rd | 4th | 근거 |
|----------|-----|-----|------|
| Epsilon | 0.10 고정 | 0.05→0.10→0.09 | WGAN 7th에서 검증 |
| Temperature | 0.07 | 0.1 | Smoother gradient |
| Contrastive | 즉시 적용 | 3 epoch warmup | 초기 안정화 |
| LR | 1e-4 고정 | Cosine (1e-4→1e-6) | 후반부 fine-tuning |
| Beta (det) | 0.5→0.6 | 0.5→0.6 (동일) | - |
| Augmentation | medium | weak | Detection 보호 |
| Grad clip | 0.1 | 0.5 | 더 유연한 학습 |

---

## 파이프라인

```
Input Image
     ↓
PerturbationGenerator (ε: 0.05→0.10→0.09 schedule)
     ↓
SimCLR Augmentation (weak: ColorJitter 0.2)
     ↓
Frozen DETR → outputs + features
     ↓
┌──────────────────┐     ┌──────────────────────────────────┐
│ Detection Score  │     │ StabilizedProjectionHead          │
│ (Top-K=10)       │     │ LayerNorm → Linear → ReLU        │
│                  │     │ → Dropout(0.1) → Linear → L2     │
└────────┬─────────┘     └──────────────┬───────────────────┘
         ↓                              ↓
┌─────────────────────────────────────────────────────────┐
│ DetachInfoNCELoss                                        │
│ • Female = Anchor, Male = Positive (DETACHED)            │
│ • Other Females = Negative                               │
│ • NO Adaptive Weighting                                  │
│ • NO M→F direction                                       │
│ • Temperature = 0.1                                      │
└─────────────────────────────────────────────────────────┘
         ↓
Total Loss = λ_c * warmup * L_contrastive   (1.0 * warmup)
           + λ_a * L_align                   (0.3)
           + λ_w * L_wasserstein             (0.2)
           + β * L_det                       (0.5→0.6)
```

---

## 하이퍼파라미터 전체 비교

| 파라미터 | 3rd | fix2 | 1st | **4th** | 근거 |
|----------|-----|------|-----|---------|------|
| temperature | 0.07 | 0.07 | 0.1 | **0.1** | 안정성 |
| epsilon | 0.10 고정 | 0.10 고정 | 0.08 고정 | **0.05→0.10→0.09** | 점진적 |
| male detach | X | O | X | **O** | 안정성 |
| M→F direction | O (0.5) | X | O (대칭) | **X** | Male 보호 |
| adaptive weight | O | O (softmax) | X | **X** | SGR 해결 |
| LayerNorm | X | X | O | **O** | 정규화 |
| Dropout | X | X | X | **O (0.1)** | 과적합 방지 |
| feature align | X | X | O | **O** | 분포 정렬 |
| LR schedule | 고정 | 고정 | 고정 | **Cosine** | fine-tuning |
| contrastive warmup | X | X | X | **O (3ep)** | 초기 안정화 |
| augmentation | medium | medium | X | **weak** | detection 보호 |
| grad clip | 0.1 | 0.1 | 0.1 | **0.5** | 유연성 |
| batch_size | 8 | 8 | 4 | **8** | - |
| epochs | 24 | 10 | 30 | **24** | - |

---

## 모니터링 지표

| 지표 | 의미 | 기대 방향 |
|------|------|-----------|
| `sim_f2m` | Female→Male cosine similarity | ↑ 증가 (feature alignment) |
| `sim_f2f` | Female→Female cosine similarity | 적정 유지 (collapse 아님) |
| `score_gap` | score_m - score_f | → 0에 가까워짐 |
| `loss_contrastive` | InfoNCE loss | ↓ 감소 후 안정화 |
| `loss_align` | Feature mean MSE | ↓ 감소 |

**과적합 감지**: `sim_f2m`이 급격히 1.0에 수렴하면 collapse 의심

---

## 기존 MoCo 4th 대비 설계 근거

| 관점 | MoCo 접근 | 현재 4th |
|------|-----------|----------|
| 문제 진단 | "batch size가 작아서 과적합" | "Male gradient + Adaptive Weighting이 원인" |
| 해결책 | Memory Bank + Momentum Centroid | Male Detach + Weighting 제거 |
| 복잡도 | 높음 (Queue, EMA, 추가 메모리) | 낮음 (detach 한 줄) |
| 검증 여부 | 미검증 | fix2에서 Male Detach 검증 완료 |
| 위험 | Memory Bank stale features 문제 | 낮은 위험 (검증된 조합) |

---

## 실행 방법

```bash
# 기본 실행
python train_faap_simclr_infonce_4th.py

# Augmentation 없이
python train_faap_simclr_infonce_4th.py --aug_strength none

# Temperature 조정
python train_faap_simclr_infonce_4th.py --temperature 0.07

# Feature alignment 가중치 조정
python train_faap_simclr_infonce_4th.py --lambda_align 0.5

# Resume
python train_faap_simclr_infonce_4th.py --resume faap_outputs/faap_outputs_infonce_4th/checkpoints/epoch_0009.pth
```

---

## 성공 기준

| 지표 | Baseline | WGAN 7th (논문) | 3rd (best) | **4th 목표** |
|------|----------|-----------------|------------|-------------|
| AP Gap | 0.1063 | 0.1059 (-0.4%) | 0.1044 (-1.8%) | **< 0.100 (~6%)** |
| AR Gap | 0.0081 | 0.0032 (-60.5%) | 0.0026 (-67.9%) | **< 0.005 (~38%)** |
| Female AP | 0.404 | 0.408 | 0.413 | **> 0.410** |
| Male AP | 0.511 | 0.514 | 0.517 | **>= 0.511** |

---

**작성일**: 2026-02-17
**기반 코드**: train_faap_simclr_infonce_3rd.py, train_faap_simclr_infonce_3rd_fix2.py, train_faap_contrastive_1st.py
