# Standard InfoNCE: Cross-Gender Contrastive Learning

`train_faap_simclr_infonce.py` 분석 문서

---

## 실험 결과: 실패

**결과**: AP Gap 0.106 → 0.110 (악화)

---

## 핵심 아이디어

GenderDiscriminator를 **표준 InfoNCE**로 대체

### 7th (Adversarial) vs InfoNCE

| 항목 | 7th | InfoNCE |
|------|-----|---------|
| 핵심 | GenderDiscriminator로 성별 구분 | Cross-gender contrastive |
| Loss | Adversarial (d_loss + fairness_loss) | InfoNCE |
| 학습 | D/G 번갈아 학습 | G만 학습 |

### InfoNCE 수식

```
L = -log(exp(sim(z_f, z_m)/τ) / [exp(sim(z_f, z_m)/τ) + Σ exp(sim(z_f, z_f')/τ)])
```

- **Positive**: cross-gender (여성 ↔ 남성)
- **Negative**: same-gender (여성 ↔ 여성, 남성 ↔ 남성)

---

## 주요 컴포넌트

### 1. SimCLRAugmentation

Detection 친화적 augmentation:

| Strength | 구성 |
|----------|------|
| none | 비활성화 |
| weak | ColorJitter(0.2) |
| **medium** (기본) | ColorJitter(0.3) |
| strong | ColorJitter(0.4) + Grayscale |

### 2. SimCLRProjectionHead

```
입력: (batch, num_queries, 256)
  ↓ mean pooling
(batch, 256)
  ↓ Linear(256 → 256) + ReLU
  ↓ Linear(256 → 128)
  ↓ L2 Normalize
출력: (batch, 128)
```

### 3. CrossGenderInfoNCELoss

```python
temperature = 0.07  # SimCLR 표준
asymmetric_f = 1.5  # Female → Male (강화)
asymmetric_m = 0.5  # Male → Female (약화)
```

---

## Loss 구성

```python
total_g = (
    lambda_infonce * loss_infonce      # 1.0
    + lambda_wass * loss_wasserstein   # 0.2
    + beta * loss_det                  # 0.5→0.6
)
```

---

## 실패 원인 분석

1. **Semantic 유사성 부재**
   - Cross-gender를 positive로 정의했지만, 실제로 여성/남성 이미지 간 semantic 유사성이 없음
   - Contrastive learning은 augmented view 간 유사성에 기반하는데, 성별 간에는 그런 관계 없음

2. **목표와 연결 안됨**
   - AP Gap 감소 목표와 cross-gender contrastive가 직접 연결되지 않음
   - Feature를 가깝게 만든다고 detection 성능이 같아지지 않음

3. **분포 붕괴 위험**
   - 모든 feature를 같게 만들려다 보니 오히려 다양성 손실

---

## 하이퍼파라미터

| 파라미터 | 값 |
|----------|-----|
| temperature | 0.07 |
| asymmetric_f | 1.5 |
| asymmetric_m | 0.5 |
| lambda_infonce | 1.0 |
| lambda_wass | 0.2 |
| aug_strength | medium |
| proj_dim | 128 |

---

## 교훈

> "단순히 성별 간 feature를 가깝게 만드는 것으로는 AP Gap을 해결할 수 없다"

→ **Score-based Contrastive** 제안으로 이어짐
