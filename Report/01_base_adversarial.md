# 1. Base Adversarial (기본 적대적 학습)

**파일**: `train_faap.py`, `train_faap_2nd.py`
**기간**: 2025-11-24
**핵심**: GAN 기반 공정성 학습의 출발점

---

## 1.1 파이프라인

```
┌─────────────────────────────────────────────────────────┐
│              Base Adversarial Pipeline                    │
│                                                         │
│  [Female Image] ──→ Generator(G) ──→ δ                  │
│                           │                              │
│  Perturbed_f = Image + ε·tanh(δ)                        │
│                           │                              │
│              ┌────────────▼──────────┐                   │
│              │      Frozen DETR      │                   │
│              └────┬─────────┬────────┘                   │
│                   │         │                            │
│            Detection     Features                        │
│            Outputs      (B×100×256)                       │
│                   │         │                            │
│              L_det     Discriminator(D)                   │
│                         │        │                       │
│                      CE_loss  Entropy                    │
│                         └───┬────┘                       │
│                        L_fairness                        │
│                                                         │
│  L_total = L_fairness + β · L_det                       │
│                                                         │
│  ※ 남성 이미지는 perturbation 없이 D 학습에만 사용        │
└─────────────────────────────────────────────────────────┘
```

**특징**: 여성 이미지에만 perturbation 적용 (단방향 교란)

---

## 1.2 Loss 수식

### Generator Loss

$$L_G = L_{fairness} + \beta \cdot L_{det}$$

**Fairness Loss** (Discriminator를 혼란시키는 적대적 손실):

$$L_{fairness} = -(CE(\hat{y}_f, 1) + \alpha \cdot H(\hat{y}_f))$$

여기서:
- $CE(\hat{y}_f, 1)$: 여성 sample의 cross-entropy (label=female)
- $H(\hat{y}_f) = -\sum p_i \log p_i$: 예측 확률의 entropy
- 음수 부호: D가 성별 분류를 **못하도록** G를 학습

**Detection Loss**:

$$L_{det} = \sum_{k \in \{ce, bbox, giou\}} w_k \cdot l_k$$

DETR criterion 그대로 사용 (검출 성능 유지 목적)

### Discriminator Loss

$$L_D = \frac{1}{2}[CE(\hat{y}_f, 1) + CE(\hat{y}_m, 0)]$$

- Female: label=1, Male: label=0
- DETR decoder feature에서 성별 분류

---

## 1.3 하이퍼파라미터

### train_faap.py (1st)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `epochs` | 30 | 학습 에폭 수 |
| `batch_size` | 8 | 배치 크기 |
| `lr_g` | 1e-4 | Generator 학습률 |
| `lr_d` | 1e-4 | Discriminator 학습률 |
| `k_d` | 2 | D 업데이트 횟수/iteration |
| `epsilon` | 0.05 | 시작 epsilon |
| `epsilon_final` | 0.12 | 최대 epsilon (warmup 후) |
| `epsilon_warmup_epochs` | 5 | Warmup 에폭 수 |
| `alpha` | 0.2 | Entropy 가중치 |
| `beta` | 0.7 | Detection loss 가중치 |
| `max_norm` | 0.1 | Gradient clipping |

**Epsilon 스케줄**: 단순 선형 Warmup (Hold/Cooldown 없음)
```
epoch 0 ─────→ epoch 5: 0.05 → 0.12 (선형)
epoch 5 ─────→ end:     0.12 유지
```

### train_faap_2nd.py

| 파라미터 | 값 | 1st 대비 변경 |
|----------|-----|---------------|
| `epochs` | 25 | -5 |
| `k_d` | 1 | -1 (D 학습 줄임) |
| `alpha` | 0.1 | -0.1 (entropy 줄임) |
| `beta` | 1.0 | +0.3 (detection 더 보존) |

**변경 근거**: D 업데이트를 줄이고 detection 보존에 집중 → 학습 안정성 우선

---

## 1.4 학습 루프

```
for each epoch:
    for each batch (samples, targets, genders):
        1. 성별별 분리: female_idx, male_idx

        2. Discriminator 업데이트 (k_d회 반복):
           - G는 frozen (torch.no_grad)
           - Female: perturbed image → DETR feature → D(feat) → CE(pred, 1)
           - Male: original image → DETR feature → D(feat) → CE(pred, 0)
           - d_loss = mean(CE_f, CE_m)

        3. Generator 업데이트 (Female만):
           - δ = G(female_image)
           - perturbed = image + ε·tanh(δ)
           - outputs, features = DETR(perturbed)
           - fairness = -(CE(D(feat), 1) + α·H(D(feat)))
           - det_loss = DETR_criterion(outputs, targets)
           - total_g = fairness + β·det_loss
           - gradient clipping (max_norm=0.1)
```

---

## 1.5 한계 및 다음 단계

| 한계 | 설명 | 해결 방향 |
|------|------|-----------|
| Wasserstein 부재 | Score 분포 직접 정렬 없음 | → WGAN에서 추가 |
| 여성만 교란 | 남성은 원본 → D 학습 불균형 | → WGAN-GD에서 양성 교란 |
| 단순 스케줄링 | Cooldown 없음 → 과적합 위험 | → 7th에서 3-phase 도입 |
