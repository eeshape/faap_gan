# Adaptive Score-based Contrastive Learning (v2)

`train_faap_score_contrastive.py` 분석 문서

---

## 핵심 아이디어

**배치 내 상대적 ranking**을 사용한 Score-based Contrastive Learning

### v1 문제점
- 고정 threshold (0.5) 사용
- 실제 detection score가 대부분 0.9 이상
- → Anchor 샘플이 없음 → Loss 작동 안 함

### v2 해결책
- 배치 내 상대적 ranking 사용
- 상위 K% = Positive (고성능, 목표)
- 하위 K% = Anchor (저성능, 이동 대상)
- **항상 균형 있는 split 보장**

---

## 주요 컴포넌트

### 1. ProjectionHead

SimCLR 스타일 2-layer MLP로 feature를 contrastive space로 매핑

```
입력: 256 (DETR hidden_dim)
  ↓
Linear(256 → 512) + BatchNorm + ReLU
  ↓
Linear(512 → 128)
  ↓
L2 Normalize
  ↓
출력: 128 (normalized)
```

### 2. AdaptiveScoreContrastiveLoss

```
파라미터:
- temperature: 0.1
- top_k_percent: 0.4 (상위 40% → Positive)
- bottom_k_percent: 0.4 (하위 40% → Anchor)
- min_samples: 2
```

**동작 방식:**
1. 배치 내 detection score 기준 정렬
2. 상위 40% → Positive (고성능 detection)
3. 하위 40% → Anchor (저성능 detection)
4. 중간 20% → 무시 (margin)
5. InfoNCE Loss: Anchor를 Positive 방향으로 당김

### 3. BidirectionalScoreContrastiveLoss (옵션)

양방향 contrastive loss:
- **Anchor → Positive**: weight=1.0 (핵심)
- **Positive 내 clustering**: weight=0.3 (보조)

---

## Loss 구성

```python
total_g = (
    lambda_contrastive * contrastive_weight * loss_contrastive
    + lambda_wass * loss_wasserstein
    + beta * loss_det
)
```

| Loss | Weight | 역할 |
|------|--------|------|
| Contrastive | 1.0 × warmup | 저성능→고성능 feature 정렬 |
| Wasserstein | 0.2 | Female score를 Male 수준으로 |
| Detection | 0.5→0.6 | Detection 성능 유지 |

---

## 하이퍼파라미터

### Perturbation
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| epsilon | 0.05 | 시작값 |
| epsilon_final | 0.10 | peak 값 |
| epsilon_min | 0.09 | cooldown 후 |
| warmup_epochs | 8 | |
| hold_epochs | 6 | |
| cooldown_epochs | 16 | |

### Contrastive
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| temperature | 0.1 | 안정적인 학습 |
| top_k_percent | 0.4 | Positive 비율 |
| bottom_k_percent | 0.4 | Anchor 비율 |
| warmup_epochs | 3 | Loss 서서히 적용 |

### Training
| 파라미터 | 값 |
|----------|-----|
| lr_g | 1e-4 |
| batch_size | 7 |
| epochs | 30 |
| max_norm | 0.1 |

---

## 평가 결과 (epoch 29, test set)

### AP/AR 비교

| 지표 | Baseline | Perturbed | Delta |
|------|----------|-----------|-------|
| Male AP | 0.511 | 0.519 | +0.008 |
| Female AP | 0.404 | 0.414 | +0.009 |
| Male AR | 0.834 | 0.838 | +0.004 |
| Female AR | 0.826 | 0.832 | +0.006 |

### Gap 분석

| Gap | Baseline | Perturbed | 변화 |
|-----|----------|-----------|------|
| AP Gap (M-F) | 0.1063 | 0.1049 | **-0.0014 (-1.3%)** |
| AR Gap (M-F) | 0.0081 | 0.0065 | -0.0016 |

### 결론

- AP Gap이 약간 감소 (0.1063 → 0.1049)
- 양쪽 성별 모두 AP/AR 소폭 상승
- 개선 폭이 크지 않음 (목표: 15% 이상 개선)

---

## 실행 방법

```bash
# 단일 GPU
python train_faap_score_contrastive.py --device cuda

# Bidirectional loss 사용
python train_faap_score_contrastive.py --bidirectional

# Resume
python train_faap_score_contrastive.py --resume faap_outputs/faap_outputs_score_contrastive/checkpoints/epoch_0029.pth
```

---

## 출력 디렉토리

```
faap_outputs/faap_outputs_score_contrastive/
├── config.json
├── dataset_layout.json
├── train_log.jsonl
└── checkpoints/
    ├── epoch_0000.pth
    ├── epoch_0001.pth
    └── ...
```

---

## 로깅 메트릭

| 메트릭 | 설명 |
|--------|------|
| loss_contrastive | Contrastive loss |
| loss_wasserstein | Score alignment loss |
| loss_det | Detection loss |
| n_positive / n_anchor | Positive/Anchor 샘플 수 |
| score_gap | 배치 내 Positive-Anchor score 차이 |
| score_f / score_m | 성별별 평균 score |
| delta_linf / delta_l2 | Perturbation 크기 |
