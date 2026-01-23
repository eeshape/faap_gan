# FAAP 14th: AP Gap Direct Optimization

`train_faap_wgan_GD_14th.py` 분석 문서

---

## 배경: 7th 분석 결과

| 지표 | Before | After | 변화 |
|------|--------|-------|------|
| AR Gap | 0.0081 | 0.0032 | **60% 개선** ✅ |
| AP Gap | 0.1063 | 0.1059 | 거의 변화 없음 ❌ |
| Female AP | 0.4045 | 0.4078 | +0.0034 ✅ |

### Contrastive IoU 실패 원인
- IoU gap이 원래 없었음 (0.766 vs 0.770)
- Feature alignment가 AP에 직접 영향 못 줌
- Projection head가 평가 시 미사용

---

## 14th 핵심 전략: AP Gap 직접 공략

### 새로운 Loss Functions

#### 1. Quantile-Weighted Wasserstein

```python
def _quantile_weighted_wasserstein(female_scores, male_scores, quantile_focus=0.7):
    """
    상위 분위에 가중치를 부여
    - 상위 30% (position >= 0.7): 2배 가중치
    - 하위 70%: 0.5배 가중치
    """
```

**이유**: AP는 high-confidence prediction의 precision에 민감

#### 2. Confidence Margin Loss

```python
def _confidence_margin_loss(female_scores, male_scores, margin=0.05):
    """
    Female mean ≥ Male mean - margin 이면 loss = 0
    그렇지 않으면 차이에 비례한 패널티
    """
```

#### 3. Direct Score Gap Penalty

```python
def _direct_score_gap_penalty(female_scores, male_scores):
    """
    gap = relu(mean_m - mean_f)
    가장 단순하지만 직접적인 방법
    """
```

#### 4. Detection Loss 비대칭

```python
det_loss = det_f_scale * det_f + det_m  # det_f_scale = 1.2
```

Female detection loss에 더 높은 가중치

---

## Loss 구성

```python
total_g = (
    lambda_fair * fairness_loss         # 2.0 (adversarial)
    + beta * det_loss                   # 0.5→0.6
    + lambda_w * wasserstein_loss       # 0.3 (7th: 0.2)
    + lambda_quantile_w * quantile_w_loss  # 0.2 (NEW)
    + lambda_margin * margin_loss       # 0.1 (NEW)
    + lambda_gap * gap_loss             # 0.15 (NEW)
)
```

---

## 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| lambda_w | 0.3 | 기본 Wasserstein (7th: 0.2) |
| lambda_quantile_w | 0.2 | Quantile-weighted (NEW) |
| lambda_margin | 0.1 | Confidence margin (NEW) |
| lambda_gap | 0.15 | Direct gap penalty (NEW) |
| quantile_focus | 0.7 | 상위 30%에 집중 |
| margin_target | 0.05 | Female confidence 마진 |
| det_f_scale | 1.2 | Female detection 가중치 |

---

## 7th 대비 변경점

| 항목 | 7th | 14th |
|------|-----|------|
| Wasserstein | 0.2 | 0.3 |
| Quantile-W | - | 0.2 (NEW) |
| Margin Loss | - | 0.1 (NEW) |
| Gap Loss | - | 0.15 (NEW) |
| Female Det Scale | 1.0 | 1.2 |

---

## 목표

| 지표 | 현재 | 목표 |
|------|------|------|
| AP Gap | 0.106 | < 0.08 (25% 개선) |
| Female AP | 0.404 | > 0.42 (4% 향상) |
| Male AP | 유지 | 유지 또는 소폭 상승 |

---

## 실행 방법

```bash
python train_faap_wgan_GD_14th.py

# 커스텀 가중치
python train_faap_wgan_GD_14th.py \
    --lambda_quantile_w 0.3 \
    --lambda_gap 0.2 \
    --det_f_scale 1.3
```
