# DINO-style Self-Distillation for Fair Object Detection

`train_faap_dino_1st.py` 분석 문서

---

## 핵심 아이디어

**DINO (Self-DIstillation with NO labels)** 방식을 성별 공정성에 적용

### 기존 방식의 한계
- Wasserstein/Contrastive: 단순 거리 최소화 → **분포 붕괴(Collapse)** 위험
- 여성/남성 분포를 억지로 맞추면 각 그룹의 다양성 손실

### DINO 논문 핵심 기법

| 기법 | 설명 | 효과 |
|------|------|------|
| **Teacher-Student** | EMA로 천천히 업데이트되는 Teacher | 안정적인 타겟 제공 |
| **Centering** | Teacher 출력에서 running mean 빼기 | Collapse 방지 |
| **Sharpening** | 낮은 Temperature | 확률 분포를 peak로 |
| **Cross-Entropy** | H(p_t, p_s) = -Σ p_t * log(p_s) | Student가 Teacher 모방 |

---

## 본 구현에서의 적용

```
Teacher: 남성 이미지의 detection score 분포 (EMA 업데이트)
Student: 여성 이미지의 detection score 분포 (직접 학습)
```

- **Centering**: 남성 score의 running mean을 빼서 collapse 방지
- **Sharpening**: Teacher에 낮은 temperature (0.04) 적용

---

## 주요 클래스

### 1. DINOHead

```python
class DINOHead(nn.Module):
    # in_dim=1 (score) → hidden=256 → out=128
    # LayerNorm + GELU 사용 (BatchNorm 대신)
    # 최종 L2 normalization
```

### 2. DINOCenter

```python
class DINOCenter:
    # center = m * center + (1 - m) * mean(teacher_output)
    # center_momentum = 0.9
```

### 3. DINOLoss

```python
# Centering → Sharpening → Cross-Entropy
teacher_centered = teacher_out - center
teacher_probs = softmax(teacher_centered / τ_t)  # τ_t = 0.04
student_log_probs = log_softmax(student_out / τ_s)  # τ_s = 0.1
loss = -Σ teacher_probs * student_log_probs
```

### 4. EMATeacher

```python
# teacher = m * teacher + (1 - m) * student
# momentum: 0.996 → 1.0 (cosine schedule)
```

---

## Loss 구성

```python
total_g = (
    lambda_dino * dino_loss          # 1.0
    + lambda_det * det_loss          # 0.5→0.6
    + lambda_entropy * entropy_loss  # 0.1 (다양성 유지)
)

# Optional: Adversarial 추가 가능
if use_discriminator:
    total_g += lambda_adv * adv_loss  # 0.5
```

---

## 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| dino_out_dim | 128 | Projection 출력 차원 |
| teacher_temp | 0.04 | Sharp distribution |
| student_temp | 0.1 | Soft distribution |
| center_momentum | 0.9 | Center EMA |
| ema_momentum | 0.996 → 1.0 | Teacher EMA |
| lambda_dino | 1.0 | DINO loss weight |
| lambda_entropy | 0.1 | 다양성 유지 |

---

## 논문 인용 포인트

> "단순히 거리를 좁히는 Contrastive Learning을 넘어, DINO의 Self-Distillation 프레임워크를 도입하여 여성/남성 detection 분포의 다양성은 유지하면서 (Collapse 방지 via Centering) 공정한 성능 분포만 일치시켰다."

---

## 실행 방법

```bash
# 기본 실행
python train_faap_dino_1st.py

# Discriminator 추가
python train_faap_dino_1st.py --use_discriminator
```
