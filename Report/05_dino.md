# 5. DINO 계열 (Self-Distillation for Fair Detection)

**파일**: `train_faap_dino_1st.py`
**기간**: 2026-01-xx
**핵심**: GAN Discriminator 없이 Teacher-Student Self-Distillation로 성별 detection 분포 정렬

---

## 5.1 파이프라인

```
┌─────────────────────────────────────────────────────────────────────┐
│              DINO Self-Distillation Fairness Pipeline                │
│                                                                     │
│  [Male Image]                      [Female Image]                   │
│       │                                  │                          │
│  Generator(G) ──────────────────── Generator(G)                     │
│       │                                  │                          │
│  Perturbed_m                        Perturbed_f                     │
│       │                                  │                          │
│       ▼                                  ▼                          │
│  ┌──────────────┐                ┌──────────────┐                   │
│  │  Frozen DETR │                │  Frozen DETR │                   │
│  └──────┬───────┘                └──────┬───────┘                   │
│         │                               │                           │
│   Male Scores                     Female Scores                     │
│   (matched)                        (matched)                        │
│         │                               │                           │
│         ▼                               ▼                           │
│  ┌─────────────────┐           ┌─────────────────┐                  │
│  │  Teacher Head   │           │  Student Head   │                  │
│  │  (EMA-updated)  │           │  (직접 학습)     │                  │
│  │  DINOHead       │           │  DINOHead       │                  │
│  │  1→256→256→128  │           │  1→256→256→128  │                  │
│  └────────┬────────┘           └────────┬────────┘                  │
│           │                             │                           │
│   teacher_proj                    student_proj                      │
│   (L2-norm)                        (L2-norm)                        │
│           │                             │                           │
│           ▼ Centering                   │                           │
│   teacher_out − center                  │                           │
│           │                             │                           │
│           ▼ Sharpening (τ_t=0.04)       ▼ Softening (τ_s=0.1)      │
│      p_teacher = softmax(·/τ_t)   log p_student = log_softmax(·/τ_s)│
│           │                             │                           │
│           └──────────┬──────────────────┘                           │
│                      ▼                                              │
│           L_dino = H(p_teacher, p_student)                          │
│                      │                                              │
│  EMA Update:  Teacher ← m·Teacher + (1-m)·Student                  │
│  Center Update: c ← 0.9·c + 0.1·mean(teacher_proj)                 │
│                      │                                              │
│  L = λ_dino·L_dino + λ_det·L_det [+ λ_adv·L_adv]                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5.2 핵심 모듈

### DINOHead (Projection Head)

Detection score를 고차원 특징 공간으로 투영한 뒤 L2-normalize하여 distillation에 사용한다.

```python
class DINOHead(nn.Module):
    # in_dim=1 (score) → hidden=256 → hidden=256 → out=128
    # LayerNorm + GELU (BatchNorm 대신 — batch_size=1 대응)
    # 최종 last_layer: Linear(128, 128, bias=False), orthogonal init
    # Forward: mlp(x) → L2-norm → last_layer → L2-norm

    def forward(x):
        x = mlp(x)                    # 1→256→256→128
        x = F.normalize(x, dim=-1)    # L2 norm (중간)
        x = last_layer(x)             # 128→128
        return F.normalize(x, dim=-1) # L2 norm (최종)
```

- `weight_norm` 미사용 (deepcopy 호환성 문제 회피)
- `BatchNorm` 대신 `LayerNorm` 사용 — 소배치(batch=1)에서도 동작

### DINOCenter (Centering 메커니즘)

Teacher 출력의 running mean을 추적하여 모든 출력이 한 점으로 붕괴하는 것을 방지한다.

```python
class DINOCenter:
    # center = m * center + (1 - m) * mean(teacher_output)
    # center_momentum = 0.9

    def update(teacher_output):
        batch_center = teacher_output.mean(dim=0)
        center = 0.9 * center + 0.1 * batch_center

    def apply(teacher_output):
        return teacher_output - center  # Centering 적용
```

### EMATeacher (EMA Teacher 업데이트)

Student 파라미터의 Exponential Moving Average로 Teacher를 유지한다. Teacher는 Student보다 천천히 변화하여 안정적인 distillation 타겟을 제공한다.

```python
class EMATeacher:
    # teacher = m * teacher + (1 - m) * student
    # momentum: 0.996 → 1.0 (cosine schedule)
    # deepcopy 미사용 → state_dict 복사 방식 (weight_norm 호환)

    def update(student):
        for name in teacher_params:
            teacher[name] = m * teacher[name] + (1-m) * student[name]
```

---

## 5.3 Loss 수식

### DINO Distillation Loss

Teacher(남성 분포)를 sharp target으로, Student(여성 분포)가 이를 모방하도록 학습하는 cross-entropy loss.

$$p_t = \text{softmax}\!\left(\frac{f_{\theta_t}(s_m) - c}{\tau_t}\right), \quad \tau_t < \tau_s$$

$$p_s = \text{softmax}\!\left(\frac{f_{\theta_s}(s_f)}{\tau_s}\right)$$

$$L_{dino} = H(p_t,\, p_s) = -\sum_i p_t^{(i)} \log p_s^{(i)}$$

여기서 $c$는 DINOCenter의 running mean, $\tau_t = 0.04$ (sharp), $\tau_s = 0.1$ (soft).

### Detection Loss (DETR 표준)

Generator가 perturbation을 추가하더라도 기존 detection 성능이 유지되도록 보존하는 손실.

$$L_{det} = \frac{1}{2}\left(L_{det}^{female} + L_{det}^{male}\right)$$

$$L_{det}^{*} = \lambda_{ce} \cdot L_{ce} + \lambda_{bbox} \cdot L_{L1} + \lambda_{giou} \cdot L_{giou}$$

### Adversarial Loss (선택적)

`--use_discriminator` 플래그 사용 시 GenderDiscriminator를 추가로 활용한다.

$$L_{adv} = \text{CrossEntropy}\!\left(D(f_f),\, 0\right) \quad \text{(여성 feature를 남성으로 fool)}$$

### Total Generator Loss

$$\boxed{L = \lambda_{dino} \cdot L_{dino} + \lambda_{det} \cdot L_{det} + \lambda_{adv} \cdot L_{adv}}$$

기본 구성 (`--use_discriminator` 미사용):

$$L = 1.0 \cdot L_{dino} + \lambda_{det} \cdot L_{det} + 0.1 \cdot L_{entropy}$$

여기서 $L_{entropy} = -H(p_s)$는 Student 출력의 다양성을 유지하는 정규화 항이다 (entropy 최대화).

---

## 5.4 하이퍼파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `epochs` | 30 | 총 학습 에폭 |
| `batch_size` | 4 | 배치 크기 (논문 버전; L40S에서는 6) |
| `epsilon` | 0.05 | warmup 시작값 |
| `epsilon_final` | 0.10 | warmup 최대값 (peak) |
| `epsilon_min` | 0.08 | cooldown 최소값 |
| `epsilon_warmup_epochs` | 8 | Phase 1: 0.05→0.10 |
| `epsilon_hold_epochs` | 8 | Phase 2: 0.10 유지 |
| `epsilon_cooldown_epochs` | 14 | Phase 3: 0.10→0.08 |
| `dino_out_dim` | 128 | Projection 출력 차원 |
| `dino_hidden_dim` | 256 | Projection 은닉 차원 |
| `teacher_temp` | 0.04 | Teacher temperature (sharp) |
| `student_temp` | 0.1 | Student temperature (soft) |
| `center_momentum` | 0.9 | DINOCenter EMA 계수 |
| `ema_momentum` | 0.996 → 1.0 | Teacher EMA (cosine schedule) |
| `lambda_dino` | 1.0 | DINO distillation loss 가중치 |
| `lambda_det` | 0.5 → 0.6 | Detection loss 가중치 (선형 증가) |
| `lambda_entropy` | 0.1 | Entropy 정규화 가중치 |
| `lambda_adv` | 0.5 | Adversarial loss 가중치 (선택) |
| `lr_g` | 1e-4 | Generator + Student head 학습률 |
| `max_norm` | 0.1 | Gradient clipping |

---

## 5.5 실험 결과

**테스트 조건**: `train_faap_dino_1st.py`, epoch 30

| 메트릭 | Baseline Gap | DINO Gap | 개선율 |
|--------|:---:|:---:|:---:|
| **AR Gap** (M−F) | 0.0081 | **0.0059** | **−27%** |
| **AP Gap** (M−F) | 0.106 | 0.106 | 0% (변화 없음) |

### 해석

- **AR Gap 27% 개선**: EMA Teacher의 안정적 타겟 덕분에 여성 Recall이 완만하게 향상됨
- **AP Gap 무변화**: Detection precision 개선으로는 이어지지 못함 — score 수준의 정렬만으로는 localization 격차 해소 불충분
- 7th (AR Gap −60.6%) 및 Contrastive 1st (AR Gap −61%)에 비해 개선 폭이 제한적

---

## 5.6 설계 근거

### EMA Teacher가 GAN Discriminator보다 안정적인 이유

GAN의 Discriminator는 Generator와 번갈아 학습하므로 타겟이 지속적으로 변화한다. 반면 EMA Teacher는 Student 파라미터의 이동평균이므로 변화가 매우 완만하다. 이로 인해 다음 효과가 발생한다.

1. **Mode collapse 없음**: Discriminator-Generator의 min-max 경쟁이 없어 학습이 발산하지 않는다.
2. **안정적 gradient**: Teacher output이 급변하지 않으므로 distillation gradient가 일관성을 유지한다.
3. **k_d 불필요**: GAN에서 필요한 Discriminator 선행 학습 단계(`k_d=4`)가 없어 학습 루프가 단순하다.

### Centering이 Feature Collapse를 방지하는 원리

Centering 없이 Teacher temperature만 낮추면 모든 예측이 동일한 하나의 차원으로 집중될 수 있다 (uniform collapse). Center를 빼줌으로써 Teacher output이 항상 mean=0 근방에 머물도록 강제하여, Student가 특정 차원만 모방하는 퇴화를 방지한다.

$$\text{Collapse 방지}: \quad f_{\theta_t}(x) - \mathbb{E}[f_{\theta_t}] \approx 0 \text{ 을 피함}$$

---

## 5.7 한계

1. **개선 폭 제한**: AR Gap 27% 개선으로, 7th (−60.6%) 및 Contrastive 1st (−61%)에 크게 미치지 못한다.
2. **AP Gap 미개선**: Score 수준의 distillation은 localization 품질 격차(AP 기준)를 해소하지 못한다. Contrastive IoU 버전처럼 IoU 인지 신호가 필요하다.
3. **Score 추상화의 한계**: Detection score를 scalar(1-dim)로만 다루기 때문에 DETR feature의 풍부한 공간 정보를 활용하지 못한다. Contrastive 계열처럼 256-dim feature를 직접 정렬하면 더 강한 신호를 줄 수 있다.
4. **Entropy 정규화 상충**: $L_{entropy}$가 Student의 분산을 유지하는 동시에 $L_{dino}$의 Teacher 모방을 방해하는 방향으로 작용할 수 있다.
