# train_faap_wgan_GD_9th 변경 요약

## 📋 목차
1. [8th 실험 문제점 분석](#8th-실험-문제점-분석)
2. [9th 설계 철학](#9th-설계-철학)
3. [핵심 개선 사항](#핵심-개선-사항)
4. [수학적 정리](#수학적-정리)
5. [하이퍼파라미터 비교](#하이퍼파라미터-비교)
6. [기대 효과](#기대-효과)
7. [실행 방법](#실행-방법)
8. [로그 분석 가이드](#로그-분석-가이드)

---

## 8th 실험 문제점 분석

### 8th의 접근 방식
```python
# 8th: 단순 Step 방식의 lambda_w 부스트
if epoch >= 16:
    lambda_w = 0.3  # 갑작스러운 증가
else:
    lambda_w = 0.2
```

### 문제점 진단

| 문제 | 원인 | 영향 |
|------|------|------|
| **갑작스러운 손실 변화** | Step 방식 (0.2→0.3) | 기존 학습 패턴 붕괴 |
| **후반부 불안정성** | 이미 수렴 중인 G에 강한 신호 | Detection 성능 하락 |
| **AP Gap만 집중** | Wasserstein만 강화 | 7th의 AR Gap 성과 손실 가능 |
| **단일 메커니즘** | 분포 정렬만 사용 | AP 개선에 비효율적 |

### 7th vs 8th 결과 비교 (예상)

| 지표 | 7th | 8th (예상) | 문제점 |
|------|-----|------------|--------|
| Female AP Delta | +0.0034 | ±0.001 | Wasserstein 강화로 불안정 |
| Male AP Delta | +0.0029 | -0.002 | 남성 보호 약화 |
| AR Gap 감소율 | 60% | 30-40% | 후반부 불안정으로 회귀 |
| AP Gap 감소율 | 0.4% | 1-2% | 목표 달성하나 AR 희생 |

---

## 9th 설계 철학

> **"7th의 모든 성공 요소를 유지하면서, 다중 Score Alignment 메커니즘 + 후반부 안정화"**

### 핵심 원칙
1. ✅ **7th 구조 완전 유지**: 모든 검증된 요소 보존
2. ✅ **점진적 변화**: 갑작스러운 손실 변화 없음
3. ✅ **다중 메커니즘**: AP와 AR 동시 개선
4. ✅ **후반부 안정화**: LR decay로 수렴 보장

---

## 핵심 개선 사항

### 1. Quantile Matching Loss (AP Gap 집중 개선)

#### 개념
Wasserstein은 전체 분포를 정렬하지만, Quantile Matching은 **특정 분위수**에서 차이를 최소화합니다.

```python
def _quantile_matching_loss(female_scores, male_scores, num_quantiles=5):
    """
    분위수 레벨: [0.1, 0.3, 0.5, 0.7, 0.9] (num_quantiles=5)
    
    AP 계산에서 높은 confidence의 detection이 중요하므로,
    높은 분위수(0.7, 0.9)에 더 높은 가중치 부여
    """
    quantile_levels = [0.167, 0.333, 0.5, 0.667, 0.833]  # num_quantiles=5
    
    q_female = torch.quantile(female_scores, quantile_levels)
    q_male = torch.quantile(male_scores.detach(), quantile_levels)
    
    # 단방향: 여성 분위수가 남성보다 낮을 때만 패널티
    # 가중치: 높은 분위수에 높은 가중치 (AP 개선에 효과적)
    weights = quantile_levels  # [0.167, 0.333, 0.5, 0.667, 0.833]
    
    return (weights * F.relu(q_male - q_female)).mean()
```

#### Wasserstein vs Quantile Matching

| 특성 | Wasserstein | Quantile Matching |
|------|-------------|-------------------|
| **정렬 방식** | 전체 분포 정렬 | 특정 분위수 정렬 |
| **계산 비용** | O(n log n) 정렬 필요 | O(n) quantile 계산 |
| **해석 가능성** | 분포 거리 | 분위수별 차이 |
| **AP 개선** | 간접적 | 직접적 (높은 분위수 집중) |
| **AR 개선** | 효과적 | 보통 |

#### 왜 둘 다 사용하나?
- **Wasserstein**: AR 개선에 효과적 (7th 성공 요소)
- **Quantile**: AP 개선에 효과적 (9th 목표)
- **상호 보완**: 두 메커니즘이 서로 다른 측면 개선

### 2. Score Gap Penalty (직접적 평균 차이 감소)

```python
def _score_gap_penalty(female_scores, male_scores):
    """
    가장 단순하고 직접적인 손실:
    여성 평균 score와 남성 평균 score의 차이를 줄임
    
    단방향: 여성이 낮을 때만 패널티
    """
    mean_f = female_scores.mean()
    mean_m = male_scores.detach().mean()
    
    return F.relu(mean_m - mean_f)
```

#### 역할
- **보조 신호**: Wasserstein/Quantile이 분포 형태를 정렬하는 동안, Gap Penalty는 단순히 평균 차이 감소
- **안정적**: 계산이 단순하고 그래디언트가 안정적
- **낮은 가중치**: `lambda_gap=0.1`로 다른 손실의 보조 역할

### 3. Learning Rate Decay (후반부 안정화)

```python
# Epoch 18부터 LR 50% 감소 (한 번만 적용)
if epoch >= lr_decay_epoch and not lr_decayed:
    for param_group in opt_g.param_groups:
        param_group['lr'] *= 0.5  # 1e-4 → 5e-5
    for param_group in opt_d.param_groups:
        param_group['lr'] *= 0.5
    lr_decayed = True
```

#### 왜 필요한가?
- **8th 문제**: 후반부에 손실 가중치만 바꾸면 불안정
- **9th 해결**: LR 감소로 후반부 미세 조정 모드 전환
- **검증된 기법**: 대부분의 딥러닝 학습에서 사용되는 표준 기법

#### 스케줄 비교
```
8th: epsilon cooldown + lambda_w 갑작스러운 증가 → 불안정
9th: epsilon cooldown + LR decay → 점진적 수렴
```

### 4. 확장된 학습 (28 epochs)

| 버전 | Epochs | 이유 |
|------|--------|------|
| 7th | 24 | 기본 학습 |
| 8th | 24 | 7th와 동일 |
| 9th | **28** | 새로운 손실 함수 학습 시간 + LR decay 후 수렴 시간 |

#### Epsilon 스케줄 조정
```
7th: warmup(8) + hold(6) + cooldown(10) = 24 epochs
9th: warmup(8) + hold(8) + cooldown(12) = 28 epochs

epsilon_min: 0.09 → 0.08 (더 낮은 최종 perturbation)
```

---

## 수학적 정리

### Generator Loss (L_G)

```
L_G = λ_fair × L_fair + β(t) × L_det + λ_w × L_W + λ_q × L_Q + λ_gap × L_gap

where:
  L_fair = fair_f_scale × L_f + fair_m_scale × L_m   # 비대칭 (1.0:0.5)
  L_det  = det_loss_f + det_loss_m                    # detection
  L_W    = mean(ReLU(sorted_m - sorted_f))            # Wasserstein (7th)
  L_Q    = mean(w_q × ReLU(quantile_m - quantile_f))  # Quantile (9th 신규)
  L_gap  = ReLU(mean_m - mean_f)                      # Gap Penalty (9th 신규)
  
  β(t)   = 0.5 + 0.15 × (t/T)                         # 0.5 → 0.65
  ε(t)   = warmup → hold → cooldown (0.05→0.10→0.08)
  lr(t)  = 1e-4 (t < 18) else 5e-5                    # LR decay
```

### Score Alignment 손실 조합

```
L_score = λ_w × L_W + λ_q × L_Q + λ_gap × L_gap
        = 0.2 × L_W + 0.15 × L_Q + 0.1 × L_gap

역할 분담:
- L_W (0.2): AR Gap 개선 (전체 분포 정렬)
- L_Q (0.15): AP Gap 개선 (높은 분위수 집중)
- L_gap (0.1): 평균 차이 직접 감소 (보조)
```

---

## 하이퍼파라미터 비교

| 파라미터 | 7th | 8th | 9th | 변경 이유 |
|----------|-----|-----|-----|-----------|
| `epochs` | 24 | 24 | **28** | 새 손실 학습 + 수렴 시간 |
| `epsilon_hold` | 6 | 8 | **8** | 8th 유지 |
| `epsilon_cooldown` | 10 | 8 | **12** | 더 긴 cooldown |
| `epsilon_min` | 0.09 | 0.09 | **0.08** | 최종 perturbation 감소 |
| `beta_final` | 0.6 | 0.6 | **0.65** | detection 보호 강화 |
| `lambda_w` | 0.2 | 0.2→0.3 | **0.2** | 고정 (안정성) |
| `lambda_q` | - | - | **0.15** | 신규: Quantile |
| `lambda_gap` | - | - | **0.1** | 신규: Gap Penalty |
| `lr_decay_epoch` | - | - | **18** | 신규: LR decay 시점 |
| `lr_decay_factor` | - | - | **0.5** | 신규: 50% 감소 |

### 손실 가중치 총합 비교

```
7th: λ_fair(2.0) + β(0.5~0.6) + λ_w(0.2) = 2.7~2.8
8th: λ_fair(2.0) + β(0.5~0.6) + λ_w(0.2~0.3) = 2.7~2.9
9th: λ_fair(2.0) + β(0.5~0.65) + λ_w(0.2) + λ_q(0.15) + λ_gap(0.1) = 2.95~3.1

→ 9th가 약간 높지만, LR decay로 후반부에 균형 맞춤
```

---

## 기대 효과

### 정량적 목표

| 지표 | Baseline | 7th | 9th 목표 |
|------|----------|-----|----------|
| Female AP | 0.404 | 0.408 (+0.0034) | **0.412+** (+0.008) |
| Female AR | 0.826 | 0.833 (+0.0070) | **0.835+** (+0.009) |
| Male AP | 0.511 | 0.514 (+0.0029) | 0.514+ |
| Male AR | 0.834 | 0.836 (+0.0021) | 0.836+ |
| AP Gap | 0.1063 | 0.1059 (-0.4%) | **< 0.102** (-4%) |
| AR Gap | 0.0081 | 0.0032 (-60%) | **< 0.002** (-75%) |

### 개선 메커니즘 분석

```
                    AR Gap 개선          AP Gap 개선
                         ↑                    ↑
                    ┌────┴────┐          ┌────┴────┐
                    │         │          │         │
              Wasserstein   Quantile   Quantile  Gap Penalty
               (L_W)         (L_Q)      (L_Q)     (L_gap)
                    │         │          │         │
                    └────┬────┘          └────┬────┘
                         │                    │
                    전체 분포               높은 분위수
                     정렬                    집중
```

### 학습 단계별 목표

| 단계 | Epochs | 목표 | 주요 메커니즘 |
|------|--------|------|---------------|
| **Warmup** | 0-7 | 기본 학습 | epsilon↑, 모든 손실 활성화 |
| **Hold** | 8-15 | 공정성 학습 | epsilon 최대, 분포 정렬 |
| **Pre-decay** | 16-17 | 최적화 진행 | 손실 균형 |
| **Post-decay** | 18-27 | 미세 조정 | LR↓, epsilon↓, 수렴 |

---

## 실행 방법

### 기본 실행
```bash
cd /home/dohyeong/Desktop/faap_gan
python train_faap_wgan_GD_9th.py
```

### 단일 GPU 지정
```bash
cd /home/dohyeong/Desktop/faap_gan
CUDA_VISIBLE_DEVICES=0 python train_faap_wgan_GD_9th.py --batch_size 8
```

### 분산 학습 (DDP)
```bash
cd /home/dohyeong/Desktop
torchrun --nproc_per_node=2 --master_port=29500 \
  -m faap_gan.train_faap_wgan_GD_9th \
  --distributed
```

### 하이퍼파라미터 실험

```bash
# Quantile 가중치 조정
python train_faap_wgan_GD_9th.py --lambda_q 0.2

# Gap Penalty 강화
python train_faap_wgan_GD_9th.py --lambda_gap 0.15

# LR decay 시점 변경
python train_faap_wgan_GD_9th.py --lr_decay_epoch 16

# 복합 변경
python train_faap_wgan_GD_9th.py \
  --lambda_q 0.2 \
  --lambda_gap 0.12 \
  --lr_decay_epoch 20 \
  --epochs 32
```

### 평가
```bash
python eval_faap.py \
  --dataset_root /home/dohyeong/Desktop/faap_dataset \
  --detr_checkpoint /home/dohyeong/Desktop/detr/detr-r50-e632da11.pth \
  --generator_checkpoint faap_outputs/faap_outputs_gd_9th/checkpoints/epoch_0027.pth \
  --epsilon 0.08 \
  --split test \
  --results_path faap_outputs/faap_outputs_gd_9th/test_metrics_epoch_0027.json
```

---

## 로그 분석 가이드

### 핵심 모니터링 지표

```jsonl
{
  "epoch": 27,
  "epsilon": 0.08,           // cooldown 완료
  "beta": 0.65,              // schedule 완료
  "lr_g": 5e-05,             // decay 적용됨 (epoch >= 18)
  "lr_d": 5e-05,
  "g_w": 0.015,              // Wasserstein loss
  "g_q": 0.008,              // Quantile loss (9th 신규)
  "g_gap": 0.005,            // Gap penalty (9th 신규)
  "obj_score_f": 0.155,      // 여성 detection score
  "obj_score_m": 0.158,      // 남성 detection score
  "fair_f_scale": 1.0,
  "fair_m_scale": 0.5
}
```

### 성공 신호 ✅

1. **Score 정렬 진행**
   - `g_w`, `g_q`, `g_gap` 모두 감소 추세
   - `obj_score_f` ≈ `obj_score_m` (차이 < 0.01)

2. **학습 안정성**
   - `g_total` 변동폭 감소 (특히 epoch 18 이후)
   - `d_loss` 0.5-0.7 범위 유지

3. **Detection 보존**
   - `obj_score` 0.14-0.17 범위 유지
   - `obj_frac` 변화 < 5%

### 문제 신호 ⚠️

| 신호 | 진단 | 해결책 |
|------|------|--------|
| `g_q`가 epoch 18 이후 증가 | Quantile 가중치 너무 높음 | `--lambda_q 0.1` |
| `obj_score` 급격히 하락 | 총 손실 가중치 과다 | `--lambda_gap 0.05` |
| `d_loss` < 0.3 | D가 G를 완전히 이김 | `--lr_d 5e-5` |
| `d_loss` > 0.9 | G가 D를 완전히 이김 | `--k_d 5` |
| Male AP 하락 | 남성 보호 부족 | `--fair_m_scale 0.6` |

### 체크포인트 선택 가이드

```
epoch 0-7:   warmup 단계, 평가하지 않음
epoch 8-15:  공정성 학습 중, 중간 평가 가능
epoch 16-17: LR decay 직전, 비교용
epoch 18-23: LR decay 후 안정화 단계
epoch 24-27: 최종 수렴, 최적 체크포인트 후보 ★
```

---

## 이론적 배경

### Quantile Matching의 수학적 의미

분위수 함수 $Q_X(p)$는 누적분포함수의 역함수:
$$Q_X(p) = \inf\{x : F_X(x) \geq p\}$$

Quantile Matching Loss:
$$L_Q = \sum_{i=1}^{k} w_i \cdot \max(0, Q_M(p_i) - Q_F(p_i))$$

여기서:
- $p_i$: 분위수 레벨 (예: 0.1, 0.3, 0.5, 0.7, 0.9)
- $w_i = p_i$: 높은 분위수에 높은 가중치
- $Q_M, Q_F$: 남성/여성 score 분위수

### AP 개선에 효과적인 이유

AP (Average Precision)는 Precision-Recall 곡선 아래 면적:
$$AP = \int_0^1 P(R) dR$$

높은 confidence의 detection이 PR 곡선 초반부를 결정하므로,
높은 분위수(0.7, 0.9)의 score를 정렬하면 AP가 직접적으로 개선됨.

### 단방향 손실의 게임 이론적 해석

```
목표: 여성 성능 ↑, 남성 성능 유지
      (Pareto 개선)

양방향 손실: |f - m| → 여성↑ OR 남성↓ (어느 쪽이든 최소화)
단방향 손실: max(0, m - f) → 여성↑만 허용 (남성 보호)
```

---

## Ablation Study 제안

9th 실험 후 추가 분석을 위한 실험:

| 실험 | 변경 | 목적 |
|------|------|------|
| 9th-A | `lambda_q=0` | Quantile 효과 측정 |
| 9th-B | `lambda_gap=0` | Gap Penalty 효과 측정 |
| 9th-C | `lr_decay=False` | LR decay 효과 측정 |
| 9th-D | `epochs=24` | 확장 학습 효과 측정 |

```bash
# 예: Quantile 없이 실행
python train_faap_wgan_GD_9th.py --lambda_q 0 --output_dir faap_outputs/faap_outputs_gd_9th_ablation_A
```

---

## 요약

### 9th의 핵심 기여

1. **다중 Score Alignment**: Wasserstein + Quantile + Gap Penalty
2. **AP Gap 직접 공략**: 높은 분위수에 가중치 부여
3. **후반부 안정화**: LR decay로 수렴 보장
4. **7th 성공 요소 100% 유지**: 비대칭 fairness, epsilon 스케줄, 단방향 손실

### 8th → 9th 개선 포인트

| 8th 문제점 | 9th 해결책 |
|------------|------------|
| 갑작스러운 lambda_w 증가 | 고정 lambda_w + 추가 손실 |
| 단일 메커니즘 | 다중 메커니즘 (W + Q + Gap) |
| 후반부 불안정 | LR decay |
| AP만 집중 | AP + AR 동시 개선 |
