# FAAP-GAN 연구 히스토리 종합 보고서

**작성일**: 2026-02-16
**대상 기간**: 2025-11-24 ~ 2026-01-31 (약 2개월)
**총 실험 수**: 30+ 버전 (WGAN-GD 14종, MMD 2종, Contrastive 3종, DINO 1종, SimCLR InfoNCE 8종+)

---

## 목차

1. [연구 개요 및 배경](#1-연구-개요-및-배경)
2. [Phase 1: 초기 FAAP + WGAN (2025년 11월)](#2-phase-1-초기-faap--wgan-2025년-11월)
3. [Phase 2: WGAN-GD 3rd~8th + MMD (2025년 12월)](#3-phase-2-wgan-gd-3rd8th--mmd-2025년-12월)
4. [Phase 3: WGAN-GD 9th~14th + Contrastive + DINO (2026년 1월 초~중순)](#4-phase-3-wgan-gd-9th14th--contrastive--dino-2026년-1월-초중순)
5. [Phase 4: SimCLR InfoNCE + Score-based Contrastive (2026년 1월 하순)](#5-phase-4-simclr-infonce--score-based-contrastive-2026년-1월-하순)
6. [전체 실험 성능 비교표](#6-전체-실험-성능-비교표)
7. [방법론 진화 흐름도](#7-방법론-진화-흐름도)
8. [핵심 인사이트 및 교훈](#8-핵심-인사이트-및-교훈)
9. [현재 상태 및 향후 방향](#9-현재-상태-및-향후-방향)

---

## 1. 연구 개요 및 배경

### 1.1 연구 목표

동결된(Frozen) DETR 객체 검출기의 **성별 간 검출 성능(AP/AR) 격차를 줄이는 것**이 핵심 목표이다. 모델 가중치를 변경하지 않고, 입력 이미지에 미세한 perturbation을 생성하는 Generator(G)를 학습하여 공정성을 개선하는 **FAAP(Fairness-Aware Adversarial Perturbation)** 방식을 사용한다.

### 1.2 문제 배경

- DETR이 남성 이미지에서 높은 AP/AR, 여성 이미지에서 낮은 성능을 보이는 **성별 편향** 존재
- 배포 환경에서 DETR 가중치를 변경하지 않고도 공정성을 개선할 수 있는 **실용적 접근법** 필요
- **Baseline 성능**: Male AP 0.511, Female AP 0.404 → **AP Gap 0.106**, AR Gap 0.008

### 1.3 공통 아키텍처

| 구성 요소 | 설명 |
|-----------|------|
| **Frozen DETR** | ResNet-50 백본, 사전학습 가중치 동결. decoder feature(`hs[-1]`, B×100×256) 노출 |
| **PerturbationGenerator (G)** | 경량 U-Net. `delta = epsilon * tanh(out)`. ImageNet 정규화 범위 유지 |
| **GenderDiscriminator (D)** | DETR decoder feature → MLP(256→256→256→2). Phase 3 이후 점차 제거 |
| **데이터** | COCO 포맷, `women_split/men_split`으로 성별 분리. WeightedRandomSampler 1:1 균형 |

---

## 2. Phase 1: 초기 FAAP + WGAN (2025년 11월)

> **기간**: 2025-11-24 ~ 2025-11-30 (7일)
> **핵심**: 기본 Adversarial Fairness에서 Wasserstein 정렬 + 양성 교란까지 빠르게 진화

### 2.1 실험 타임라인

| 날짜 | 버전 | 핵심 변경 |
|------|------|-----------|
| 11/24 | `train_faap_2nd.py` | 최초 구현. 여성만 perturbation, CE+Entropy fairness loss |
| 11/24 | `train_faap.py` | epsilon 워밍업(0.05→0.12), DDP 지원, 상세 모니터링 추가 |
| 11/24 | `train_faap_wgan.py` | **Wasserstein 검출 점수 정렬** 도입. 1D Wasserstein distance로 여/남 score 분포 직접 정렬 |
| 11/28 | `train_faap_wgan_GD.py` | **남녀 모두 G 적용** (양성 교란). D도 양쪽 교란본으로 학습. batch 8→4 |
| 11/30 | `train_faap_wgan_GD_2nd.py` | 공정성 가중치 대폭 강화: `lambda_fair=5.0`, `beta=0.3`, `lambda_w=0.1` |

### 2.2 손실 함수 진화

```
[2nd] 기본: L_G = -(CE + α·H) + β·L_det
  ↓ Wasserstein 추가
[WGAN] L_G = -(CE + α·H) + β·L_det + λ_w·W(s_f, s_m)
  ↓ 남녀 모두 G 적용
[GD 1st] L_G = Σ_{g∈{f,m}} -(CE_g + α·H_g) + β·Σ L_det_g + λ_w·W
  ↓ 가중치 재조정
[GD 2nd] L_G = λ_fair·(Σ fairness) + β·(Σ L_det) + λ_w·W    (λ_fair=5.0)
```

### 2.3 Phase 1 핵심 성과

- 11/24 하루에 3개 버전 생성 → **매우 빠른 실험 반복**
- Wasserstein 검출 점수 정렬이라는 핵심 아이디어 확립
- 남녀 모두 G 적용하는 양성 교란 패러다임 도입
- epsilon 워밍업, DDP, 상세 모니터링 등 인프라 구축

---

## 3. Phase 2: WGAN-GD 3rd~8th + MMD (2025년 12월)

> **기간**: 2025-12-17 ~ 2025-12-28 (약 2주)
> **핵심**: 단방향 Wasserstein + 비대칭 fairness로 **7th 국내학회 제출 버전** 탄생

### 3.1 버전별 핵심 변경과 결과

| 버전 | 핵심 변경 | Male AP Δ | Female AP Δ | Female AR Δ | 평가 |
|------|-----------|-----------|-------------|-------------|------|
| **3rd** | 양방향 Wasserstein, k_d=4 | -0.0149 | -0.0160 | -0.0045 | **나쁨** |
| **4th** | **단방향 Wasserstein** `ReLU(M-F)`, Male detach | -0.0076 | -0.0078 | **+0.0044** | 전환점 |
| **5th** | per-gender 데이터 제한 (4500장) | -0.0068 | -0.0087 | +0.0012 | 보통 |
| **6th** | epsilon cooldown + beta 0.5→0.7 | -0.0064 | -0.0093 | -0.0001 | 보통 |
| **6th_2** | stress window (epsilon/fair 1.3~2.0배) | 악화 | 악화 | 악화 | **나쁨** |
| **7th** | **fair_m_scale=0.5**, 24ep, beta 0.5→0.6 | **+0.0029** | **+0.0034** | **+0.0070** | **최고** |
| **8th** | 후반부 lambda_w boost (0.2→0.3) | - | - | - | 7th 미세 변형 |

### 3.2 MMD 접근법

| 버전 | 핵심 아이디어 | 결과 |
|------|---------------|------|
| MMD 1st | Discriminator-free, Gaussian Kernel MMD (양방향) | 7th 미달 |
| MMD 2nd | 비대칭 MMD (Male detach) + beta=1.5 | 7th 미달 |

### 3.3 7th 버전 상세 (국내학회 제출)

**핵심 성공 요인**:
1. **단방향 Wasserstein**: `ReLU(sorted_m - sorted_f)` — Female만 끌어올리고 Male은 고정(detach)
2. **비대칭 fairness**: `fair_f_scale=1.0, fair_m_scale=0.5` — 남성 perturbation 억제
3. **완만한 스케줄링**: epsilon 0.05→0.10→0.09 cooldown, beta 0.5→0.6
4. **충분한 학습**: 24 epoch (이전 12 epoch 대비 2배)

**실험 결과 (Epoch 23)**:

| 지표 | Baseline | Perturbed | Delta |
|------|----------|-----------|-------|
| Male AP | 0.5108 | 0.5137 | +0.0029 |
| Female AP | 0.4045 | 0.4078 | +0.0034 |
| Male AR | 0.8339 | 0.8359 | +0.0021 |
| Female AR | 0.8258 | 0.8328 | +0.0070 |
| **AR Gap** | **0.0081** | **0.0032** | **-60.5% 개선** |
| AP Gap | 0.1063 | 0.1059 | -0.38% 개선 |

### 3.4 Phase 2 핵심 교훈

- **성공**: 단방향/비대칭 원칙, 완만한 스케줄링, 충분한 학습 시간
- **실패**: 양방향 Wasserstein(3rd), 데이터 제한(5th), 과도한 beta(6th), stress window(6th_2)
- **원칙 확립**: "약자 그룹(Female)만 끌어올리고, 강자 그룹(Male)은 보호한다"

---

## 4. Phase 3: WGAN-GD 9th~14th + Contrastive + DINO (2026년 1월 초~중순)

> **기간**: 2026-01-03 ~ 2026-01-20 (약 3주)
> **핵심**: AP Gap 미해결 → Contrastive/DINO로 방법론 전환 시도

### 4.1 WGAN-GD 후반부 (9th~14th) — AP Gap 해결 시도

| 버전 | 핵심 아이디어 | 신규 Loss/모듈 | 결과 |
|------|---------------|----------------|------|
| 9th | Quantile Matching + Score Gap Penalty | `quantile_matching_loss`, `score_gap_penalty` | AP Gap 미변화 |
| 10th | Adaptive Alignment + Detection Guard | `gap_scale`, `detection_guard`, `score_uplift_loss` | 복잡도↑, 효과 불명 |
| **11th** | **Contrastive + GAN 하이브리드** | `ProjectionHead`, `cross_gender_contrastive_loss` | **전환점** |
| 12th | **Discriminator 제거**, Focal Loss | `focal_score_alignment_loss`, batch 4→7 | GAN-free 가능성 확인 |
| 13th | Multi-Scale Contrastive + Hard Mining | 3-branch ProjectionHead, hard_negative_mining | 복잡도 극대화 |
| 14th | AP Gap 직접 최적화 | `quantile_weighted_wasserstein`, `confidence_margin` | AP에 집중하나 효과 미미 |

### 4.2 Contrastive 시리즈 — GAN-Free 패러다임

| 버전 | 핵심 변경 | AP Gap | AR Gap | 평가 |
|------|-----------|--------|--------|------|
| **Contrastive 1st** | GAN-free, Cross-Gender InfoNCE | 0.108 | **0.0031** | AR에서 7th 동등 |
| Contrastive 2nd | 7th 스케줄링 통합 | **0.115** | 0.0069 | **악화** |
| Contrastive 3rd | 비대칭 Contrastive (1.5:0.5) | - | - | 1st 기반 안정화 |

### 4.3 DINO 1st — Self-Distillation

- Teacher: Male score 분포 (EMA, momentum 0.996→1.0)
- Student: Female score 분포
- 결과: AR Gap 0.0059 (27% 개선), AP Gap 0.106 (미변화)
- 안정적이나 개선폭 제한적

### 4.4 방법론 전환 경로

```
WGAN-GD 7th (국내학회)
    ↓ AP Gap 미개선 → 새 접근 필요
9th~10th (Score 정렬 강화) → 복잡도↑, 효과 불명
    ↓
11th (Contrastive + GAN 하이브리드) ← ★ 전환점
    ↓
12th (Discriminator 제거) → GAN 없이도 가능 확인
    ↓
Contrastive 1st (완전한 GAN-Free) → AR Gap 동등, 구조 단순
    ↓
DINO 1st (Self-Distillation) → 안정적이나 개선폭↓
    ↓
→ Phase 4: SimCLR InfoNCE 탐색
```

### 4.5 Phase 3 핵심 발견

1. **AR Gap vs AP Gap의 근본적 차이**: AR은 score threshold 조정으로 개선 가능, AP는 localization + calibration 동시 필요
2. **Feature 정렬의 한계**: Contrastive learning으로 feature space를 정렬해도 AP 개선으로 이어지지 않음
3. **복잡도와 성능의 역관계**: 9th→13th 복잡도 증가에도 핵심 지표(AP Gap) 미개선
4. **Contrastive의 가능성**: Discriminator 없이도 7th와 동등한 AR Gap 달성 → 구조 단순화 이점

---

## 5. Phase 4: SimCLR InfoNCE + Score-based Contrastive (2026년 1월 하순)

> **기간**: 2026-01-21 ~ 2026-01-31 (약 10일)
> **핵심**: Gender-Aware InfoNCE로 **AP Gap 최고 1.8% 개선** 달성, 그러나 프레임워크 한계 확인

### 5.1 실험 타임라인

```
1/21 ── SimCLR InfoNCE 1st (Cross-Gender) ────── 실패 (AP Gap 악화)
  │
  └──── Score-Based Contrastive v1/v2 ──────── AP Gap -1.3%
  │
1/24 ── InfoNCE 2nd (Score 기반, 성별 무관) ──── 실패
  │
1/27 ── InfoNCE 3rd (Gender-Aware + Adaptive) ── ★ 최고 결과 (AP Gap -1.8%)
  │      ├── fix1 (Fair Centroid) ──────────── 실패 (loss 포화)
  │      ├── fix2 (Male-Anchored) ─────────── 부분 성공 (AP Gap -1.2%)
  │      ├── fix3 (Direct Confidence Boost) ── 실패 (AP Gap 악화)
  │      └── fix4 (SupCon 정규화) ─────────── 미평가
  │
1/30 ── 4th (MoCo-Inspired) ──────────────── 미평가
```

### 5.2 주요 실험 결과

| 버전 | 접근법 | Best Epoch | AP Gap | Δ AP Gap | 평가 |
|------|--------|------------|--------|----------|------|
| Baseline | - | - | 0.1063 | - | - |
| 7th (WGAN) | Adversarial | 23 | 0.1059 | -0.04pp (0.4%) | 안정적 |
| Score v2 | Adaptive Ranking | 29 | 0.1049 | -0.14pp (1.3%) | 성별 무관 |
| **3rd (ep3)** | **Gender-Aware InfoNCE** | **3** | **0.1044** | **-0.19pp (1.8%)** | **최고** |
| 3rd (ep10) | Gender-Aware InfoNCE | 10 | 0.1048 | -0.15pp | Female AP 최고(41.3%) |
| fix1 | Fair Centroid | 9 | 0.1072 | +0.09pp | 실패 |
| fix2 | Male-Anchored | 29 | 0.1050 | -0.13pp (1.2%) | 안정적 |
| fix3 | Direct Boost | 13 | 0.1119 | +0.56pp | 실패 |

### 5.3 InfoNCE 3rd 상세 (전체 연구 기간 최고 AP Gap 개선)

**핵심 설계**:
- Anchor: Female, Positive: Male, Negative: 다른 Female
- **Adaptive Score Weighting**: score 차이가 큰 쌍에 더 강한 학습 신호
  ```
  w = 0.5 + sigmoid((score_m - score_f) × 5)  → [0.5, 1.5]
  ```
- 비대칭: `1.5 × L(F→M) + 0.5 × L(M→F)`
- Loss: `1.0 × L_contrastive + 0.2 × L_wasserstein + β × L_det`

**결과**:
| Epoch | Male AP | Female AP | AP Gap | 비고 |
|-------|---------|-----------|--------|------|
| 3 | 0.511 | 0.406 | **0.1044** | Best AP Gap |
| 10 | 0.517 | **0.413** | 0.1048 | Best Female AP |
| 23 | 0.518 | 0.408 | 0.1095 | 과적합 |

**한계**: Epoch 3에서 최적 → 이후 과적합. Score Gap 역전 현상 발생.

### 5.4 WGAN-GD 7th vs InfoNCE 3rd 비교

| 비교 항목 | WGAN 7th | InfoNCE 3rd (ep3) | 우위 |
|-----------|----------|-------------------|------|
| AP Gap 개선 | -0.04pp (0.4%) | -0.19pp (1.8%) | **InfoNCE** |
| AR Gap 개선 | -0.49pp (62%) | -0.55pp (69%) | **InfoNCE** |
| 학습 안정성 | Epoch 23까지 안정 | Epoch 3 최적 | **WGAN** |
| 구조 복잡도 | Generator + Discriminator | Generator + ProjectionHead | 동등 |
| 실용성 | 높음 (안정적) | 낮음 (Early Stop 필수) | **WGAN** |

### 5.5 Phase 4 핵심 발견

1. **Gender 정보 명시적 활용**이 Score-only 대비 효과적
2. **비대칭 학습** (F→M 강조)이 일관적으로 우위
3. **Male detach**가 과적합 방지에 효과적 (fix2)
4. **Representation-Performance Gap**: feature 정렬 ≠ detection 개선
5. **Score Gap 역전 현상**: 학습 중 score_m < score_f이지만 실제 AP_m > AP_f

---

## 6. 전체 실험 성능 비교표

### 6.1 주요 실험 Delta 성능 비교 (Baseline 대비)

| Phase | 버전 | Male AP Δ | Female AP Δ | AP Gap | AP Gap Δ | AR Gap Δ | 종합 |
|-------|------|-----------|-------------|--------|----------|----------|------|
| P1 | GD 2nd | +0.0005 | -0.0017 | - | - | - | 보통 |
| P2 | 3rd | -0.0149 | -0.0160 | - | 악화 | 개선 | 나쁨 |
| P2 | **4th** | -0.0076 | -0.0078 | - | 미변 | **-55%** | 전환점 |
| P2 | **7th** | **+0.0029** | **+0.0034** | 0.1059 | **-0.38%** | **-60.5%** | **최고(안정)** |
| P2 | 8th | ~7th | ~7th | ~7th | ~7th | ~7th | 미세 변형 |
| P3 | Contr.1st | +0.003 | +0.002 | 0.108 | +0.2% | **-61%** | GAN-free |
| P3 | DINO 1st | +0.001 | +0.002 | 0.106 | 0% | -27% | 안정적 |
| P4 | Score v2 | +0.008 | +0.009 | 0.1049 | **-1.3%** | -20% | 좋음 |
| P4 | **3rd(ep3)** | +0.001 | +0.002 | **0.1044** | **-1.8%** | **-69%** | **최고(AP)** |
| P4 | fix2 | +0.009 | +0.011 | 0.1050 | -1.2% | - | 안정적 |

### 6.2 핵심 지표 최고 기록

| 지표 | 최고 기록 | 달성 버전 | Phase |
|------|-----------|-----------|-------|
| **AP Gap 최소** | 0.1044 (-1.8%) | InfoNCE 3rd (ep3) | Phase 4 |
| **AR Gap 최소** | 0.0031 (-61%) | Contrastive 1st | Phase 3 |
| **Female AP 최고** | 0.413 (+0.9%) | InfoNCE 3rd (ep10) | Phase 4 |
| **가장 안정적** | 모든 지표 양수 delta | WGAN-GD 7th (ep23) | Phase 2 |

---

## 7. 방법론 진화 흐름도

```
2025-11 ┌─────────────────────────────────────────────────────┐
        │ Phase 1: 기본 Adversarial Fairness                  │
        │                                                     │
        │  기본GAN → +Wasserstein → +양성교란 → +가중치강화    │
        │  (2nd)     (WGAN)        (GD 1st)    (GD 2nd)       │
        └──────────────────────────┬──────────────────────────┘
                                   │
2025-12 ┌──────────────────────────▼──────────────────────────┐
        │ Phase 2: WGAN-GD 최적화                              │
        │                                                     │
        │  양방향W(3rd,실패) → 단방향W(4th,전환점)              │
        │  데이터제한(5th,실패) → 스케줄(6th,보통)              │
        │  stress(6th_2,실패) → ★ 7th(국내학회) → 8th         │
        │                                                     │
        │  [별도] MMD 1st → MMD 2nd (7th 미달)                 │
        └──────────────────────────┬──────────────────────────┘
                                   │ AP Gap 미해결
2026-01 ┌──────────────────────────▼──────────────────────────┐
 초~중순 │ Phase 3: 방법론 전환                                 │
        │                                                     │
        │  9th~10th(Score강화) → 11th(★Contrastive 도입)       │
        │  → 12th(D 제거) → Contrastive 1st(GAN-free)         │
        │  → 13th(Multi-Scale) → 14th(AP직접최적화)            │
        │  → Contrastive 2nd(실패) → 3rd → DINO 1st           │
        └──────────────────────────┬──────────────────────────┘
                                   │ Feature 정렬 ≠ AP 개선
2026-01 ┌──────────────────────────▼──────────────────────────┐
 하순    │ Phase 4: InfoNCE 집중 탐색                           │
        │                                                     │
        │  InfoNCE 1st(Cross-Gender,실패)                      │
        │  → Score-based v1/v2(1.3%↓) → 2nd(실패)             │
        │  → ★ 3rd(Gender-Aware,1.8%↓최고)                    │
        │  → fix1(실패) → fix2(안정적) → fix3(실패)            │
        │  → fix4,4th(미평가)                                  │
        └─────────────────────────────────────────────────────┘
```

---

## 8. 핵심 인사이트 및 교훈

### 8.1 성공한 전략 패턴

| 패턴 | 첫 발견 | 적용 범위 | 설명 |
|------|---------|-----------|------|
| **단방향/비대칭 정렬** | GD 4th | 전체 | Female만 끌어올리고 Male은 고정(detach) |
| **fair_m_scale** | GD 7th | WGAN 계열 | 남성 perturbation 억제 (0.5배) |
| **완만한 스케줄링** | GD 7th | WGAN 계열 | epsilon cooldown + beta 점진 증가 |
| **Gender 정보 명시적 사용** | InfoNCE 3rd | Contrastive 계열 | Score만으로는 성별 구분 불가 |
| **Male detach** | GD 4th, fix2 | 전체 | Male gradient 차단으로 과적합 방지 |

### 8.2 실패한 전략 패턴

| 패턴 | 실패 사례 | 교훈 |
|------|-----------|------|
| 양방향 정렬 | GD 3rd, MMD 1st | 남성 성능까지 끌어내림 |
| 과도한 가중치 변화 | GD 6th, 6th_2 | 학습 불안정 유발 |
| 데이터 제한 | GD 5th | per-gender cap이 학습 방해 |
| Score-only 분리 | InfoNCE 2nd | Low/High에 남녀 균등 분포 |
| 대칭 학습 | fix1 | representation collapse |
| Direct confidence boost | fix3 | Male이 더 큰 수혜 |
| 복잡도 증가 | 9th~13th | 7개 loss 항목이 단순 구조보다 나쁨 |

### 8.3 핵심 발견

1. **AR Gap은 해결 가능, AP Gap은 근본적으로 어려움**
   - AR: score threshold 조정으로 개선 가능 → 다양한 방법이 60%+ 개선 달성
   - AP: precision-recall AUC → localization + calibration 동시 필요 → 최선 1.8% 개선

2. **"비대칭" 원칙이 FAAP 연구의 핵심 설계 철학**
   - 약자 그룹(Female)만 끌어올리고 강자 그룹(Male)은 보호
   - WGAN(단방향 Wasserstein), Contrastive(비대칭 가중치), DINO(Teacher=Male) 모두 동일 원칙

3. **Feature 정렬 ≠ Detection 성능 개선**
   - Contrastive learning으로 representation을 정렬해도 detection logit 개선으로 직결되지 않음
   - Mean pooling(100 query → 1 vector)으로 per-object 정보 손실

4. **보수적 변경이 급진적 변경보다 효과적**
   - 검증된 설정에서 1가지만 변경하는 것이 가장 안정적
   - 한 번에 여러 가지 바꾸면 실패 확률 급증

5. **Occam's Razor**: 단순한 구조가 복잡한 구조와 동등하거나 더 나은 성능
   - Contrastive 1st(단순) ≈ 13th(7개 loss, Multi-Scale)

---

## 9. 현재 상태 및 향후 방향

### 9.1 현재 최고 성능

| 용도 | 추천 버전 | AP Gap | AR Gap | 비고 |
|------|-----------|--------|--------|------|
| **안정성 우선** | WGAN-GD 7th | 0.1059 (-0.4%) | 0.0032 (-60.5%) | 국내학회 제출 |
| **AP Gap 우선** | InfoNCE 3rd (ep3) | 0.1044 (-1.8%) | - | Early Stop 필수 |
| **Female AP 우선** | InfoNCE 3rd (ep10) | 0.1048 | - | Female AP 41.3% |

### 9.2 미실행/미평가 실험

| 버전 | 아이디어 | 기대 효과 |
|------|----------|-----------|
| InfoNCE 3rd_fix4 | SupCon 정규화 | positive 균등 기여 |
| InfoNCE 4th | MoCo Memory Bank + Momentum | 과적합 방지, 강한 contrastive signal |

### 9.3 근본적 한계

현재 **input perturbation + frozen DETR** 프레임워크의 구조적 한계:

1. DETR의 bias는 모델 가중치에 인코딩 → 입력만 변경해서는 근본적 해결 어려움
2. epsilon 제약 (±0.10 ≈ 픽셀 ±6) → 변화 범위 제한
3. 이미지 전체에 동일 perturbation 적용 → gender-specific 조절 불가
4. Score distribution 정렬 ≠ AP 개선 (localization 품질 미영향)

### 9.4 제안된 향후 연구 방향

| 방향 | 설명 | 우선순위 |
|------|------|----------|
| **IoU-Aware Contrastive** | IoU를 고려한 positive/negative sampling, AP와 직접 연관 | 최우선 |
| **Per-object Perturbation** | 객체별 차등 perturbation 적용 | 높음 |
| **DETR Fine-tuning** | Detector 가중치 직접 수정 (프레임워크 전환) | 높음 |
| **Post-processing Calibration** | 성별별 confidence 보정 | 중간 |
| Prototype-Guided Alignment | EMA prototype + diversity loss | 중간 |
| Hard Negative Mining (SCHaNe) | 어려운 negative에 가중치 | 낮음 |
| Group-wise Normalization | 성별별 별도 BatchNorm | 낮음 |

### 9.5 목표 지표

| 지표 | 현재 최선 | 목표 | 필요 개선 |
|------|----------|------|----------|
| AP Gap | 0.1044 (10.4%) | < 0.08 (8%) | 25%↓ |
| AR Gap | 0.0031 | < 0.002 | 35%↓ |
| Female AP | 0.413 | > 0.42 | 1.7%↑ |

---

## 부록: 주요 하이퍼파라미터 변화 추적

### WGAN-GD 계열 (3rd~8th)

| 파라미터 | 3rd | 4th | 5th | 6th | 7th | 8th |
|----------|-----|-----|-----|-----|-----|-----|
| epochs | 12 | 12 | 12 | 12 | **24** | 24 |
| epsilon_final | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| epsilon_min | - | - | - | 0.08 | **0.09** | 0.09 |
| beta | 0.5 | 0.5 | 0.5 | 0.5→0.7 | **0.5→0.6** | 0.5→0.6 |
| lambda_fair | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 |
| lambda_w | 0.1 | **0.2** | 0.2 | 0.2 | 0.2 | 0.2→**0.3** |
| Wasserstein 방향 | 양방향 | **단방향** | 단방향 | 단방향 | 단방향 | 단방향 |
| fair_m_scale | 1.0 | 1.0 | 1.0 | 1.0 | **0.5** | 0.5 |

### InfoNCE 계열

| 파라미터 | 1st | 2nd | 3rd | fix1 | fix2 |
|----------|-----|-----|-----|------|------|
| Anchor 정의 | Cross-Gender | Score 기반 | **Gender** | Centroid | **Gender(F)** |
| 비대칭 비율 | 1.5:0.5 | 1.5:0.5 | **1.5:0.5** | 1.0:1.0 | **1.0:0.0** |
| Male detach | No | No | No | No | **Yes** |
| Temperature | 0.07 | 0.07 | 0.07 | 0.1 | 0.07 |
| epsilon | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| Augmentation | ColorJitter | ColorJitter | ColorJitter | Weak | ColorJitter |

---

## 부록: 참조 논문

| 논문 | 적용 방식 |
|------|-----------|
| FAAP (Wang et al., CVPR 2022) | Perturbation 기반 fairness의 원 논문 |
| SimCLR (Chen et al., 2020) | Projection Head, Temperature, Augmentation 설계 |
| SupCon (Khosla et al., NeurIPS 2020) | fix4의 정규화 방식 |
| MoCo (He et al., CVPR 2020) | 4th의 Memory Bank + Momentum |
| DINO (Caron et al., 2021) | DINO 1st의 Self-Distillation 구조 |
| FSCL (Park et al., CVPR 2022) | Group-wise normalization 아이디어 |
| SCHaNe (2022) | Hard negative mining 제안 |
| FairAdaBN (2023) | Gender-wise normalization 제안 |

---

*본 보고서는 2025-11-24부터 2026-01-31까지의 FAAP-GAN 연구를 4개 Phase로 나누어 분석한 종합 보고서입니다.*
*총 분석 파일: 코드 30+개, 문서 20+개, Git 커밋 20+개*
