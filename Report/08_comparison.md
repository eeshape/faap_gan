# 8. 전체 비교 요약 및 결론

> **연구 기간**: 2025-11-24 ~ 2026-01-31 (약 2개월)
> **총 실험 수**: 30+ 버전 (WGAN-GD 14종, MMD 2종, Contrastive 3종, DINO 1종, SimCLR InfoNCE 8종+)
> **Baseline**: Male AP 0.511, Female AP 0.404, AP Gap 0.1063, AR Gap 0.0081

---

## 8.1 전체 성능 비교표

### 8.1.1 Phase별 주요 버전 성능 (Baseline 대비 Delta)

| Phase | 버전 | 접근법 | Discriminator | Male AP Δ | Female AP Δ | AP Gap | AP Gap Δ | AR Gap | AR Gap Δ | 비고 |
|-------|------|--------|---------------|-----------|-------------|--------|----------|--------|----------|------|
| - | **Baseline** | 원본 DETR | - | - | - | 0.1063 | - | 0.0081 | - | 기준선 |
| P1 | GD 1st | 양성 교란 + Wasserstein | O | +0.000 | -0.002 | - | - | - | - | 패러다임 확립 |
| P1 | GD 2nd | 가중치 강화 (λ_fair=5.0) | O | +0.001 | -0.002 | - | - | - | - | 보통 |
| P2 | **3rd** | 양방향 Wasserstein | O | **-0.015** | **-0.016** | - | **악화** | 개선 | - | 실패 |
| P2 | **4th** | 단방향 Wasserstein (전환점) | O | -0.008 | -0.008 | - | 미변 | - | **-55%** | 전환점 |
| P2 | 5th | per-gender 제한 (4500장) | O | -0.007 | -0.009 | - | 미변 | - | -15% | 보통 |
| P2 | 6th | epsilon cooldown + β 강화 | O | -0.006 | -0.009 | - | 미변 | - | -1% | 보통 |
| P2 | **7th** | 비대칭 Wasserstein (국내학회) | O | **+0.003** | **+0.003** | **0.1059** | **-0.4%** | **0.0032** | **-60.5%** | **안정 최고** |
| P2 | 8th | lambda_w 후반 boost | O | ~7th | ~7th | ~7th | ~7th | ~7th | ~7th | 7th 미세변형 |
| P2 | MMD 1st | Gaussian Kernel MMD (양방향) | X | - | - | - | 7th 미달 | - | 7th 미달 | 보통 |
| P2 | MMD 2nd | 비대칭 MMD | X | - | - | - | 7th 미달 | - | 7th 미달 | 보통 |
| P3 | 11th | Contrastive + GAN 하이브리드 | O | - | - | - | - | - | - | 전환점 |
| P3 | 12th | Discriminator 제거 | X | - | - | - | - | - | - | GAN-free 확인 |
| P3 | **Contr. 1st** | GAN-free, Cross-Gender InfoNCE | X | +0.003 | +0.002 | 0.108 | +0.2% | **0.0031** | **-61%** | AR 최고 타이 |
| P3 | Contr. 2nd | 7th 스케줄 통합 | X | - | - | **0.115** | **악화** | 0.0069 | -15% | 실패 |
| P3 | Contr. 3rd | 비대칭 Contrastive (1.5:0.5) | X | - | - | - | 안정화 | - | - | 1st 기반 |
| P3 | **DINO 1st** | Self-Distillation (EMA) | X | +0.001 | +0.002 | 0.106 | 0% | 0.0059 | -27% | 안정적 |
| P4 | InfoNCE 1st | Cross-Gender InfoNCE | X | - | - | - | 악화 | - | - | 실패 |
| P4 | **Score v2** | Adaptive Percentile Ranking | X | +0.008 | +0.009 | **0.1049** | **-1.3%** | - | -20% | 좋음 |
| P4 | InfoNCE 2nd | Score 기반 분리 | X | - | - | - | 악화 | - | - | 실패 |
| P4 | **3rd (ep3)** | Gender-Aware InfoNCE | X | +0.001 | +0.002 | **0.1044** | **-1.8%** | - | **-69%** | **AP 최고** |
| P4 | 3rd (ep10) | Gender-Aware InfoNCE | X | +0.007 | **+0.009** | 0.1048 | -1.5% | - | - | Female AP 최고 |
| P4 | fix1 | Fair Centroid | X | - | - | 0.1072 | +0.9% | - | - | 실패 |
| P4 | **fix2** | Male-Anchored Contrastive | X | +0.009 | +0.011 | **0.1050** | **-1.2%** | - | - | 안정적 |
| P4 | fix3 | Direct Confidence Boost | X | - | - | 0.1119 | **+0.56%** | - | - | 실패 |

*Δ: Baseline 대비 변화량. "-": 측정값 없음 또는 미기재. AP Gap Δ는 (Gap_perturbed - Gap_baseline) / Gap_baseline × 100%.*

---

## 8.2 접근법별 비교 매트릭스

| 카테고리 | 대표 버전 | 최고 AP Gap 개선 | 최고 AR Gap 개선 | 학습 안정성 | 구조 복잡도 | 특이사항 |
|----------|-----------|-----------------|-----------------|------------|------------|---------|
| **WGAN-GD (단방향)** | 7th | -0.4% | **-60.5%** | ★★★★★ | 중간 (G+D) | 국내학회 제출, 모든 지표 양수 |
| **MMD** | MMD 2nd | 7th 미달 | 7th 미달 | ★★★☆☆ | 낮음 (G만) | D 없이 분포 정렬, 한계 명확 |
| **GAN-free Contrastive** | Contr. 1st | +0.2% (악화) | **-61%** | ★★★★☆ | 낮음 (G만) | AR 최고, AP 미개선 |
| **DINO Self-Distillation** | DINO 1st | 0% | -27% | ★★★★★ | 중간 (EMA) | 안정적이나 개선폭 제한 |
| **Score-Based Contrastive** | Score v2 | **-1.3%** | -20% | ★★★☆☆ | 중간 (ProjHead) | 성별 정보 없이 ranking |
| **Gender-Aware InfoNCE** | 3rd (ep3) | **-1.8%** | **-69%** | ★★☆☆☆ | 중간 (ProjHead) | AP 최고, epoch 3 후 과적합 |
| **Male-Anchored (fix2)** | fix2 (ep29) | -1.2% | - | ★★★★☆ | 중간 (ProjHead) | 안정적, Male detach 효과 |
| **Direct Boost (fix3)** | fix3 | -0% (악화) | - | ★★☆☆☆ | 낮음 | Gender-agnostic 한계 노출 |

---

## 8.3 핵심 지표 최고 기록

| 지표 | 최고 기록 | 달성 버전 | Phase | 비고 |
|------|-----------|-----------|-------|------|
| **AP Gap 최소** | **0.1044** (-1.8%) | InfoNCE 3rd (ep3) | Phase 4 | Early Stop 필수, 불안정 |
| **AR Gap 최소** | **0.0031** (-61.7%) | Contrastive 1st | Phase 3 | WGAN 7th(0.0032)와 사실상 동등 |
| **Female AP 최고** | **0.413** (+2.2%) | InfoNCE 3rd (ep10) | Phase 4 | AP Gap은 0.1048로 소폭 상승 |
| **Male AP 최고** | **~0.520** | fix2 (ep29), 3rd (ep23) | Phase 4 | Baseline 0.511 대비 +1.8% |
| **가장 안정적** | 모든 지표 양수 delta | WGAN-GD 7th (ep23) | Phase 2 | 국내학회 제출 버전 |
| **AP Gap 최악** | 0.115 (+8.2%) | Contrastive 2nd | Phase 3 | 7th 스케줄 통합 실패 |

---

## 8.4 성공 패턴 vs 실패 패턴

### 8.4.1 성공 패턴

| 패턴 | 첫 발견 | 이후 적용 | 메커니즘 |
|------|---------|-----------|---------|
| **단방향/비대칭 정렬** | GD 4th (ReLU Wasserstein) | 전체 계열 | Female만 끌어올리고 Male은 detach로 보호 |
| **fair_m_scale 억제** | GD 7th (0.5배) | WGAN 계열 | 남성 perturbation 억제로 성능 유지 |
| **완만한 ε 스케줄링** | GD 7th (cooldown) | 대부분 계열 | 0.05→0.10→0.09: 안정적 수렴 |
| **충분한 학습 시간** | GD 7th (24ep) | 대부분 계열 | 12ep 대비 2배: 수렴 품질 향상 |
| **Gender 정보 명시적 사용** | InfoNCE 3rd | Contrastive 계열 | Score-only 대비 명확한 최적화 방향 |
| **Male detach** | GD 4th, fix2 | 전체 계열 | Male gradient 차단으로 과적합 방지 |
| **Gradient Clipping** | 초기부터 | 전체 계열 | max_norm=0.1로 학습 안정화 |

### 8.4.2 실패 패턴

| 패턴 | 실패 사례 | 근본 원인 |
|------|-----------|-----------|
| **양방향 Wasserstein** | GD 3rd, MMD 1st | Male 성능까지 하락, 전체 AP 저하 |
| **과도한 가중치 변화** | GD 6th_2 (stress window 1.3~2.0배) | 학습 불안정, 급격한 Loss 변동 |
| **데이터 제한** | GD 5th (4500장 cap) | 학습 다양성 저하, 수렴 방해 |
| **Score-only 분리** | InfoNCE 2nd | Low/High score 내에 male/female 균등 분포 → gender 정보 소실 |
| **대칭 학습** | fix1 (Fair Centroid) | Representation collapse, loss 포화 |
| **Direct Confidence Boost** | fix3 | Gender-agnostic perturbation → male도 동등 수혜 |
| **복잡도 과잉** | GD 9th~13th (7개 loss 항) | 단순 구조와 동등하거나 열세, 학습 불안정 |
| **대칭 InfoNCE** | InfoNCE 1st (1.5:0.5 동등 적용) | 방향성 없는 feature 정렬 |

---

## 8.5 근본적 한계

### 8.5.1 Input Perturbation + Frozen DETR 프레임워크의 구조적 제약

본 연구 전반에 걸쳐 확인된 근본적 한계는 방법론의 문제가 아닌, 채택한 프레임워크 자체의 구조적 제약에서 비롯된다.

| 한계 | 설명 | 실험적 근거 |
|------|------|------------|
| **편향의 위치** | DETR의 gender bias는 모델 가중치에 인코딩됨. 입력 변경만으로는 근본적 해결 불가 | 모든 버전에서 AP Gap 최대 1.8% 개선에 그침 |
| **ε 제약** | 허용 perturbation 범위 ±0.10 ≈ 픽셀 ±6 (ImageNet 정규화 기준) | ε 증가 시 detection 성능 저하 (3rd vs 7th) |
| **Gender-Agnostic Perturbation** | 이미지 전체에 동일 perturbation 적용 → female-specific 조절 불가 | fix3 실패: female boost 시 male도 동등 수혜 |
| **Score 정렬 ≠ AP 개선** | Detection score 분포 정렬이 AP(localization 품질)로 이어지지 않음 | AR Gap은 60%+ 개선, AP Gap은 최대 1.8% 개선 |
| **Mean Pooling 병목** | 100 object query → mean → 1 vector로 per-object 정보 손실 | Contrastive feature 정렬이 detection logit 개선으로 미연결 |
| **AR vs AP 비대칭** | AR: score threshold 조정으로 개선 가능. AP: precision-recall 전 구간 AUC → localization + calibration 동시 필요 | WGAN 7th: AR Gap -60%, AP Gap -0.4% |

### 8.5.2 Representation-Performance Gap

연구 전반의 핵심 발견: feature space 정렬과 detection performance 향상은 별개의 문제다.

```
[Feature Space]            [Detection Performance]
  Female feature     ≠→       Female AP
  ≈ Male feature              ≈ Male AP

  (Contrastive으로           (Localization 품질,
   정렬 가능)                 IOU 분포 개선 필요)
```

---

## 8.6 향후 연구 방향

| 방향 | 핵심 아이디어 | 현재 한계 극복 | 기대 효과 | 우선순위 |
|------|---------------|----------------|-----------|---------|
| **IoU-Aware Contrastive** | Positive/Negative 샘플링 시 IoU 품질 반영. AP와 직접 연관된 신호 사용 | Score 정렬 ≠ AP 개선 | AP Gap -5~10% | **최우선** |
| **Per-object Perturbation** | 객체 위치 정보를 사용한 region-specific perturbation 생성 | Gender-agnostic 제약 | Female 객체 직접 조작 | 높음 |
| **DETR Fine-tuning** | Detector 가중치 직접 수정, 공정성 제약 추가 (프레임워크 전환) | 편향이 가중치에 인코딩 | AP Gap 근본적 해결 | 높음 |
| **Post-processing Calibration** | 성별 감지 후 confidence score 후처리 보정 | epsilon 제약 | 구현 용이, 즉각 적용 | 중간 |
| **Prototype-Guided Alignment** | EMA prototype + diversity loss로 표현 다양성 유지 | Representation collapse | 안정적 contrastive 학습 | 중간 |
| **Hard Negative Mining (SCHaNe)** | 어려운 negative 샘플에 적응적 가중치 | 학습 효율성 | InfoNCE 3rd 개선 | 낮음 |
| **Group-wise Normalization** | 성별별 별도 BatchNorm/LayerNorm | Feature 분포 불일치 | 안정적 gender 분리 | 낮음 |

### 8.6.1 목표 지표

| 지표 | 현재 최선 | 향후 목표 | 필요 개선율 | 추천 접근법 |
|------|----------|-----------|------------|------------|
| **AP Gap** | 0.1044 (-1.8%) | < 0.080 | 25%↓ 추가 | IoU-Aware Contrastive + DETR Fine-tuning |
| **AR Gap** | 0.0031 (-61%) | < 0.002 | 35%↓ 추가 | 현재 방법 유지 + Per-object Perturbation |
| **Female AP** | 0.413 (+2.2%) | > 0.420 | 1.7%↑ 추가 | DETR Fine-tuning |

---

## 8.7 결론

### 8.7.1 연구 여정 요약

본 연구는 2025년 11월부터 2026년 1월까지 약 두 달에 걸쳐 30개 이상의 실험 버전을 통해 Frozen DETR 기반 성별 공정성 개선 문제를 탐구했다. 연구는 크게 네 단계로 진화했다. Phase 1에서 기본적인 Adversarial Fairness 프레임워크와 Wasserstein 정렬 아이디어를 확립했으며, Phase 2에서 단방향/비대칭 원칙이라는 핵심 설계 철학을 발견하여 국내학회 제출 버전(WGAN-GD 7th)을 완성했다. AP Gap의 근본적 어려움에 직면한 Phase 3에서는 Discriminator를 제거하고 Contrastive Learning으로 패러다임을 전환했으며, Phase 4에서는 Gender-Aware InfoNCE를 통해 AP Gap 1.8% 개선이라는 전체 연구 기간 최고 기록을 달성했다.

### 8.7.2 핵심 성과와 한계

가장 중요한 성과는 두 가지로 요약된다. 첫째, WGAN-GD 7th는 학습 안정성과 AR Gap 60.5% 개선을 동시에 달성하여 실용적 관점에서 가장 완성도 높은 결과를 보여주었다. 둘째, InfoNCE 3rd (ep3)는 AP Gap 1.8% 개선과 AR Gap 69% 개선을 기록했으나 epoch 3 이후 급격한 과적합이 발생하여 실용성에 제한이 있었다. 반면 AP Gap 개선의 절대적 한계는 명확하게 드러났다. 30개 이상의 다양한 접근법에도 불구하고 AP Gap 최대 개선은 1.8%에 그쳤으며, 이는 DETR의 gender bias가 모델 가중치 내부에 깊이 인코딩되어 있어 입력 수준의 perturbation으로는 근본적 해결이 어렵다는 프레임워크의 구조적 한계를 반영한다.

### 8.7.3 연구의 기여와 의의

본 연구는 결과 수치 이상의 기여를 남겼다. "단방향/비대칭 정렬" 원칙이 FAAP 연구의 핵심 설계 철학으로 확립되었으며, AR Gap과 AP Gap 개선의 본질적 차이에 대한 깊은 이해를 얻었다. 또한 Contrastive Learning 기반 GAN-free 접근법이 Discriminator 기반 방법과 동등한 성능을 달성할 수 있음을 실증했고, Direct Confidence Boosting 실험을 통해 Input Perturbation 프레임워크의 구조적 한계를 명확히 규명했다. 이러한 발견들은 향후 IoU-Aware Contrastive Learning 또는 DETR 가중치 직접 수정(Fine-tuning) 방향으로의 연구 전환에 대한 명확한 근거를 제공한다.

---

## 부록: 방법론 진화 전체 흐름도

```
2025-11 ┌────────────────────────────────────────────────────────┐
        │ Phase 1: 기본 프레임워크 확립                           │
        │  2nd(기본GAN) → WGAN(Wasserstein 도입)                  │
        │  → GD 1st(양성교란) → GD 2nd(가중치강화)                │
        └──────────────────────┬─────────────────────────────────┘
                               │
2025-12 ┌──────────────────────▼─────────────────────────────────┐
        │ Phase 2: WGAN-GD 최적화 → 국내학회 제출                  │
        │  3rd(양방향,실패) → 4th(단방향,전환점)                   │
        │  → 5th(데이터제한) → 6th(스케줄) → 6th_2(stress,실패)   │
        │  → ★ 7th(비대칭+24ep, 국내학회) → 8th                  │
        │  [별도] MMD 1st/2nd (7th 미달)                          │
        └──────────────────────┬─────────────────────────────────┘
                               │ AP Gap 미해결
2026-01 ┌──────────────────────▼─────────────────────────────────┐
 초~중순 │ Phase 3: 방법론 전환 (GAN-Free)                         │
        │  9th~10th(Score강화) → 11th(Contrastive 도입,전환점)    │
        │  → 12th(D제거) → Contr.1st(GAN-free,AR최고)            │
        │  → Contr.2nd(실패) → Contr.3rd → DINO 1st             │
        └──────────────────────┬─────────────────────────────────┘
                               │ Feature 정렬 ≠ AP 개선
2026-01 ┌──────────────────────▼─────────────────────────────────┐
 하순    │ Phase 4: InfoNCE 집중 탐색                              │
        │  InfoNCE 1st(Cross-Gender,실패)                         │
        │  → Score v2(1.3%↓) → InfoNCE 2nd(실패)                 │
        │  → ★ 3rd(Gender-Aware, 1.8%↓ 최고)                    │
        │     ├── fix1(Centroid,실패)                             │
        │     ├── fix2(Male-Anchored, 1.2%↓, 안정)               │
        │     ├── fix3(Direct Boost, 실패) ←── [7장]              │
        │     └── fix4(SupCon, 미평가)                            │
        │  → 4th(MoCo, 미평가)                                    │
        └────────────────────────────────────────────────────────┘
                               │
                    [구조적 한계 확인]
                               ↓
             향후: IoU-Aware Contrastive / DETR Fine-tuning
```

---

*이전 장: [07. Specialized 접근법 (Score Contrastive, Direct Boost)](./07_specialized.md)*
*본 보고서는 2025-11-24부터 2026-01-31까지의 FAAP-GAN 연구 전체를 종합한 최종 비교 요약본입니다.*
