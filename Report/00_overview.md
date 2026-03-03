# FAAP-GAN 버전별 Loss & Pipeline 종합 보고서

**작성일**: 2026-02-24
**대상 기간**: 2025-11-24 ~ 2026-01-31 (약 2개월)
**총 실험 수**: 39개 학습 스크립트, 7개 접근법 카테고리

---

## 1. 연구 개요

### 1.1 연구 목표
동결된(Frozen) DETR 객체 검출기에서 **성별 간 검출 성능(AP/AR) 격차를 줄이는 것**이 핵심 목표이다.
모델 가중치를 변경하지 않고, 입력 이미지에 미세한 perturbation을 생성하는 Generator를 학습하여 공정성을 개선하는 **FAAP(Fairness-Aware Adversarial Perturbation)** 프레임워크를 사용한다.

### 1.2 Baseline 성능 (원본 DETR)

| 지표 | Male | Female | Gap (M-F) |
|------|------|--------|-----------|
| AP@[0.50:0.95] | 0.511 | 0.404 | **0.106** |
| AR@[0.50:0.95] | 0.834 | 0.826 | **0.008** |

---

## 2. 공통 아키텍처

```
┌──────────────────────────────────────────────────────────┐
│                    FAAP 공통 파이프라인                      │
│                                                          │
│  Input Image ──→ PerturbationGenerator(G) ──→ δ          │
│                         │                                │
│  Perturbed = Image + ε·tanh(δ)                           │
│                         │                                │
│            ┌────────────▼────────────┐                    │
│            │     Frozen DETR         │                    │
│            │  (ResNet-50 + Transformer)│                  │
│            │  모든 파라미터 동결       │                    │
│            └────┬───────────┬────────┘                    │
│                 │           │                             │
│          Detection       Decoder                         │
│          Outputs        Features                         │
│       (logits, boxes)   (B×100×256)                      │
│                 │           │                             │
│            L_det        L_fairness                        │
│         (검출 유지)    (공정성 개선)                        │
└──────────────────────────────────────────────────────────┘
```

### 2.1 핵심 모듈 (`models.py`)

| 모듈 | 구조 | 역할 |
|------|------|------|
| **FrozenDETR** | ResNet-50 + Transformer (d_model=256) | 동결된 검출기. decoder feature 노출 |
| **PerturbationGenerator** | 경량 U-Net (32→64→128→64→32→3) + skip connection | `δ = ε·tanh(output)` bounded perturbation |
| **GenderDiscriminator** | LayerNorm → MLP(256→256→256→2) | decoder feature의 성별 분류 (일부 버전만) |

---

## 3. 접근법 카테고리 분류

| # | 카테고리 | 파일 범위 | Discriminator | 핵심 Loss | 상세 문서 |
|---|----------|-----------|:---:|-----------|-----------|
| 1 | **Base Adversarial** | `train_faap.py`, `2nd` | O | CE + Entropy | [01_base_adversarial.md](01_base_adversarial.md) |
| 2 | **WGAN-GD** | `wgan`, `wgan_GD` ~ `14th` | O→X | Adversarial + Wasserstein | [02_wgan_gd.md](02_wgan_gd.md) |
| 3 | **Contrastive** | `contrastive_1st` ~ `3rd`, `iou` | X | Cross-Gender InfoNCE | [03_contrastive.md](03_contrastive.md) |
| 4 | **MMD** | `mmd_1st`, `mmd_2nd` | X | Gaussian Kernel MMD | [04_mmd.md](04_mmd.md) |
| 5 | **DINO** | `dino_1st` | X | Self-Distillation | [05_dino.md](05_dino.md) |
| 6 | **SimCLR InfoNCE** | `simclr_infonce` ~ `9th` + fix* | X | InfoNCE + Augmentation | [06_simclr_infonce.md](06_simclr_infonce.md) |
| 7 | **Specialized** | `score_contrastive`, `direct_boost` | X | Adaptive Score / Direct Boost | [07_specialized.md](07_specialized.md) |

전체 비교 요약: [08_comparison.md](08_comparison.md)

---

## 4. 버전 진화 계보도

```
2025-11   ┌─────────────────────────────────────────────────────────────┐
 Phase 1  │  Base Adversarial                                          │
          │  2nd → train_faap → +Wasserstein(WGAN) → +양성교란(GD 1st) │
          │       → GD 2nd(가중치 강화)                                 │
          └────────────────────────┬────────────────────────────────────┘
                                   │
2025-12   ┌────────────────────────▼────────────────────────────────────┐
 Phase 2  │  WGAN-GD 최적화 + MMD 탐색                                  │
          │                                                            │
          │  3rd(양방향,실패) → 4th(★단방향 전환) → 5th(데이터제한)      │
          │  → 6th(스케줄링) → ★7th(국내학회 제출) → 8th                │
          │                                                            │
          │  [별도] MMD 1st(양방향) → MMD 2nd(비대칭) ← 7th 미달        │
          └────────────────────────┬────────────────────────────────────┘
                                   │ AP Gap 미해결
2026-01   ┌────────────────────────▼────────────────────────────────────┐
 초~중순   │  Phase 3: 방법론 전환                                        │
 Phase 3  │                                                            │
          │  9th(Score강화) → 10th → 11th(★Contrastive 도입)            │
          │  → 12th(D 제거) → 13th(Multi-Scale) → 14th(AP직접최적화)    │
          │                                                            │
          │  Contrastive 1st(GAN-free) → 2nd → 3rd                     │
          │  DINO 1st(Self-Distillation)                                │
          └────────────────────────┬────────────────────────────────────┘
                                   │ Feature 정렬 ≠ AP 개선
2026-01   ┌────────────────────────▼────────────────────────────────────┐
 하순      │  Phase 4: SimCLR InfoNCE 집중 탐색                          │
 Phase 4  │                                                            │
          │  InfoNCE 1st → Score-Contrastive v1/v2 → 2nd               │
          │  → ★3rd(Gender-Aware, AP Gap 최고 -1.8%)                   │
          │  → fix1~fix4 → 4th(MoCo) → 5th~9th                        │
          └────────────────────────────────────────────────────────────┘
```

---

## 5. 핵심 설계 원칙 (연구 전체를 관통하는 교훈)

| # | 원칙 | 첫 발견 | 설명 |
|---|------|---------|------|
| 1 | **단방향 정렬** | GD 4th | Female만 끌어올리고 Male은 고정(`detach`) |
| 2 | **비대칭 가중치** | GD 7th | `fair_f_scale=1.0 > fair_m_scale=0.5` |
| 3 | **완만한 스케줄링** | GD 6th~7th | Epsilon Warmup→Hold→Cooldown, Beta 점진 증가 |
| 4 | **단순성 우위** | 전체 | 복잡한 Multi-Loss < 단순한 구조 (Occam's Razor) |
| 5 | **AR ≠ AP** | Phase 3~4 | AR은 score threshold, AP는 localization+calibration 필요 |

---

## 6. 문서 구조

| 파일 | 내용 |
|------|------|
| `00_overview.md` | 본 문서 — 전체 개요, 계보도 |
| `01_base_adversarial.md` | Base GAN: Loss 수식, 파이프라인, 하이퍼파라미터 |
| `02_wgan_gd.md` | WGAN-GD 전체 (wgan ~ 14th): 핵심 계열 상세 |
| `03_contrastive.md` | Contrastive 계열 (1st~3rd, IoU) |
| `04_mmd.md` | MMD 계열 (1st~2nd) |
| `05_dino.md` | DINO 자기증류 (1st) |
| `06_simclr_infonce.md` | SimCLR InfoNCE 계열 (base~9th, fix*) |
| `07_specialized.md` | Score Contrastive, Direct Boost |
| `08_comparison.md` | 전체 비교 요약표, 결론, 향후 방향 |
