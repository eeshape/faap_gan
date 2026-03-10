<div align="center">

# FAAP-GAN

### 객체 검출의 공정성을 위한 적대적 섭동 생성 프레임워크

**동결된 DETR의 성별 편향을 입력 공간 섭동으로 개선**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.9+](https://img.shields.io/badge/pytorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[개요](#개요) · [아키텍처](#아키텍처) · [주요 결과](#주요-결과) · [시작하기](#시작하기) · [방법론](#방법론) · [실험](#실험)

</div>

---

## 개요

DETR과 같은 객체 검출 모델은 **성별에 따른 검출 성능 격차**를 보입니다 — 남성 이미지에서의 인물 검출이 여성 이미지보다 더 정확합니다. FAAP-GAN은 경량 **섭동 생성기(Perturbation Generator)**를 학습하여 입력 이미지에 눈에 보이지 않는 미세한 노이즈를 추가함으로써, *동결된 검출기를 수정하지 않고도* 성별 간 검출 격차를 줄입니다.

### 핵심 기여

- **입력 공간 공정성 보정** — U-Net 생성기가 제한된 크기의 섭동(ε ≤ 0.10)을 생성하여 성별 편향을 줄이면서 검출 품질을 유지
- **단방향 Wasserstein 정렬** — 여성 검출 점수를 남성 수준으로 끌어올리되, 남성 성능은 저하시키지 않음
- **적대적 디바이어싱** — DETR decoder feature에 대한 성별 판별기가 생성기를 학습시켜 성별 식별 단서를 억제
- **모델 수정 없음** — 사전학습된 DETR은 완전히 동결 상태를 유지하며, 입력 전처리만 변경

## 아키텍처

```
                          ┌──────────────────────────────────────────────────┐
                          │              FAAP-GAN 프레임워크                   │
                          └──────────────────────────────────────────────────┘

    ┌─────────┐     ┌───────────────────┐     ┌──────────────┐     ┌──────────────────┐
    │  입력   │────▶│     섭동 생성기     │────▶│  섭동 적용   │────▶│   동결된 DETR    │
    │  이미지  │     │  Generator (G)     │     │   이미지     │     │  (ResNet-50 +    │
    │  x      │     │  U-Net, ε·tanh     │     │  x + δ       │     │   Transformer)   │
    └─────────┘     └───────────────────┘     └──────────────┘     └────────┬─────────┘
                              ▲                                             │
                              │                              ┌──────────────┼──────────────┐
                         적대적                               ▼              ▼              │
                        그래디언트             ┌──────────────┐  ┌──────────────┐           │
                              │              │  검출 출력    │  │  Decoder     │           │
                              │              │  (박스, 클래스)│  │  Features    │           │
                              │              │              │  │  (B,100,256) │           │
                              │              └──────┬───────┘  └──────┬───────┘           │
                              │                     │                 │                   │
                              │                L_det │            ┌────▼────┐              │
                              │                     │            │  성별   │              │
                              │                     │            │ 판별기  │              │
                              │                     │            │  (D)    │              │
                              │                     │            └────┬────┘              │
                              │                     │                 │                   │
                              │                ┌────▼─────────────────▼────┐              │
                              └────────────────│   생성기 손실:              │              │
                                               │   L_G = λ·L_fair         │              │
                                               │       + β·L_det          │              │
                                               │       + λ_w·L_wass       │              │
                                               └──────────────────────────┘              │
                                                                                          │
                                       ┌──────────────────────────────────┐               │
                                       │   Wasserstein 정렬               │◀──────────────┘
                                       │   ↑ 여성 점수 → 남성 수준으로     │
                                       │   (단방향)                       │
                                       └──────────────────────────────────┘
```

## 주요 결과

### 검출 성능 비교 (테스트셋)

<table>
<tr>
<th rowspan="2">지표</th>
<th colspan="2">Baseline DETR</th>
<th colspan="2">FAAP-GAN (Ours)</th>
<th rowspan="2">Gap 변화</th>
</tr>
<tr>
<th>남성</th><th>여성</th><th>남성</th><th>여성</th>
</tr>
<tr>
<td><b>AP @[.50:.95]</b></td>
<td>0.511</td><td>0.404</td>
<td>0.514 <sub>(+0.003)</sub></td><td>0.408 <sub>(+0.004)</sub></td>
<td>0.106 → 0.106</td>
</tr>
<tr>
<td><b>AP @.50</b></td>
<td>0.628</td><td>0.510</td>
<td>0.631 <sub>(+0.003)</sub></td><td>0.513 <sub>(+0.003)</sub></td>
<td>0.118 → 0.118</td>
</tr>
<tr>
<td><b>AP @.75</b></td>
<td>0.565</td><td>0.452</td>
<td>0.568 <sub>(+0.003)</sub></td><td>0.457 <sub>(+0.005)</sub></td>
<td>0.113 → 0.111</td>
</tr>
<tr>
<td><b>AR @[.50:.95]</b></td>
<td>0.834</td><td>0.826</td>
<td>0.836 <sub>(+0.002)</sub></td><td>0.833 <sub>(+0.007)</sub></td>
<td><b>0.008 → 0.003 (↓60.5%)</b></td>
</tr>
</table>

### 공정성 Gap 감소

| 지표 | Baseline Gap | FAAP-GAN Gap | 감소율 |
|:----:|:-----------:|:------------:|:-----:|
| **AR Gap** (남−여) | 0.0081 | **0.0032** | **60.5% ↓** |
| AP Gap (남−여) | 0.1063 | 0.1059 | 0.4% ↓ |

### 핵심 성과

- **남녀 모두 성능 향상** — AP, AR이 남녀 *모두* 증가하며, 성능 희생 없음
- **AR Gap 60.5% 감소** — 0.0081에서 0.0032로, 가장 큰 공정성 개선
- **소형 객체 검출 향상** — AP_small: 0.068→0.153 (남성), AR_small: 0.18→0.28 (남성)
- **지각 불가능한 섭동** — ε = 0.09, 원본과 시각적으로 구분 불가

---

## 시작하기

### 사전 준비

```bash
# DETR 클론 (백본으로 필요)
git clone https://github.com/facebookresearch/detr.git
cd detr && pip install -e .

# 사전학습된 DETR 가중치 다운로드
wget https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth
```

### 설치

```bash
git clone https://github.com/your-username/faap_gan.git
cd faap_gan
pip install torch torchvision pycocotools
```

### 데이터셋 구조

성별 분리된 COCO 형식 데이터셋을 준비합니다:

```
faap_dataset/
├── women_split/
│   ├── train/                          # 여성 학습 이미지
│   ├── val/
│   ├── test/
│   ├── gender_women_train.json         # COCO 어노테이션
│   ├── gender_women_val.json
│   └── gender_women_test.json
└── men_split/
    ├── train/                          # 남성 학습 이미지
    ├── val/
    ├── test/
    ├── gender_men_train.json
    ├── gender_men_val.json
    └── gender_men_test.json
```

### 학습

```bash
# 단일 GPU
python train_faap_wgan_GD_7th.py \
  --dataset_root /path/to/faap_dataset \
  --detr_repo /path/to/detr \
  --detr_checkpoint /path/to/detr-r50-e632da11.pth \
  --output_dir faap_outputs

# 다중 GPU (분산 학습)
torchrun --nproc_per_node=4 train_faap_wgan_GD_7th.py --distributed

# 체크포인트에서 재개
python train_faap_wgan_GD_7th.py --resume faap_outputs/faap_outputs_gd_7th/checkpoints/epoch_0010.pth
```

### 평가

```bash
python eval_faap.py \
  --dataset_root /path/to/faap_dataset \
  --detr_repo /path/to/detr \
  --detr_checkpoint /path/to/detr-r50-e632da11.pth \
  --generator_checkpoint faap_outputs/faap_outputs_gd_7th/checkpoints/epoch_0023.pth \
  --results_path faap_outputs/faap_metrics.json
```

---

## 방법론

### 손실 함수

생성기는 세 가지 목표를 균형 잡는 복합 손실로 학습됩니다:

$$\mathcal{L}_G = \lambda_{\text{fair}} \cdot \mathcal{L}_{\text{fair}} + \beta \cdot \mathcal{L}_{\text{det}} + \lambda_w \cdot \mathcal{L}_{\text{wass}}$$

| 손실 | 수식 | 목적 |
|:-----|:-----|:-----|
| **공정성 손실** | $-(\text{CE}(D(h_f), \text{female}) + \alpha \cdot H(D(h_f)))$ | 적대적 디바이어싱 — 판별기를 혼란시킴 |
| **검출 손실** | DETR criterion (cls + bbox + GIoU) | 섭동 이미지의 검출 품질 유지 |
| **Wasserstein 손실** | $\text{ReLU}(\text{sort}(s_m) - \text{sort}(s_f)).\text{mean}()$ | 단방향 정렬 — 여성 점수를 남성 수준으로 끌어올림 |

### Epsilon 스케줄링

섭동 크기는 학습 안정성을 위해 **3단계 스케줄**을 따릅니다:

```
ε
0.10 ┤          ┌──────────────┐
     │         ╱                ╲
0.09 ┤        ╱                  └──────────
     │       ╱
0.05 ┤──────╱
     └──────┬──────┬──────────────┬─────────── epoch
            0      8             14          24
         Warmup    Hold         Cooldown
```

| 단계 | 구간 | 설명 |
|:----:|:----:|:-----|
| **Warmup** | 0 → 8 epoch | ε를 0.05에서 0.10으로 점진적 증가, 안정적 학습 시작 |
| **Hold** | 8 → 14 epoch | ε = 0.10 유지, 충분한 섭동으로 학습 |
| **Cooldown** | 14 → 24 epoch | ε를 0.10에서 0.09로 감소, 미세 조정으로 수렴 |

### 비대칭 공정성 가중치

성별별 차등 가중치로 여성 공정성 개선에 집중합니다:

| 구성 요소 | 가중치 | 근거 |
|:----------|:------:|:-----|
| 여성 공정성 스케일 | 1.0 | 여성 디바이어싱에 전체 그래디언트 적용 |
| 남성 공정성 스케일 | 0.5 | 남성 성능 방해를 방지하기 위해 축소 |
| Wasserstein의 남성 점수 | `detach()` | 남성을 고정 타겟으로 사용 — 여성만 끌어올림 |

---

## 실험

### 접근법 비교

다양한 공정성 접근법을 탐색한 결과, WGAN-GD (7th)가 최적의 균형을 달성했습니다:

| 버전 | 접근법 | ε | 남성 AP | 여성 AP | AP Gap 변화 | AR Gap 변화 | 평가 |
|:----:|:-------|:-:|:------:|:-------:|:----------:|:----------:|:-----|
| Baseline | — | — | 0.511 | 0.404 | — | — | — |
| 3rd | 양방향 Wasserstein | 0.10 | 0.496 | 0.388 | +0.1% | −19.8% | 성능 하락 |
| 4th | 단방향 Wasserstein | 0.08 | 0.503 | 0.397 | +0.9% | −67.9% | AR 양호, AP 손실 |
| **7th** | **WGAN-GD (최종)** | **0.09** | **0.514** | **0.408** | **−0.4%** | **−60.5%** | **최적 균형** |
| 11th | 낮은 epsilon | 0.01 | 0.514 | 0.409 | −1.9% | −1.2% | 효과 미미 |
| InfoNCE | Gender-Aware Contrastive | 0.10 | 0.521 | 0.415 | −0.8% | −40.7% | 불안정 |

### 상세 COCO 지표 (7th, ε=0.09)

<details>
<summary><b>전체 지표 펼치기</b></summary>

#### 남성

| 지표 | Baseline | FAAP-GAN | 변화량 |
|:-----|:--------:|:--------:|:-----:|
| AP @[.50:.95] | 0.511 | 0.514 | +0.003 |
| AP @.50 | 0.628 | 0.631 | +0.003 |
| AP @.75 | 0.565 | 0.568 | +0.003 |
| AP (small) | 0.068 | 0.153 | **+0.085** |
| AP (medium) | 0.063 | 0.062 | −0.001 |
| AP (large) | 0.525 | 0.527 | +0.002 |
| AR @[.50:.95] | 0.834 | 0.836 | +0.002 |
| AR (small) | 0.180 | 0.280 | **+0.100** |
| AR (medium) | 0.453 | 0.471 | +0.018 |
| AR (large) | 0.843 | 0.844 | +0.001 |

#### 여성

| 지표 | Baseline | FAAP-GAN | 변화량 |
|:-----|:--------:|:--------:|:-----:|
| AP @[.50:.95] | 0.404 | 0.408 | +0.004 |
| AP @.50 | 0.510 | 0.513 | +0.003 |
| AP @.75 | 0.452 | 0.457 | +0.005 |
| AP (small) | 0.017 | 0.023 | +0.006 |
| AP (medium) | 0.091 | 0.080 | −0.011 |
| AP (large) | 0.415 | 0.418 | +0.003 |
| AR @[.50:.95] | 0.826 | 0.833 | **+0.007** |
| AR (small) | 0.150 | 0.250 | **+0.100** |
| AR (medium) | 0.479 | 0.509 | +0.030 |
| AR (large) | 0.832 | 0.839 | +0.007 |

</details>

### InfoNCE Contrastive 변형

GAN 학습 대신 SimCLR 스타일 InfoNCE 손실을 사용하는 대안적 접근:

<details>
<summary><b>InfoNCE 결과 펼치기</b></summary>

| Epoch | 남성 AP | 여성 AP | AP Gap 변화율 | AR Gap 변화율 |
|:-----:|:------:|:-------:|:------------:|:------------:|
| E0 | 0.513 | 0.413 | **−6.0%** | **−91.4%** |
| E1 | 0.516 | 0.413 | −2.5% | −45.7% |
| E3 | **0.521** | **0.415** | −0.8% | −40.7% |
| E5 | 0.519 | 0.414 | −0.7% | −1.2% |
| E10 | 0.514 | 0.406 | +1.5% | −11.1% |

**분석**: InfoNCE는 초기 epoch에서 강력한 공정성 개선(E0: AR Gap −91.4%)을 보이나 불안정합니다 — 학습이 진행될수록 공정성이 퇴화합니다. WGAN-GD(7th)가 더 일관된 개선을 제공합니다.

</details>

---

## 하이퍼파라미터

| 범주 | 파라미터 | 값 | 설명 |
|:-----|:--------|:--:|:-----|
| **학습** | `epochs` | 24 | 총 학습 에폭 수 |
| | `batch_size` | 4 | GPU당 배치 크기 |
| | `lr_g` | 1e-4 | 생성기 학습률 (Adam) |
| | `lr_d` | 1e-4 | 판별기 학습률 (Adam) |
| | `k_d` | 4 | 반복당 판별기 업데이트 횟수 |
| | `max_norm` | 0.1 | 생성기 그래디언트 클리핑 |
| **섭동** | `epsilon` | 0.05 → 0.10 → 0.09 | Warmup → Hold → Cooldown |
| | `warmup_epochs` | 8 | Epsilon 워밍업 기간 |
| | `hold_epochs` | 6 | Epsilon 유지 기간 |
| | `cooldown_epochs` | 10 | Epsilon 쿨다운 기간 |
| **손실 가중치** | `lambda_fair` | 2.0 | 공정성 손실 가중치 |
| | `lambda_w` | 0.2 | Wasserstein 손실 가중치 |
| | `beta` | 0.5 → 0.6 | 검출 손실 가중치 (선형 증가) |
| | `alpha` | 0.2 | 엔트로피 정규화 |
| **공정성** | `fair_f_scale` | 1.0 | 여성 공정성 가중치 |
| | `fair_m_scale` | 0.5 | 남성 공정성 가중치 |

---

## 프로젝트 구조

```
faap_gan/
├── models.py                           # FrozenDETR, PerturbationGenerator, GenderDiscriminator
├── datasets.py                         # GenderCocoDataset, 균형 샘플링
├── train_faap_wgan_GD_7th.py           # 메인 학습 스크립트 (최종 버전)
├── eval_faap.py                        # 평가: 성별별 AP/AR + Gap 분석
├── eval_perturb.py                     # 섭동 분석 유틸리티
├── gen_images.py                       # 시각화: 원본 vs 섭동 이미지
├── path_utils.py                       # 경로 처리 헬퍼
│
├── train_faap_simclr_infonce_*.py      # InfoNCE contrastive 변형들
├── train_faap_wgan_GD_*.py             # WGAN-GD 반복 히스토리
├── train_faap_contrastive_*.py         # Contrastive learning 변형들
├── train_faap_mmd_*.py                 # MMD 기반 변형들
├── train_faap_dino_*.py                # DINO 기반 변형들
│
├── faap_outputs/                       # 학습 출력
│   ├── faap_outputs_gd_7th/            # 최종 모델 체크포인트 & 로그
│   │   ├── checkpoints/epoch_XXXX.pth
│   │   ├── train_log.jsonl
│   │   └── dataset_layout.json
│   └── ...
│
├── docs/                               # 연구 문서
│   ├── 00_overview.md
│   ├── FAAP_Research_History.md
│   └── ...
└── Report/                             # 실험 보고서
```

### 출력 형식

**체크포인트** (`epoch_XXXX.pth`):
```python
{"epoch": int, "generator": state_dict, "discriminator": state_dict,
 "opt_g": optimizer_state, "opt_d": optimizer_state, "args": dict}
```

**평가 결과** (`faap_metrics.json`):
```json
{
  "baseline": {"male": {"AP": 0.511, "AR": 0.834}, "female": {"AP": 0.404, "AR": 0.826}},
  "perturbed": {"male": {"AP": 0.514, "AR": 0.836}, "female": {"AP": 0.408, "AR": 0.833}},
  "gaps": {"AP": {"baseline": 0.106, "perturbed": 0.106}, "AR": {"baseline": 0.008, "perturbed": 0.003}}
}
```

---

## 설계 결정

| 결정 | 근거 |
|:-----|:-----|
| **동결된 DETR** | 실제 배포 환경을 모사 — 검출기를 재학습할 수 없는 상황 가정 |
| **단방향 Wasserstein** | `ReLU(남성 − 여성)`: 여성이 남성보다 뒤처질 때만 페널티 부여. 남성 성능 저하 방지 |
| **비대칭 공정성 가중치** | 여성(1.0) vs 남성(0.5): 불리한 그룹에 학습을 집중 |
| **3단계 Epsilon 스케줄** | Warmup으로 초기 불안정 방지, Hold로 충분한 효과, Cooldown으로 수렴 미세 조정 |
| **Beta 선형 증가** (0.5→0.6) | 초반에는 공정성을 우선시하고, 후반에는 검출 성능 보존으로 전환 |

---

## 의존성

- Python ≥ 3.8
- PyTorch ≥ 1.9
- [DETR](https://github.com/facebookresearch/detr) (facebook/detr)
- pycocotools
- torchvision

---

## 감사의 글

이 프로젝트는 Facebook AI Research의 [DETR: End-to-End Object Detection with Transformers](https://github.com/facebookresearch/detr)를 기반으로 구축되었습니다.

## 라이선스

이 프로젝트는 MIT 라이선스로 배포됩니다.
