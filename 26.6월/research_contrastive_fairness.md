# Contrastive Learning for Fair Object Detection: 연구 계획

---

## 1. 기존 연구 분석

### 1.1 실험 결과 요약

| 방법 | Male AP | Female AP | AP Gap | AR Gap | 비고 |
|------|---------|-----------|--------|--------|------|
| Baseline (DETR) | 0.511 | 0.404 | 0.106 | 0.0081 | 원본 |
| **7th (WGAN-GD)** | **0.514** | **0.408** | 0.106 | **0.0032** | 논문 최종 |
| Contrastive 1st | 0.514 | 0.406 | 0.108 | 0.0031 | AR 개선 |
| Contrastive 2nd | 0.516 | 0.402 | 0.115 | 0.0069 | 여성 AP 하락 |
| DINO 1st | 0.512 | 0.406 | 0.106 | 0.0059 | 안정적 |

### 1.2 기존 방법의 한계

1. **AP Gap 미개선**: 모든 방법이 AP Gap을 줄이지 못함 (0.106 유지 또는 증가)
2. **AR vs AP 트레이드오프**: AR Gap은 개선되나 AP Gap은 오히려 악화
3. **여성 AP 향상 부족**: 남성 수준(0.51+)까지 올리지 못함
4. **Feature-Score 불일치**: Feature 정렬이 detection score 개선으로 이어지지 않음

### 1.3 기존 방법별 특성

#### WGAN-GD (7th)
```
장점: Discriminator의 adversarial signal이 강력함
      단방향 Wasserstein으로 방향성 명확
단점: GAN 학습 불안정성
      Detection loss와의 균형 어려움
```

#### Contrastive (1st, 2nd, 3rd)
```
장점: SimCLR 스타일 projection head로 안정적
      Feature space에서 직접 정렬
단점: Cross-gender contrastive가 AP 개선에 효과적이지 않음
      Score distribution과 연결 부족
```

#### DINO (1st)
```
장점: Teacher-Student로 안정적 학습
      Centering으로 collapse 방지
단점: Score 기반이라 feature 정보 부족
      개선폭이 작음
```

---

## 2. 핵심 문제 정의

### 2.1 왜 AP Gap이 개선되지 않는가?

**가설 1: Feature 정렬 ≠ Precision 개선**
- Contrastive learning은 feature similarity를 높임
- 하지만 높은 similarity ≠ 높은 IoU ≠ 높은 AP
- AP는 IoU threshold에 따른 precision 평균

**가설 2: Score 정렬만으로는 부족**
- Detection score를 높여도 localization이 부정확하면 AP 하락
- DINO는 score만 정렬하여 한계

**가설 3: 방향성 손실의 효과 부족**
- 단방향 Wasserstein이 AR에는 효과적
- 하지만 AP는 precision 기반이라 다른 접근 필요

### 2.2 AP 개선을 위한 핵심 요구사항

1. **Localization 품질 향상**: Bounding box 정확도 개선
2. **High-IoU Detection 증가**: IoU > 0.75인 detection 비율 증가
3. **False Positive 감소**: 여성 이미지에서의 오탐지 줄이기
4. **Score Calibration**: Confidence score와 실제 IoU 정렬

---

## 3. 새로운 연구 방향

### 3.1 Approach A: IoU-Aware Contrastive Learning

**핵심 아이디어**: IoU를 고려한 contrastive loss로 localization 품질 직접 최적화

```python
# Pseudo-code
def iou_aware_contrastive_loss(feat_f, feat_m, iou_f, iou_m):
    # High IoU detections를 positive로, Low IoU를 negative로
    # 여성 high-IoU feature가 남성 high-IoU feature와 유사해지도록

    high_iou_f = feat_f[iou_f > 0.5]  # 여성 high IoU
    high_iou_m = feat_m[iou_m > 0.5]  # 남성 high IoU (타겟)
    low_iou_f = feat_f[iou_f < 0.3]   # 여성 low IoU (negative)

    # 여성 high IoU → 남성 high IoU (positive)
    # 여성 high IoU → 여성 low IoU (negative)
    loss = InfoNCE(anchor=high_iou_f, positive=high_iou_m, negative=low_iou_f)
    return loss
```

**장점**:
- AP와 직접 연관된 IoU를 최적화
- High-quality detection을 명시적으로 학습

**구현 과제**:
- IoU 계산을 위한 GT 매칭 필요
- Hungarian matching과 연동

### 3.2 Approach B: Hierarchical Feature Contrastive (HFC)

**핵심 아이디어**: DETR decoder의 여러 레이어에서 multi-scale contrastive learning

```
DETR Decoder Structure:
Layer 1 (Coarse) → 전체적 위치 정보
Layer 2 (...)    → 중간 수준 특징
...
Layer 6 (Fine)   → 세밀한 특징, 최종 detection

제안: 각 레이어에서 cross-gender contrastive 수행
      → 모든 수준에서 성별 불변 특징 학습
```

```python
class HierarchicalContrastive:
    def forward(self, decoder_layers_f, decoder_layers_m):
        losses = []
        for i, (layer_f, layer_m) in enumerate(zip(decoder_layers_f, decoder_layers_m)):
            # 레이어별 가중치: 깊은 레이어일수록 중요
            weight = 0.5 + 0.1 * i
            loss = cross_gender_contrastive(layer_f, layer_m)
            losses.append(weight * loss)
        return sum(losses)
```

**장점**:
- 다양한 추상화 수준에서 공정성 학습
- Localization과 classification 모두 개선 가능

### 3.3 Approach C: Prototype-Guided Alignment (PGA)

**핵심 아이디어**: 성별별 prototype(대표 feature)를 학습하고 정렬

```
학습 과정:
1. 각 배치에서 남녀 평균 feature 계산
2. EMA로 global prototype 업데이트
3. 여성 샘플을 남성 prototype 방향으로 이동
4. 동시에 개별 다양성 유지 (collapse 방지)

손실 함수:
L = L_prototype + L_diversity + L_detection

L_prototype: 여성 feature → 남성 prototype 거리 최소화
L_diversity: 여성 feature 간 분산 유지
L_detection: DETR detection loss
```

```python
class PrototypeAlignment:
    def __init__(self, feature_dim, momentum=0.99):
        self.prototype_f = zeros(feature_dim)  # 여성 prototype
        self.prototype_m = zeros(feature_dim)  # 남성 prototype
        self.momentum = momentum

    @torch.no_grad()
    def update_prototypes(self, feat_f, feat_m):
        self.prototype_f = self.momentum * self.prototype_f + (1-self.momentum) * feat_f.mean(0)
        self.prototype_m = self.momentum * self.prototype_m + (1-self.momentum) * feat_m.mean(0)

    def alignment_loss(self, feat_f):
        # 여성 feature를 남성 prototype으로 끌어당김
        return F.mse_loss(feat_f.mean(0), self.prototype_m.detach())

    def diversity_loss(self, feat_f):
        # 분산 유지 (collapse 방지)
        return -feat_f.var(dim=0).mean()
```

### 3.4 Approach D: Score-Feature Joint Contrastive (SFJC)

**핵심 아이디어**: Detection score와 feature를 동시에 contrastive learning

```
기존 문제: Feature만 정렬 → Score 개선 안 됨
          Score만 정렬 → Localization 개선 안 됨

해결: Joint embedding space에서 동시 정렬

Joint Embedding = Projection(concat(Feature, Score))
```

```python
class ScoreFeatureJoint:
    def __init__(self, feature_dim=256, score_dim=1, proj_dim=128):
        self.feature_proj = Linear(feature_dim, proj_dim // 2)
        self.score_proj = Linear(score_dim, proj_dim // 2)
        self.joint_proj = Linear(proj_dim, proj_dim)

    def forward(self, feature, score):
        feat_emb = self.feature_proj(feature)
        score_emb = self.score_proj(score.unsqueeze(-1))
        joint = torch.cat([feat_emb, score_emb], dim=-1)
        return F.normalize(self.joint_proj(joint), dim=-1)
```

### 3.5 Approach E: Hard Sample Mining + Curriculum (HSM-CL)

**핵심 아이디어**: 어려운 샘플에 집중 + 점진적 난이도 증가

```
Hard Sample 정의:
- 여성: Detection score가 낮은 샘플 (개선 필요)
- 남성: Detection score가 높은 샘플 (타겟으로 활용)

Curriculum 전략:
Epoch 0-10:  Easy pairs (score 차이 작은 샘플들)
Epoch 10-20: Medium pairs
Epoch 20-30: Hard pairs (score 차이 큰 샘플들)
```

```python
def hard_sample_mining(scores_f, scores_m, epoch, max_epochs):
    # 난이도 스케줄
    difficulty = min(epoch / (max_epochs * 0.7), 1.0)  # 0→1

    # 여성: 하위 k%만 선택 (낮은 score)
    k_f = int((1 - 0.7 * difficulty) * len(scores_f))  # 100% → 30%
    hard_f_idx = scores_f.argsort()[:k_f]

    # 남성: 상위 k%만 선택 (높은 score)
    k_m = int((1 - 0.7 * difficulty) * len(scores_m))
    hard_m_idx = scores_m.argsort(descending=True)[:k_m]

    return hard_f_idx, hard_m_idx
```

---

## 4. 추천 연구 우선순위

### 4.1 1차 추천: IoU-Aware + Prototype 결합 (A + C)

**이유**:
1. AP 개선의 핵심인 IoU를 직접 다룸
2. Prototype으로 안정적인 정렬 가능
3. 기존 DINO의 장점(안정성) + 새로운 목표(IoU) 결합

**구현 복잡도**: 중간
**예상 효과**: AP Gap 0.106 → 0.08 목표 (25% 개선)

### 4.2 2차 추천: Hierarchical + Hard Mining (B + E)

**이유**:
1. Multi-scale feature로 전체적인 품질 향상
2. Hard mining으로 어려운 샘플 집중 공략
3. Curriculum으로 안정적 학습

**구현 복잡도**: 높음 (DETR 내부 수정 필요)
**예상 효과**: AP Gap, AR Gap 동시 개선

### 4.3 3차 추천: Score-Feature Joint (D)

**이유**:
1. 구현이 간단
2. 기존 방법의 한계(분리된 최적화) 직접 해결
3. 빠른 실험 가능

**구현 복잡도**: 낮음
**예상 효과**: 중간 수준 개선

---

## 5. 구체적 실험 계획

### Phase 1: Baseline 재현 및 분석 (1주)
- [ ] 7th WGAN-GD 결과 재현
- [ ] Contrastive 1st 결과 재현
- [ ] 상세 분석: 어떤 샘플에서 실패하는지?
- [ ] IoU 분포 분석: 여성 vs 남성 high-IoU 비율

### Phase 2: IoU-Aware Contrastive 구현 (2주)
- [ ] DETR matcher에서 IoU 정보 추출
- [ ] IoU 기반 positive/negative sampling 구현
- [ ] 하이퍼파라미터 탐색

### Phase 3: Prototype Alignment 결합 (1주)
- [ ] EMA prototype 모듈 구현
- [ ] Diversity loss 추가
- [ ] IoU-Aware + Prototype 통합

### Phase 4: 성능 평가 및 분석 (1주)
- [ ] Test set 평가
- [ ] AP/AR breakdown 분석
- [ ] Ablation study

### Phase 5: 논문 작성 (2주)
- [ ] 실험 결과 정리
- [ ] 비교 분석
- [ ] 논문 초안

---

## 6. 핵심 코드 구조

```
train_faap_contrastive_iou.py
├── IoUAwareProjectionHead      # IoU 조건부 projection
├── PrototypeBank               # EMA prototype 관리
├── IoUAwareContrastiveLoss     # 핵심 손실 함수
├── HardSampleMiner             # 어려운 샘플 선택
└── main()                      # 학습 루프
```

---

## 7. 예상 결과 및 목표

### 7.1 정량적 목표

| 메트릭 | Baseline | 7th | 목표 | 개선율 |
|--------|----------|-----|------|--------|
| AP Gap | 0.106 | 0.106 | **0.08** | 25%↓ |
| AR Gap | 0.0081 | 0.0032 | **0.002** | 40%↓ |
| Female AP | 0.404 | 0.408 | **0.42** | 4%↑ |
| Male AP | 0.511 | 0.514 | 0.51 | 유지 |

### 7.2 정성적 목표

1. **AP Gap 감소**: 현재까지 어떤 방법도 성공하지 못한 목표
2. **이론적 기여**: IoU-aware contrastive learning의 새로운 프레임워크
3. **실용적 가치**: 실제 detection 시스템에 적용 가능한 fairness 개선

---

## 8. 참고문헌

1. **SimCLR**: A Simple Framework for Contrastive Learning of Visual Representations
2. **DINO**: Emerging Properties in Self-Supervised Vision Transformers
3. **SupCon**: Supervised Contrastive Learning
4. **Focal Loss**: Focal Loss for Dense Object Detection
5. **DETR**: End-to-End Object Detection with Transformers

---

*작성일: 2026-01-19*
*다음 단계: Phase 1 시작 (Baseline 재현)*
