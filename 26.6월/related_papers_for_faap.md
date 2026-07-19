# FAAP 연구에 적용 가능한 논문 모음

## 현재 연구 특성
- **태스크**: Object Detection에서의 Gender Fairness
- **방법**: Perturbation Generator + Cross-gender InfoNCE
- **Backbone**: DETR (frozen)
- **핵심 아이디어**: 성별 간 feature를 가깝게, 같은 성별 간 feature를 멀게

---

## 1. Fair Contrastive Learning (핵심 관련)

### 1.1 FSCL: Fair Contrastive Learning for Facial Attribute Classification
- **학회**: CVPR 2022
- **저자**: Sungho Park, Jewook Lee, Pilhyeon Lee, Sunhee Hwang, Dohyung Kim, Hyeran Byun
- **핵심 아이디어**:
  - Fair Supervised Contrastive Loss (FSCL): SupCon에 fairness penalty 추가
  - Group-wise Normalization: 그룹 간 intra-class compactness 불균형 해소
  - Equalized Odds: 30.5 → 6.5 개선
- **적용 가능성**: Group-wise Normalization 개념 참고 가능
- **링크**: [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Park_Fair_Contrastive_Learning_for_Facial_Attribute_Classification_CVPR_2022_paper.html) | [GitHub](https://github.com/sungho-CoolG/FSCL) | [arXiv](https://arxiv.org/abs/2203.16209)

### 1.2 SupCon: Supervised Contrastive Learning
- **학회**: NeurIPS 2020
- **저자**: Prannay Khosla et al. (Google Research)
- **핵심 아이디어**:
  - 같은 클래스 샘플을 positive로 사용하는 supervised contrastive loss
  - Cross-entropy 대비 robustness 및 정확도 향상
  - ImageNet ResNet-200에서 81.4% top-1 accuracy
- **적용 가능성**: 현재 InfoNCE 구현의 기반 이론
- **링크**: [Paper](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html) | [arXiv](https://arxiv.org/abs/2004.11362)

### 1.3 FALCON: Fairness Learning via Contrastive Attention
- **학회**: CVPR 2025
- **저자**: Thanh-Dat Truong, Utsav Prabhu, Bhiksha Raj, Jackson Cothren, Khoa Luu
- **핵심 아이디어**: Continual semantic segmentation에서 contrastive attention을 통한 fairness
- **적용 가능성**: Attention 기반 fairness 학습 기법 참고
- **링크**: [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Truong_FALCON_Fairness_Learning_via_Contrastive_Attention_Approach_to_Continual_Semantic_CVPR_2025_paper.pdf)

---

## 2. Adversarial Perturbation for Fairness (직접 관련)

### 2.1 FAAP: Fairness-Aware Adversarial Perturbation
- **학회**: CVPR 2022
- **저자**: Wang et al.
- **핵심 아이디어**:
  - 배포된 모델을 수정하지 않고 입력 perturbation으로 fairness 달성
  - Gender, ethnicity 등 민감한 속성에 대해 모델을 "blind"하게 함
- **적용 가능성**: **현재 연구와 직접 관련** - 동일한 perturbation 기반 접근
- **링크**: [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Fairness-Aware_Adversarial_Perturbation_Towards_Bias_Mitigation_for_Deployed_Deep_Models_CVPR_2022_paper.pdf) | [arXiv](https://arxiv.org/abs/2203.01584)

### 2.2 Adversarial Debiasing (Mitigating Unwanted Biases)
- **학회**: AIES 2018
- **핵심 아이디어**:
  - Classifier와 adversary를 동시에 학습
  - Adversary가 bias 활용을 시도하고, classifier가 이를 억제
- **적용 가능성**: Adversarial learning 프레임워크 참고
- **링크**: [Paper](https://dl.acm.org/doi/pdf/10.1145/3278721.3278779)

### 2.3 ALFA: Adversarial Latent Feature Augmentation for Fairness
- **핵심 아이디어**:
  - Adversarial attack과 data augmentation을 latent space에서 결합
  - Hyperplane rotation을 통한 fairness 향상
- **적용 가능성**: Latent space에서의 fairness augmentation
- **링크**: [OpenReview](https://openreview.net/forum?id=eFS9Pm7bsM)

### 2.4 Intra-Processing Methods for Debiasing Neural Networks
- **학회**: NeurIPS 2020
- **핵심 아이디어**:
  - Random perturbation, adversarial fine-tuning, layer-wise optimization
  - 모델 재훈련 없이 fine-tuning으로 debiasing
- **적용 가능성**: Fine-tuning 기반 debiasing 기법
- **링크**: [Paper](https://proceedings.neurips.cc/paper/2020/file/1d8d70dddf147d2d92a634817f01b239-Paper.pdf) | [arXiv](https://arxiv.org/abs/2006.08564)

---

## 3. Object Detection Fairness (태스크 관련)

### 3.1 Fairness in Autonomous Driving: Object Detection under Challenging Weather
- **저자**: 2024
- **핵심 아이디어**:
  - DETR (ResNet-50)를 사용한 pedestrian detection fairness 분석
  - 어두운 환경에서 어두운 피부톤의 성능 저하 확인
  - Transformer attention이 피부 패치보다 전체 사람을 봄
- **적용 가능성**: **DETR 기반 fairness 분석의 직접적 참고**
- **링크**: [arXiv](https://arxiv.org/abs/2406.00219)

### 3.2 Beyond Overall Accuracy: Pose- and Occlusion-driven Fairness in Pedestrian Detection
- **핵심 아이디어**:
  - Pose(다리 상태, 팔꿈치, 몸 방향)와 occlusion에 따른 detection bias 분석
  - Lateral view, parallel legs에서 bias 발견
- **적용 가능성**: Detection에서의 fairness 평가 기준 참고
- **링크**: [arXiv](https://arxiv.org/abs/2509.26166)

### 3.3 Predictive Inequity in Object Detection
- **핵심 아이디어**:
  - BDD100K에 Fitzpatrick skin tone 주석 추가
  - Light skin이 dark skin보다 일관되게 높은 AP
  - 시간대나 occlusion으로 설명되지 않는 disparity
- **적용 가능성**: **Object detection에서의 demographic bias 분석 참고**
- **링크**: [arXiv](https://arxiv.org/abs/1902.11097)

### 3.4 FairMOT: Fairness of Detection and Re-ID in Multi-Object Tracking
- **학회**: IJCV 2021
- **핵심 아이디어**:
  - Detection과 Re-ID 간의 "fairness" (task balance)
  - Anchor-free detection (CenterNet) 기반
  - 두 task에 동등한 비중 부여
- **적용 가능성**: Multi-task learning에서의 balance 참고
- **링크**: [Paper](https://link.springer.com/article/10.1007/s11263-021-01513-4) | [GitHub](https://github.com/ifzhang/FairMOT) | [arXiv](https://arxiv.org/abs/2004.01888)

---

## 4. Feature Disentanglement for Fairness

### 4.1 FarconVAE: Learning Fair Representation via Distributional Contrastive Disentanglement
- **핵심 아이디어**:
  - Non-sensitive representation과 sensitive representation 분리
  - Swap-recon: 다른 샘플의 non-sensitive representation으로 교체 후 재구성
  - Fairness와 domain generalization 모두에 효과적
- **적용 가능성**: **Disentanglement + Contrastive의 결합 기법**
- **링크**: [arXiv](https://arxiv.org/abs/2206.08743)

### 4.2 FairSAD: Fair Graph Representation via Sensitive Attribute Disentanglement
- **학회**: ACM Web Conference 2024
- **핵심 아이디어**:
  - Sensitive attribute를 독립 component로 분리
  - Masking을 통해 fairness 달성
- **적용 가능성**: Attribute disentanglement 기법
- **링크**: [Paper](https://dl.acm.org/doi/10.1145/3589334.3645532) | [GitHub](https://github.com/zzoomd/fairsad) | [arXiv](https://arxiv.org/abs/2405.07011)

### 4.3 FFVAE: Flexibly Fair Representation Learning by Disentanglement
- **핵심 아이디어**:
  - Multi-attribute fair representation learning
  - Sensitive attributes를 label로 사용하여 factorized latent structure 유도
- **적용 가능성**: VAE 기반 fair disentanglement
- **링크**: [arXiv](https://arxiv.org/pdf/1906.02589)

### 4.4 DAB-GNN: Disentangling, Amplifying, and Debiasing
- **핵심 아이디어**:
  - Attribute bias, structure bias, potential bias로 3-way 분리
  - Bias Contrast Optimizer (BCO)와 Fairness Harmonizer (FH) 사용
- **적용 가능성**: Multi-source bias disentanglement
- **링크**: [arXiv](https://arxiv.org/abs/2408.12875)

---

## 5. Wasserstein Distance for Fairness (현재 사용 중)

### 5.1 Wasserstein-based Fairness Interpretability Framework
- **학회**: Machine Learning (Springer) 2022
- **핵심 아이디어**:
  - Wasserstein metric으로 sub-population 간 model bias 측정
  - Transport theory를 통한 bias decomposition 및 설명
- **적용 가능성**: **현재 Wasserstein loss의 이론적 배경**
- **링크**: [Paper](https://link.springer.com/article/10.1007/s10994-022-06213-9) | [arXiv](https://arxiv.org/abs/2011.03156)

### 5.2 FairWASP: Fast and Optimal Fair Wasserstein Pre-processing
- **핵심 아이디어**:
  - 원본 데이터를 수정하지 않고 sample-level weight 학습
  - Wasserstein distance 최소화하면서 demographic parity 달성
- **적용 가능성**: Wasserstein 기반 pre-processing 기법
- **링크**: [arXiv](https://arxiv.org/abs/2311.00109)

### 5.3 Distributionally Fair Stochastic Optimization using Wasserstein Distance
- **핵심 아이디어**:
  - Wasserstein distance를 사용한 distributional fairness 최적화
  - Support mismatch에서도 의미 있는 metric
- **적용 가능성**: Optimization 관점의 Wasserstein fairness
- **링크**: [Paper](https://optimization-online.org/wp-content/uploads/2024/02/Distributional_Fairness_Project_OPT.pdf)

---

## 6. Normalization for Fairness

### 6.1 FairAdaBN: Adaptive Batch Normalization for Fairness
- **핵심 아이디어**:
  - Batch Normalization을 sensitive attribute에 adaptive하게
  - 각 subgroup에 대해 별도의 normalization block
  - Feature map alignment를 통한 unfairness 완화
- **적용 가능성**: **Group-wise normalization의 실용적 구현**
- **링크**: [arXiv](https://arxiv.org/abs/2303.08325)

### 6.2 Group Normalization
- **학회**: ECCV 2018
- **저자**: Yuxin Wu, Kaiming He (FAIR)
- **핵심 아이디어**:
  - Batch size에 독립적인 normalization
  - Channel을 그룹으로 나누어 정규화
- **적용 가능성**: Small batch에서의 normalization 기법
- **링크**: [Paper](https://arxiv.org/abs/1803.08494)

---

## 7. Hard Negative Mining & Sampling Strategies

### 7.1 SCHaNe: Supervised Contrastive Learning with Hard Negative Samples
- **핵심 아이디어**:
  - Fine-tuning 단계에서 hard negative sampling
  - Negative를 positive와의 dissimilarity로 weighting
  - ImageNet-1k에서 86.14% accuracy SOTA
- **적용 가능성**: **현재 InfoNCE에 hard negative mining 추가 가능**
- **링크**: [arXiv](https://arxiv.org/abs/2209.00078)

### 7.2 Curriculum Learning for Hard Negative Mining
- **핵심 아이디어**:
  - Easy-to-hard 순서로 negative 학습
  - False negative에 대한 regularization
  - 수렴 속도 향상
- **적용 가능성**: Curriculum learning 기반 negative mining
- **링크**: [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S002002552400447X)

### 7.3 X-Sample Contrastive Loss
- **학회**: 2024
- **핵심 아이디어**:
  - InfoNCE에 soft cross-sample similarity 추가
  - Multiple positives 지원
  - Soft targets를 사용한 distillation
- **적용 가능성**: InfoNCE 확장 기법
- **링크**: [arXiv](https://arxiv.org/abs/2407.18134)

---

## 8. Domain Adaptation & Cross-Domain Learning

### 8.1 CDCL: Cross-domain Contrastive Learning for UDA
- **핵심 아이디어**:
  - Contrastive learning으로 domain discrepancy 감소
  - Domain-invariant feature alignment
- **적용 가능성**: Cross-gender를 cross-domain처럼 취급
- **링크**: [arXiv](https://arxiv.org/abs/2106.05528)

### 8.2 Multi-Source Domain Adaptation via Supervised Contrastive Learning
- **학회**: BMVC 2021
- **핵심 아이디어**:
  - SCL이 자연스럽게 domain-invariant feature 학습
  - 같은 클래스를 당기고 다른 클래스를 밀면서 domain alignment
- **적용 가능성**: Multi-source fairness learning
- **링크**: [Paper](https://www.bmvc2021-virtualconference.com/assets/papers/0699.pdf)

---

## 9. Datasets & Benchmarks

### 9.1 BDD100K with Demographic Annotations
- 100K+ driving videos, skin tone 주석 추가
- Light skin vs dark skin detection disparity 분석
- **링크**: [BAIR Blog](https://bair.berkeley.edu/blog/2018/05/30/bdd/)

### 9.2 CelebA / UTKFace
- CelebA: 200K+ facial images, 40 attributes
- UTKFace: Age, gender, ethnicity labels
- **한계**: Race imbalance (White 편향)
- **링크**: [FairFace Paper](https://arxiv.org/pdf/1908.04913)

### 9.3 Attribute Annotation for Autonomous Driving Datasets
- **학회**: Journal of Big Data 2024
- **핵심 아이디어**:
  - BDD100K, nuImages에 age, sex, skin tone 주석
  - 90K+ people, 50K+ vehicles 주석
  - 아동 미검출률이 성인보다 20.14% 높음
- **적용 가능성**: **Fairness 평가 데이터셋 및 기준**
- **링크**: [Paper](https://link.springer.com/article/10.1186/s40537-024-00976-9)

---

## 10. Surveys & Comprehensive Reviews

### 10.1 Fairness and Bias Mitigation in Computer Vision: A Survey
- **연도**: 2024
- **핵심 내용**:
  - Pre-processing, in-processing, post-processing 분류
  - Distributional methods, algorithmic approaches 정리
- **링크**: [arXiv](https://arxiv.org/abs/2408.02464)

### 10.2 Gender Bias in NLP and Computer Vision: A Comparative Survey
- **학회**: ACM Computing Surveys
- **핵심 내용**:
  - NLP, CV, visual-linguistic 모델의 gender bias
  - 방법론의 cross-disciplinary 적용
- **링크**: [Paper](https://dl.acm.org/doi/10.1145/3700438)

### 10.3 Racial Bias within Face Recognition: A Survey
- **학회**: ACM Computing Surveys
- **핵심 내용**: Face recognition에서의 racial bias 종합 정리
- **링크**: [Paper](https://dl.acm.org/doi/10.1145/3705295)

---

## 적용 우선순위 추천

### 🔴 높은 우선순위 (직접 적용 가능)
1. **SCHaNe** - Hard negative mining 추가
2. **FairAdaBN** - Group-wise normalization
3. **FAAP (CVPR 2022)** - 동일 패러다임 비교

### 🟡 중간 우선순위 (아이디어 참고)
4. **FarconVAE** - Disentanglement + Contrastive
5. **Curriculum Hard Negative Mining** - 학습 안정성
6. **Wasserstein Fairness Framework** - 이론적 보강

### 🟢 낮은 우선순위 (참고용)
7. **FSCL** - Classification용이지만 normalization 참고
8. **FairMOT** - Multi-task balance 참고
9. **Domain Adaptation** - Cross-gender를 cross-domain으로 해석

---

## 현재 연구와의 비교표

| 논문 | 태스크 | Backbone | 학습 대상 | Contrastive 방식 |
|------|--------|----------|-----------|------------------|
| **FAAP (현재)** | Detection | DETR (frozen) | Generator | Cross-gender positive |
| FSCL | Classification | ResNet | Encoder | Same-class positive |
| FAAP (CVPR22) | Classification | Various (frozen) | Perturbation | Adversarial |
| FarconVAE | Various | VAE | Encoder | Disentanglement |
| SCHaNe | Classification | ResNet | Encoder | Hard negative |

---

## 11. 🎯 현재 연구에 구체적 적용 방법

### 11.1 Hard Negative Mining (SCHaNe) - **최우선 추천**

**현재 문제점**: 모든 same-gender 샘플을 동일한 가중치로 negative 처리

**적용 방법**: Hard negative에 더 높은 가중치 부여

```python
class CrossGenderInfoNCELossWithHardNegative(nn.Module):
    """
    SCHaNe 스타일 hard negative mining 적용
    """
    def __init__(self, temperature=0.07, beta=0.5):
        super().__init__()
        self.temperature = temperature
        self.beta = beta  # hard negative 강도 조절

    def forward(self, proj_f, proj_m):
        # 기존 similarity 계산
        sim_f2m = torch.mm(proj_f, proj_m.t()) / self.temperature  # positive
        sim_f2f = torch.mm(proj_f, proj_f.t()) / self.temperature  # negative

        # Hard negative weighting: similarity가 높은 negative에 더 높은 가중치
        # (positive와 헷갈리기 쉬운 negative가 hard negative)
        neg_weights = torch.exp(self.beta * sim_f2f)  # hard negative 강조
        neg_weights = neg_weights / neg_weights.sum(dim=1, keepdim=True)  # normalize

        # Weighted negative
        weighted_neg = (neg_weights * torch.exp(sim_f2f)).sum(dim=1)

        # InfoNCE with hard negative
        pos_exp = torch.exp(sim_f2m).sum(dim=1)
        loss = -torch.log(pos_exp / (pos_exp + weighted_neg)).mean()

        return loss
```

**기대 효과**:
- 성별 간 구분이 어려운 샘플에 집중 → 더 robust한 fairness
- 수렴 속도 향상

---

### 11.2 Group-wise Feature Normalization (FSCL/FairAdaBN) - **높은 우선순위**

**현재 문제점**: 여성/남성 그룹 간 feature 분포 불균형 가능

**적용 방법**: Projection 전 성별별 normalization

```python
class GenderAwareProjectionHead(nn.Module):
    """
    FairAdaBN 스타일: 성별별 별도의 normalization
    """
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=128):
        super().__init__()
        # 성별별 별도의 BatchNorm
        self.bn_female = nn.BatchNorm1d(input_dim)
        self.bn_male = nn.BatchNorm1d(input_dim)

        # 공유 projection layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, gender_mask):
        """
        x: (batch, num_queries, feature_dim)
        gender_mask: (batch,) - True for female, False for male
        """
        pooled = x.mean(dim=1)  # (batch, feature_dim)

        # 성별별 normalization
        normalized = torch.zeros_like(pooled)
        if gender_mask.any():
            normalized[gender_mask] = self.bn_female(pooled[gender_mask])
        if (~gender_mask).any():
            normalized[~gender_mask] = self.bn_male(pooled[~gender_mask])

        proj = self.net(normalized)
        return F.normalize(proj, dim=-1, p=2)
```

**대안: Instance Normalization 방식**
```python
class GroupWiseInstanceNorm(nn.Module):
    """
    FSCL 스타일: 그룹 내 분산을 정규화하여 그룹 간 compactness 균형
    """
    def forward(self, feat_f, feat_m):
        # 각 그룹 내에서 mean/std 정규화
        feat_f_norm = (feat_f - feat_f.mean(dim=0)) / (feat_f.std(dim=0) + 1e-6)
        feat_m_norm = (feat_m - feat_m.mean(dim=0)) / (feat_m.std(dim=0) + 1e-6)
        return feat_f_norm, feat_m_norm
```

**기대 효과**:
- 그룹 간 feature 분포 정렬
- Intra-class compactness 균형

---

### 11.3 Curriculum Learning for Negatives - **중간 우선순위**

**현재 문제점**: 학습 초기에 어려운 negative로 인한 불안정

**적용 방법**: Easy-to-hard negative curriculum

```python
class CurriculumInfoNCELoss(nn.Module):
    """
    학습 진행에 따라 hard negative 비중 증가
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, proj_f, proj_m, epoch, total_epochs):
        # Curriculum: 초기에는 쉬운 negative, 후기에는 hard negative
        curriculum_beta = min(1.0, epoch / (total_epochs * 0.5))  # 50% 지점에서 최대

        sim_f2m = torch.mm(proj_f, proj_m.t()) / self.temperature
        sim_f2f = torch.mm(proj_f, proj_f.t()) / self.temperature

        # 대각선 마스킹
        mask = torch.eye(proj_f.size(0), device=proj_f.device, dtype=torch.bool)
        sim_f2f = sim_f2f.masked_fill(mask, float('-inf'))

        # Curriculum-based hard negative weighting
        if curriculum_beta > 0:
            neg_weights = F.softmax(curriculum_beta * sim_f2f, dim=1)
            weighted_neg_logsumexp = torch.log((neg_weights * torch.exp(sim_f2f)).sum(dim=1))
        else:
            weighted_neg_logsumexp = torch.logsumexp(sim_f2f, dim=1)

        pos_logsumexp = torch.logsumexp(sim_f2m, dim=1)
        all_logsumexp = torch.logsumexp(
            torch.stack([pos_logsumexp, weighted_neg_logsumexp], dim=1), dim=1
        )

        return -(pos_logsumexp - all_logsumexp).mean()
```

**기대 효과**:
- 학습 초기 안정성 향상
- 점진적으로 어려운 케이스 학습

---

### 11.4 Feature Disentanglement (FarconVAE) - **중간 우선순위**

**현재 문제점**: Generator가 성별 정보와 detection 정보를 함께 학습

**적용 방법**: Gender-invariant와 gender-specific feature 분리

```python
class DisentangledProjectionHead(nn.Module):
    """
    FarconVAE 스타일: sensitive/non-sensitive feature 분리
    """
    def __init__(self, input_dim=256, hidden_dim=256, output_dim=128):
        super().__init__()
        # Gender-invariant branch (fairness용)
        self.invariant_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        # Gender-specific branch (disentanglement 확인용)
        self.specific_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # binary gender classification
        )

    def forward(self, x):
        pooled = x.mean(dim=1)
        z_inv = F.normalize(self.invariant_head(pooled), dim=-1)
        z_spec = self.specific_head(pooled)
        return z_inv, z_spec


class DisentanglementLoss(nn.Module):
    """
    z_inv가 성별 정보를 포함하지 않도록 adversarial loss
    """
    def __init__(self, lambda_adv=0.1):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.gender_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, z_inv, gender_labels):
        # Gradient reversal: z_inv가 성별 예측 못하게
        gender_pred = self.gender_classifier(z_inv)

        # Adversarial: 성별 예측 정확도를 낮추는 방향
        ce_loss = F.cross_entropy(gender_pred, gender_labels)
        entropy = -(F.softmax(gender_pred, dim=1) * F.log_softmax(gender_pred, dim=1)).sum(dim=1).mean()

        # 성별 예측 어렵게 + 엔트로피 최대화
        return -ce_loss + self.lambda_adv * entropy
```

**기대 효과**:
- Detection feature에서 성별 정보 명시적 제거
- 더 interpretable한 모델

---

### 11.5 Wasserstein Loss 개선 - **낮은 우선순위 (이미 구현됨)**

**현재 구현**: Score-level 1D Wasserstein (단방향)

**개선 방법**: Feature-level Wasserstein 추가

```python
def sliced_wasserstein_distance(feat_f, feat_m, num_projections=50):
    """
    Feature-level Sliced Wasserstein Distance
    고차원 feature 분포 정렬에 효과적
    """
    dim = feat_f.size(1)

    # Random projections
    projections = torch.randn(num_projections, dim, device=feat_f.device)
    projections = F.normalize(projections, dim=1)

    # Project features
    proj_f = torch.mm(feat_f, projections.t())  # (N_f, num_proj)
    proj_m = torch.mm(feat_m, projections.t())  # (N_m, num_proj)

    # 1D Wasserstein for each projection
    total_dist = 0
    for i in range(num_projections):
        sorted_f = proj_f[:, i].sort().values
        sorted_m = proj_m[:, i].sort().values

        # Interpolate to same size
        k = max(len(sorted_f), len(sorted_m))
        sorted_f = _resize_sorted(sorted_f, k)
        sorted_m = _resize_sorted(sorted_m, k)

        total_dist += (sorted_f - sorted_m).abs().mean()

    return total_dist / num_projections
```

**기대 효과**:
- Score 외에 feature 분포도 정렬
- 더 근본적인 fairness 달성

---

### 11.6 Multi-view Augmentation (SimCLR 확장) - **낮은 우선순위**

**현재 구현**: Single augmentation (ColorJitter)

**개선 방법**: 두 개의 다른 augmentation view 생성

```python
class DualViewSimCLRAugmentation(nn.Module):
    """
    SimCLR 원본 스타일: 같은 이미지에서 두 개의 다른 view 생성
    """
    def __init__(self):
        super().__init__()
        self.aug1 = T.Compose([
            T.ColorJitter(0.4, 0.4, 0.4, 0.1),
            T.RandomGrayscale(p=0.2),
        ])
        self.aug2 = T.Compose([
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
            # GaussianBlur는 detection 성능 저하 우려로 제외
        ])

    def forward(self, x):
        # ... denormalize, apply aug, renormalize ...
        view1 = self._apply(x, self.aug1)
        view2 = self._apply(x, self.aug2)
        return view1, view2
```

**추가 Loss**: 같은 이미지의 두 view도 positive로 처리
```python
# Cross-gender InfoNCE + Self-consistency
loss_cross_gender = infonce_loss(proj_f, proj_m)
loss_self_view = self_consistency_loss(proj_f_v1, proj_f_v2)  # 같은 이미지 두 view
total = loss_cross_gender + 0.5 * loss_self_view
```

---

## 12. 🔬 실험 설계 제안

### Phase 1: Baseline 확립 (현재)
- `train_faap_simclr_infonce.py` 실행
- AP Gap, AR Gap 측정

### Phase 2: Hard Negative Mining 추가
```bash
# 새 파일: train_faap_simclr_hard_negative.py
python train_faap_simclr_hard_negative.py --hard_neg_beta 0.5
```

### Phase 3: Group-wise Normalization 추가
```bash
# 새 파일: train_faap_simclr_groupnorm.py
python train_faap_simclr_groupnorm.py --use_group_norm
```

### Phase 4: 조합 실험
```bash
# Hard Negative + Group Norm
python train_faap_simclr_combined.py --hard_neg_beta 0.5 --use_group_norm
```

### 평가 지표
| 지표 | 목표 | 측정 방법 |
|------|------|-----------|
| AP Gap (M-F) | < 0.09 | eval 스크립트 |
| AR Gap (M-F) | < 0.05 | eval 스크립트 |
| Female AP | > 0.41 | eval 스크립트 |
| Overall AP | ≥ baseline | 성능 저하 없어야 함 |

---

## 13. 📊 예상 효과 요약

| 기법 | 구현 난이도 | 예상 AP Gap 개선 | 주의사항 |
|------|-------------|------------------|----------|
| Hard Negative Mining | ⭐⭐ | 10-15% | beta 튜닝 필요 |
| Group-wise Norm | ⭐⭐ | 5-10% | batch size 의존 |
| Curriculum Learning | ⭐⭐⭐ | 5-10% | 수렴 안정성 |
| Disentanglement | ⭐⭐⭐⭐ | 10-20% | 추가 loss 복잡 |
| Feature Wasserstein | ⭐⭐⭐ | 5-10% | 계산량 증가 |
| Dual-view Aug | ⭐⭐ | 3-5% | 메모리 2배 |

---

## 14. 🚀 즉시 적용 가능한 코드 변경

### 14.1 `train_faap_simclr_infonce.py`에 Hard Negative 추가

```python
# CrossGenderInfoNCELoss 클래스 수정
def forward(self, proj_f, proj_m, hard_neg_beta=0.0):
    # ... 기존 코드 ...

    # Hard negative weighting 추가
    if hard_neg_beta > 0:
        neg_weights_f = F.softmax(hard_neg_beta * sim_f2f, dim=1)
        sim_f2f_weighted = torch.log((neg_weights_f * torch.exp(sim_f2f)).sum(dim=1, keepdim=True))
    else:
        sim_f2f_weighted = sim_f2f

    # ... 나머지 코드 ...
```

### 14.2 argparse에 추가할 인자

```python
parser.add_argument("--hard_neg_beta", type=float, default=0.0,
                    help="Hard negative mining strength (0=off, 0.5=medium, 1.0=strong)")
parser.add_argument("--use_group_norm", action="store_true",
                    help="Use gender-wise normalization in projection head")
parser.add_argument("--curriculum", action="store_true",
                    help="Use curriculum learning for hard negatives")
```

---

*생성일: 2025-01-20*
*목적: FAAP 연구에 적용 가능한 관련 논문 수집 및 구체적 적용 방법*
*최종 수정: 구체적 적용 방법 및 코드 예시 추가*
