# 4. MMD 계열 (Maximum Mean Discrepancy)

**파일**: `train_faap_mmd_1st.py`, `train_faap_mmd_2nd.py`
**핵심**: Discriminator 없이 Gaussian Kernel 기반 MMD로 Feature 분포 직접 정렬

---

## 4.1 파이프라인

```
┌───────────────────────────────────────────────────────────────┐
│                  MMD Fairness Pipeline                         │
│                                                               │
│  [Female Image]        [Male Image]                           │
│       │                     │                                 │
│       ▼                     ▼                                 │
│  Generator(G)          Generator(G)                           │
│       │                     │                                 │
│  Perturbed_f           Perturbed_m                            │
│       │                     │                                 │
│       ▼                     ▼                                 │
│  ┌──────────────────────────────────┐                         │
│  │           Frozen DETR            │                         │
│  └────┬─────────┬──────┬───────────┘                          │
│       │         │      │                                      │
│  Detection  Features  Scores                                  │
│  Outputs  (B×100×256) (matched)                               │
│       │         │       │                                     │
│  L_det   GAP Pooling   L_w                                    │
│       │  (B×256)  │    (Wasserstein)                          │
│       │    feat_f  feat_m                                     │
│       │       │      │                                        │
│       │   MMD (RBF Kernel)                                    │
│       │   σ ∈ {1,2,4,8,16}                                    │
│       │       │                                               │
│       └───────┴──────────────┐                                │
│                              ▼                                │
│  L = λ_fair·L_MMD + β·L_det + λ_w·L_w                        │
│                                                               │
│  ※ Discriminator 없음 (Kernel-based, GAN-Free)                │
└───────────────────────────────────────────────────────────────┘
```

---

## 4.2 Loss 수식

### MMD Loss (Gaussian Kernel 기반 Feature 분포 거리)

Multi-scale Gaussian RBF 커널을 사용하여 female/male feature 분포 간 거리를 측정:

$$L_{MMD} = \sum_{\sigma \in \{1,2,4,8,16\}} \left[ \mathbb{E}[k_\sigma(x_f, x_f')] + \mathbb{E}[k_\sigma(x_m, x_m')] - 2\,\mathbb{E}[k_\sigma(x_f, x_m)] \right]$$

Gaussian RBF 커널:

$$k_\sigma(x, y) = \exp\!\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)$$

Feature pooling (DETR decoder output $\rightarrow$ per-sample vector):

$$x_f = \frac{1}{N}\sum_{n=1}^{N} h_{f,n} \quad (B \times N \times D \;\xrightarrow{\text{GAP}}\; B \times D)$$

**1st (Symmetric MMD)**: $x_f$와 $x_m$ 모두 gradient 흐름 — 양방향 정렬

**2nd (Asymmetric MMD)**: $x_m = \text{detach}(x_m)$ — female만 male 분포 방향으로 이동

### Wasserstein Loss (Score-level, 단방향)

$$L_w = \frac{1}{K}\sum_{k=1}^{K} \text{ReLU}\!\left(\hat{s}_{m,k} - s_{f,k}\right)$$

- $\hat{s}_m$: detach된 남성 detection score (타겟으로 고정)
- $s_f$: 여성 detection score (학습 대상)
- $\text{ReLU}$: 여성 score가 남성보다 낮을 때만 패널티 발생

### Detection Loss (DETR Criterion)

$$L_{det} = L_{det,f} + L_{det,m}$$

$$L_{det,g} = \sum_{k \in \{ce,\,bbox,\,giou\}} w_k \cdot l_k^{(g)}$$

Hungarian Matching 기반 DETR 원본 criterion 그대로 사용.

### Total Loss

$$L_{total} = \lambda_{fair} \cdot L_{MMD} + \beta \cdot L_{det} + \lambda_w \cdot L_w$$

---

## 4.3 버전별 상세

### MMD 1st — Symmetric MMD 기반 확립

**설계 원칙**: Discriminator를 완전히 제거하고 커널 거리 최소화만으로 공정성 달성.
Female feature와 Male feature 양쪽 모두 gradient가 흘러 **상호 접근**하는 대칭 구조.

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `epochs` | 12 | |
| `batch_size` | 4 | |
| `lr_g` | 1e-4 | |
| `epsilon` (시작) | 0.05 | Warmup 시작값 |
| `epsilon_final` | 0.10 | Warmup 종료값 |
| `epsilon_warmup_epochs` | 10 | 선형 Warmup 기간 |
| `beta` | 0.5 | Detection Loss 가중치 (고정) |
| `lambda_fair` | 2.0 | MMD 손실 가중치 |
| `lambda_w` | 0.2 | Wasserstein 손실 가중치 |
| `max_norm` | 0.1 | Gradient clipping |
| MMD 방향 | **Symmetric** | Female ↔ Male 양방향 |
| Discriminator | **없음** | |
| Kernel σ | {1, 2, 4, 8, 16} | Multi-scale Gaussian |

**MMD 구현 핵심** (`_mmd_rbf`):
```python
# (B, N, D) → (B, D): Global Average Pooling
X_flat = X.mean(dim=1)
Y_flat = Y.mean(dim=1)

# 쌍별 제곱 유클리드 거리
XX = torch.cdist(X_flat, X_flat, p=2).pow(2)
YY = torch.cdist(Y_flat, Y_flat, p=2).pow(2)
XY = torch.cdist(X_flat, Y_flat, p=2).pow(2)

# 5개 스케일 합산
for sigma in [1.0, 2.0, 4.0, 8.0, 16.0]:
    gamma = 1.0 / (2 * sigma**2)
    mmd += exp(-gamma*XX).mean() + exp(-gamma*YY).mean() - 2*exp(-gamma*XY).mean()
```

**특징**: `_mmd_rbf(feat_f, feat_m)` — 두 feature 모두 gradient 허용

---

### MMD 2nd — Asymmetric MMD 개선

**설계 원칙**: 남성을 고정 타겟으로 설정하고 여성 feature만 남성 분포를 향해 이동.
1st의 대칭 구조가 남성 feature도 불필요하게 교란시킨다는 문제를 해결.

| 파라미터 | 값 | 1st 대비 변경 |
|----------|-----|---------------|
| `epochs` | 12 | 동일 |
| `batch_size` | 4 | 동일 |
| `epsilon` 스케줄 | 0.05→0.10 | 동일 |
| `beta` | **1.5** | **0.5 → 1.5 (+1.0)** |
| `lambda_fair` | **0.5** | **2.0 → 0.5 (−1.5)** |
| `lambda_w` | **0.5** | **0.2 → 0.5 (+0.3)** |
| MMD 방향 | **Asymmetric** | **Female → Male (단방향)** |
| Male feature | **detach** | Gradient 차단 |

**Asymmetric MMD 구현 핵심** (`_mmd_rbf_asymmetric`):
```python
# Male feature를 타겟으로 고정 (detach)
X_flat = X.mean(dim=1)           # Female: gradient 흐름
Y_flat = Y_target.detach().mean(dim=1)  # Male: gradient 차단

# MMD^2 = E[K(X,X)] + E[K(Y,Y)] - 2*E[K(X,Y)]
# Y가 detach되어 있으므로 gradient는 X(Female)를 통해서만 역전파
```

**하이퍼파라미터 조정 근거**:
- `beta` 0.5 → 1.5: Detection Loss 비중 대폭 증가 → 성능 하락 방지
- `lambda_fair` 2.0 → 0.5: 과도한 feature 정렬 방지 (1st에서 너무 강했음)
- `lambda_w` 0.2 → 0.5: Score-level alignment 강화로 실질적 공정성 개선

---

## 4.4 하이퍼파라미터 비교

| 파라미터 | MMD 1st | MMD 2nd | 변경 방향 | 이유 |
|----------|---------|---------|:---:|------|
| `epochs` | 12 | 12 | — | 동일 |
| `batch_size` | 4 | 4 | — | 동일 |
| `lr_g` | 1e-4 | 1e-4 | — | 동일 |
| `epsilon` 시작 | 0.05 | 0.05 | — | 동일 |
| `epsilon_final` | 0.10 | 0.10 | — | 동일 |
| `epsilon_warmup_epochs` | 10 | 10 | — | 동일 |
| `beta` | **0.5** | **1.5** | ↑ 3배 | 성능 하락 방지 |
| `lambda_fair` | **2.0** | **0.5** | ↓ 4분의 1 | 과정렬 방지 |
| `lambda_w` | **0.2** | **0.5** | ↑ 2.5배 | Score alignment 강화 |
| MMD 대칭성 | **대칭** | **비대칭** | — | Female만 이동 |
| Male gradient | 흐름 | **detach** | — | Male을 타겟 고정 |
| Discriminator | 없음 | 없음 | — | 동일 |
| Kernel σ | {1,2,4,8,16} | {1,2,4,8,16} | — | 동일 |
| `max_norm` | 0.1 | 0.1 | — | 동일 |

---

## 4.5 핵심 교훈

1. **MMD는 GAN보다 학습이 안정적**: Discriminator 없이 커널 거리를 직접 최소화하므로 D/G 균형 문제나 mode collapse가 발생하지 않음. 손실 값이 단조적으로 감소하는 안정적인 학습 곡선을 보임.

2. **두 버전 모두 WGAN-GD 7th 성능에 미달**: 안정성은 확보했으나 AR Gap 감소 폭이 7th(-60.5%)에 비해 작았음. 커널 거리 최소화만으로는 실제 score-level 공정성 개선이 충분하지 않을 수 있음.

3. **비대칭(2nd)이 대칭(1st)보다 우수**: Male feature를 detach하여 고정 타겟으로 설정하는 방식이 더 명확한 학습 방향을 제공함. `lambda_fair` 축소와 `beta` 증가 조합으로 detection 성능 하락도 억제.

4. **multi-scale 커널(σ ∈ {1,2,4,8,16})의 중요성**: 단일 σ보다 여러 스케일을 합산하면 feature 공간의 다양한 구조적 차이를 포착할 수 있음.

5. **Score-level 정렬(`lambda_w`)의 필요성**: Feature 분포 정렬(MMD)만으로는 detection score 격차를 직접 줄이기 어려움. 2nd에서 `lambda_w`를 0.2→0.5로 높인 것이 올바른 방향이었음.

---

## 4.6 한계 및 후속 연구로의 연결

| 한계 | 영향 |
|------|------|
| 두 버전 모두 7th 성능 미달 | MMD 단독으로는 adversarial D의 정보를 대체하기 부족 |
| 12 epoch의 짧은 학습 | 7th(24 epoch)와 학습량 차이가 결과에 영향 가능 |
| epsilon 3-phase 스케줄 미적용 | Hold/Cooldown 없이 단순 Warmup만 사용 (7th 대비 열위) |
| Feature 정렬 ≠ Score 정렬 직접 연결 | MMD로 feature 공간을 맞춰도 AP/AR Gap 개선으로 이어지지 않을 수 있음 |

**후속 전략**: MMD 계열의 한계로 인해 보다 풍부한 self-supervised 표현 학습을 활용하는 **Contrastive (InfoNCE) 계열** 및 **DINO 기반 접근**을 탐색하게 됨. Contrastive 1st에서 동일한 GAN-free 구조로 AR Gap -61%를 달성하며 MMD 대비 우수성을 확인.
