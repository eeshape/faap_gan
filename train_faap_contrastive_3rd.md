# Contrastive 3rd: 비대칭 Contrastive Fairness

## 개요

`train_faap_contrastive_3rd.py`는 1st 버전을 기반으로 **비대칭 Contrastive Loss**를 도입한 버전입니다.
7th WGAN-GD의 성능 우위 요인 중 하나인 비대칭 가중치(`fair_f_scale`, `fair_m_scale`)를 Contrastive 방식에 적용했습니다.

---

## 1st vs 3rd 핵심 차이점

### 1.1 Contrastive Loss 변경

| 버전 | 수식 | 설명 |
|------|------|------|
| **1st (대칭)** | $\mathcal{L} = \frac{L_{f \to m} + L_{m \to f}}{2}$ | 양방향 동등 가중치 |
| **3rd (비대칭)** | $\mathcal{L} = 1.5 \cdot L_{f \to m} + 0.5 \cdot L_{m \to f}$ | 여성→남성 방향 강화 |

### 1.2 변경 파일 diff

```diff
# parse_args() 함수 내 추가된 인자
+ parser.add_argument(
+     "--contrast_f_scale",
+     type=float,
+     default=1.5,
+     help="weight for female→male contrastive direction",
+ )
+ parser.add_argument(
+     "--contrast_m_scale",
+     type=float,
+     default=0.5,
+     help="weight for male→female contrastive direction",
+ )

# 손실 함수 변경
- def _cross_gender_contrastive_loss(proj_f, proj_m, temperature=0.1):
-     ...
-     return (loss_f_to_m + loss_m_to_f) / 2
+ def _asymmetric_cross_gender_contrastive_loss(
+     proj_f, proj_m, temperature=0.1, f_scale=1.5, m_scale=0.5
+ ):
+     ...
+     total_loss = f_scale * loss_f_to_m + m_scale * loss_m_to_f
+     return total_loss, loss_f_to_m, loss_m_to_f

# 학습 루프 내 호출 변경
- contrast_loss = _cross_gender_contrastive_loss(proj_f, proj_m, args.temperature)
+ contrast_loss, contrast_f2m, contrast_m2f = _asymmetric_cross_gender_contrastive_loss(
+     proj_f, proj_m, args.temperature,
+     f_scale=args.contrast_f_scale,
+     m_scale=args.contrast_m_scale,
+ )

# 로깅 추가
+ "g_contrast_f2m": ...,  # 여성→남성 방향 손실
+ "g_contrast_m2f": ...,  # 남성→여성 방향 손실
+ "contrast_f_scale": ...,
+ "contrast_m_scale": ...,
```

### 1.3 설계 근거

1. **문제 인식**: 여성 탐지 성능이 남성보다 낮음
2. **대칭 처리의 한계**: 양쪽을 동등하게 이동시키면 비효율적
3. **비대칭 해법**: 문제가 있는 여성 그룹을 남성 방향으로 더 강하게 이동
4. **7th WGAN-GD 영감**: `fair_f_scale=1.0`, `fair_m_scale=0.5`와 유사한 비율 적용

---

## 하이퍼파라미터 비교

| 파라미터 | 1st | 3rd | 비고 |
|----------|-----|-----|------|
| `contrast_f_scale` | 1.0 (암묵적) | **1.5** | 여성→남성 강화 |
| `contrast_m_scale` | 1.0 (암묵적) | **0.5** | 남성→여성 약화 |
| `lambda_contrast` | 1.0 | 1.0 | 동일 |
| `temperature` | 0.1 | 0.1 | 동일 |
| `lambda_align` | 0.5 | 0.5 | 동일 |
| `lambda_var` | 0.1 | 0.1 | 동일 |
| `lambda_score` | 0.3 | 0.3 | 동일 |
| `beta` | 0.6 | 0.6 | 동일 |
| `epsilon` | 0.08 | 0.08 | 동일 |

---

## 학습 명령어

### 기본 학습

```bash
cd /home/dohyeong/Desktop/faap_gan

# 기본 설정으로 학습
python train_faap_contrastive_3rd.py

# 커스텀 비대칭 가중치
python train_faap_contrastive_3rd.py \
    --contrast_f_scale 2.0 \
    --contrast_m_scale 0.3

# 전체 옵션 지정
python train_faap_contrastive_3rd.py \
    --dataset_root /home/dohyeong/Desktop/faap_dataset \
    --epochs 30 \
    --batch_size 4 \
    --epsilon 0.08 \
    --lambda_contrast 1.0 \
    --contrast_f_scale 1.5 \
    --contrast_m_scale 0.5 \
    --temperature 0.1 \
    --output_dir faap_outputs/faap_outputs_3rd
```

### 분산 학습

```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train_faap_contrastive_3rd.py \
    --distributed \
    --batch_size 2
```

### 체크포인트에서 재개

```bash
python train_faap_contrastive_3rd.py \
    --resume faap_outputs/faap_outputs_3rd/checkpoints/epoch_0015.pth
```

---

## 평가/테스트 명령어

### eval_faap.py 사용 (평가)

```bash
cd /home/dohyeong/Desktop/faap_gan

# 특정 체크포인트 평가
python eval_faap.py \
    --checkpoint faap_outputs/faap_outputs_3rd/checkpoints/epoch_0029.pth \
    --dataset_root /home/dohyeong/Desktop/faap_dataset \
    --split val

# 테스트 셋 평가
python eval_faap.py \
    --checkpoint faap_outputs/faap_outputs_3rd/checkpoints/epoch_0029.pth \
    --dataset_root /home/dohyeong/Desktop/faap_dataset \
    --split test

# 성별별 상세 평가
python eval_faap.py \
    --checkpoint faap_outputs/faap_outputs_3rd/checkpoints/epoch_0029.pth \
    --dataset_root /home/dohyeong/Desktop/faap_dataset \
    --split val \
    --per_gender
```

### test.py 사용 (DETR 표준 테스트)

```bash
cd /home/dohyeong/Desktop/detr

# Generator 적용 후 COCO 평가
python test.py \
    --generator_checkpoint /home/dohyeong/Desktop/faap_gan/faap_outputs/faap_outputs_3rd/checkpoints/epoch_0029.pth \
    --dataset_root /home/dohyeong/Desktop/faap_dataset
```

---

## 출력 디렉토리 구조

```
faap_outputs/faap_outputs_3rd/
├── checkpoints/
│   ├── epoch_0000.pth
│   ├── epoch_0001.pth
│   ├── ...
│   └── epoch_0029.pth
├── train_log.jsonl          # 학습 로그 (JSON Lines)
└── dataset_layout.json      # 데이터셋 정보
```

### train_log.jsonl 분석

```bash
# 마지막 에폭 확인
tail -1 faap_outputs/faap_outputs_3rd/train_log.jsonl | python -m json.tool

# 손실 추이 시각화 (Python)
python -c "
import json
import matplotlib.pyplot as plt

logs = [json.loads(l) for l in open('faap_outputs/faap_outputs_3rd/train_log.jsonl')]
epochs = [l['epoch'] for l in logs]
contrast = [l['g_contrast'] for l in logs]
f2m = [l['g_contrast_f2m'] for l in logs]
m2f = [l['g_contrast_m2f'] for l in logs]

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, contrast, label='Total Contrast')
plt.plot(epochs, f2m, label='F→M (scaled 1.5)')
plt.plot(epochs, m2f, label='M→F (scaled 0.5)')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.title('Asymmetric Contrastive Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs, [l['obj_score_f'] for l in logs], label='Female')
plt.plot(epochs, [l['obj_score_m'] for l in logs], label='Male')
plt.xlabel('Epoch'); plt.ylabel('Objectness Score'); plt.legend()
plt.title('Detection Score by Gender')
plt.tight_layout()
plt.savefig('faap_outputs/faap_outputs_3rd/training_curves.png')
print('Saved to training_curves.png')
"
```

---

## 1st vs 3rd 비교 실험

### 동일 조건 비교 학습

```bash
# 1st 학습
python train_faap_contrastive_1st.py \
    --epochs 30 \
    --output_dir faap_outputs/comparison_1st

# 3rd 학습
python train_faap_contrastive_3rd.py \
    --epochs 30 \
    --output_dir faap_outputs/comparison_3rd

# 결과 비교
python -c "
import json

def load_final(path):
    logs = [json.loads(l) for l in open(path)]
    return logs[-1]

r1 = load_final('faap_outputs/comparison_1st/train_log.jsonl')
r3 = load_final('faap_outputs/comparison_3rd/train_log.jsonl')

print('=== Final Epoch Comparison ===')
print(f'                    1st        3rd')
print(f'obj_score_f:   {r1[\"obj_score_f\"]:.4f}    {r3[\"obj_score_f\"]:.4f}')
print(f'obj_score_m:   {r1[\"obj_score_m\"]:.4f}    {r3[\"obj_score_m\"]:.4f}')
print(f'gap (m-f):     {r1[\"obj_score_m\"]-r1[\"obj_score_f\"]:.4f}    {r3[\"obj_score_m\"]-r3[\"obj_score_f\"]:.4f}')
print(f'g_contrast:    {r1[\"g_contrast\"]:.4f}    {r3[\"g_contrast\"]:.4f}')
"
```

---

## 기대 효과

1. **여성 탐지 성능 향상**: 여성→남성 방향 손실 강화로 여성 특징이 남성 수준으로 이동
2. **남성 성능 유지**: 남성→여성 방향 약화로 남성 특징의 변화 최소화
3. **Fairness Gap 감소**: 성별 간 탐지 성능 차이 축소
4. **7th WGAN-GD 수준의 성능**: 비대칭 전략의 장점을 Contrastive에 적용

---

## 추가 실험 제안

### 비대칭 가중치 탐색

```bash
# 더 강한 비대칭
python train_faap_contrastive_3rd.py \
    --contrast_f_scale 2.0 \
    --contrast_m_scale 0.2 \
    --output_dir faap_outputs/3rd_asymm_strong

# 약한 비대칭
python train_faap_contrastive_3rd.py \
    --contrast_f_scale 1.2 \
    --contrast_m_scale 0.8 \
    --output_dir faap_outputs/3rd_asymm_weak
```

### 7th WGAN-GD 비율 적용

```bash
# 7th의 fair_f_scale=1.0, fair_m_scale=0.5 비율 적용
# 정규화: 1.0/(1.0+0.5)=0.667, 0.5/(1.0+0.5)=0.333
# 스케일 맞춤: 1.33, 0.67 (합=2.0으로 1st와 동일)
python train_faap_contrastive_3rd.py \
    --contrast_f_scale 1.33 \
    --contrast_m_scale 0.67 \
    --output_dir faap_outputs/3rd_7th_ratio
```

---

## 수학적 분석

### 1st (대칭)

$$\mathcal{L}_{\text{contrast}}^{1st} = \frac{1}{2}\left(\mathcal{L}_{f \to m} + \mathcal{L}_{m \to f}\right)$$

Gradient 관점:
- 여성 특징: $\nabla_{z_f} = \frac{1}{2} \nabla_{z_f} \mathcal{L}_{f \to m}$
- 남성 특징: $\nabla_{z_m} = \frac{1}{2} \nabla_{z_m} \mathcal{L}_{m \to f}$

### 3rd (비대칭)

$$\mathcal{L}_{\text{contrast}}^{3rd} = 1.5 \cdot \mathcal{L}_{f \to m} + 0.5 \cdot \mathcal{L}_{m \to f}$$

Gradient 관점:
- 여성 특징: $\nabla_{z_f} = 1.5 \cdot \nabla_{z_f} \mathcal{L}_{f \to m}$ **(3배 강화)**
- 남성 특징: $\nabla_{z_m} = 0.5 \cdot \nabla_{z_m} \mathcal{L}_{m \to f}$ **(동일)**

→ 여성 특징에 대한 gradient가 3배 강해져 빠르게 남성 방향으로 이동

---

*생성일: 2026-01-12*
*기반 버전: train_faap_contrastive_1st.py*
