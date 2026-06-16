# Train FAAP Contrastive 2nd - 실행 가이드

## 개요
Contrastive 2nd는 1st와 7th GD의 장점을 결합한 버전입니다.

### 주요 개선점 (1st → 2nd)
1. **7th GD 스케줄 적용**:
   - Epsilon: 0.05 → 0.10 → 0.09 (warmup-hold-cooldown)
   - Beta: 0.2 → 0.6 (선형 증가)
2. **Contrastive Loss 최적화**:
   - Temperature: 0.1 → 0.07 (sharper)
   - Lambda_contrast: 1.0 → 1.5 (강화)
   - Lambda_var: 0.1 → 0.15 (분산 매칭 강화)
3. **단방향 Score Alignment**: 여성→남성 향상 집중 (7th 개념)

### 비교: 7th GD vs Contrastive 1st vs Contrastive 2nd

| 항목 | 7th GD | Contrastive 1st | Contrastive 2nd |
|------|--------|-----------------|-----------------|
| **Epsilon** | 0.09 (최종) | 0.08 (고정) | 0.09 (최종) |
| **Epoch** | 24 | 30 | 24 |
| **Fairness 방식** | GAN (Discriminator) | Contrastive Learning | Contrastive Learning |
| **AP Gap (Pert)** | 0.1059 ✓ | 0.1082 | ? (목표: <0.106) |
| **AR Gap (Pert)** | 0.0032 | 0.0031 ✓ | ? (목표: <0.003) |
| **안정성** | GAN 불안정성 | 매우 안정 ✓ | 매우 안정 ✓ |

## 학습 실행

### 1. Single GPU 학습
```bash
cd /home/dohyeong/Desktop/faap_gan
python -m faap_gan.train_faap_contrastive_2nd \
  --dataset_root /home/dohyeong/Desktop/faap_dataset \
  --epochs 24 \
  --batch_size 4 \
  --num_workers 6 \
  --lr_g 1e-4 \
  --epsilon 0.05 \
  --epsilon_final 0.10 \
  --epsilon_min 0.09 \
  --epsilon_warmup_epochs 8 \
  --epsilon_hold_epochs 6 \
  --epsilon_cooldown_epochs 10 \
  --beta 0.2 \
  --beta_final 0.6 \
  --lambda_contrast 1.5 \
  --temperature 0.07 \
  --lambda_align 0.4 \
  --lambda_var 0.15 \
  --lambda_score 0.4 \
  --save_every 1
```

### 2. 이전 체크포인트에서 재개
```bash
python -m faap_gan.train_faap_contrastive_2nd \
  --resume faap_outputs/faap_outputs_contrastive_2nd/checkpoints/epoch_0010.pth \
  --epochs 24
```

## 평가 (Evaluation)

### 자동 파일명 생성 (방법명 + Epoch 포함)
eval_faap.py가 자동으로 방법명과 epoch를 포함한 파일명을 생성합니다:

```bash
# Epoch 23 평가 (자동으로 test_metrics_contrastive_2nd_epoch_0023.json 생성)
python -m faap_gan.eval_faap \
  --generator_checkpoint faap_outputs/faap_outputs_contrastive_2nd/checkpoints/epoch_0023.pth \
  --epsilon 0.09 \
  --split test \
  --batch_size 10

# 결과 저장 위치 (자동):
# faap_outputs/faap_outputs_contrastive_2nd/test_metrics_contrastive_2nd_epoch_0023.json
```

### 다른 방법 비교 예시
```bash
# 7th GD 평가
python -m faap_gan.eval_faap \
  --generator_checkpoint faap_outputs/faap_outputs_gd_7th/checkpoints/epoch_0023.pth \
  --epsilon 0.09 \
  --split test

# 생성: test_metrics_gd_7th_epoch_0023.json

# Contrastive 1st 평가
python -m faap_gan.eval_faap \
  --generator_checkpoint faap_outputs/faap_outputs_contrastive_1st/checkpoints/epoch_0029.pth \
  --epsilon 0.08 \
  --split test

# 생성: test_metrics_contrastive_1st_epoch_0029.json
```

### 수동으로 경로 지정
```bash
python -m faap_gan.eval_faap \
  --generator_checkpoint faap_outputs/faap_outputs_contrastive_2nd/checkpoints/epoch_0023.pth \
  --epsilon 0.09 \
  --split test \
  --results_path faap_outputs/faap_outputs_contrastive_2nd/custom_results.json
```

### 여러 epoch 평가 (스크립트)
```bash
# 모든 체크포인트 평가 (방법명 자동 포함)
for epoch in {0005..0023..1}; do
  python -m faap_gan.eval_faap \
    --generator_checkpoint faap_outputs/faap_outputs_contrastive_2nd/checkpoints/epoch_${epoch}.pth \
    --epsilon 0.09 \
    --split test \
    --batch_size 10
done

# 결과 파일:
# test_metrics_contrastive_2nd_epoch_0005.json
# test_metrics_contrastive_2nd_epoch_0006.json
# ...
# test_metrics_contrastive_2nd_epoch_0023.json
```

## Output 구조 (방법명 포함)

```
faap_outputs/faap_outputs_contrastive_2nd/
├── checkpoints/
│   ├── epoch_0000.pth
│   ├── epoch_0001.pth
│   └── ...
├── dataset_layout.json
├── train_log.jsonl
├── test_metrics_contrastive_2nd_epoch_0000.json  # 자동 생성 (방법명 포함)
├── test_metrics_contrastive_2nd_epoch_0001.json
└── ...
```

### 파일명 규칙
- 형식: `{split}_metrics_{method}_{version}_epoch_{XXXX}.json`
- 예시:
  - `test_metrics_contrastive_2nd_epoch_0023.json`
  - `test_metrics_gd_7th_epoch_0023.json`
  - `val_metrics_contrastive_1st_epoch_0029.json`

## 결과 JSON 구조 (7th와 동일)

```json
{
  "baseline": {
    "male": {"AP": 0.511, "AR": 0.834},
    "female": {"AP": 0.404, "AR": 0.826}
  },
  "perturbed": {
    "male": {"AP": ..., "AR": ...},
    "female": {"AP": ..., "AR": ...}
  },
  "gaps": {
    "AP": {
      "baseline": 0.1063,
      "perturbed": ...
    },
    "AR": {
      "baseline": 0.0081,
      "perturbed": ...
    }
  },
  "hyperparams": {
    "epsilon": 0.09,
    "generator_checkpoint": "...",
    "split": "test"
  }
}
```

## 기대 효과

1. **AP Gap 개선**: 7th의 0.1059 수준 달성 목표
2. **AR Gap 유지**: Contrastive 1st의 0.0031 수준 유지
3. **안정적 학습**: GAN 없이 Contrastive로 안정성 확보
4. **Detection 성능 보존**: Beta 스케줄로 점진적 공정성 향상

## 하이퍼파라미터 튜닝 가이드

성능이 기대에 못 미칠 경우:

1. **AP Gap이 너무 클 때** (>0.11):
   - `lambda_score` 증가 (0.4 → 0.5)
   - `lambda_contrast` 증가 (1.5 → 2.0)

2. **Detection 성능 저하** (AP < 0.50):
   - `beta_final` 증가 (0.6 → 0.7)
   - `lambda_contrast` 감소 (1.5 → 1.2)

3. **학습 불안정**:
   - `temperature` 증가 (0.07 → 0.1)
   - `max_norm` 조정 (0.1 → 0.05)

## 모니터링

학습 중 train_log.jsonl 확인:
```bash
tail -f faap_outputs/faap_outputs_contrastive_2nd/train_log.jsonl
```

주요 지표:
- `g_contrast`: Contrastive loss (낮을수록 좋음)
- `g_score`: Score alignment loss (낮을수록 좋음)
- `g_det`: Detection loss (적정 수준 유지)
- `epsilon`: 현재 epsilon 값
- `beta`: 현재 beta 가중치
