# FAAP-style Perturbation Training for DETR

## 목적
- 동결된 DETR에 대해 여성 이미지에만 작은 perturbation을 적용하는 Generator(G)를 학습하여 성별 간 검출 격차(AP/AR)를 줄입니다.
- Discriminator(D)는 perturbation 이후 DETR feature로 성별을 예측하려 하고, G는 이를 혼란시키면서 여성 검출 성능(DETR loss)을 유지/향상합니다.

## 주요 스크립트
- `train_faap.py`: 학습 루프 (G/D 업데이트, 체크포인트 저장, dataset 레이아웃 기록).
- `eval_faap.py`: 남/녀 원본과 perturb 적용 4가지 경우의 AP/AR 및 gap 계산.
- `models.py`: 동결 DETR 래퍼, perturbation generator(U-Net), gender discriminator.
- `datasets.py`: men/women split COCO 포맷 로더, 균형 샘플링 지원.

## 기본 하이퍼파라미터 (CLI로 덮어쓰기 가능)
- `epochs`: 20
- `batch_size`: 8 (L40S 기준, OOM 시 줄이기)
- `num_workers`: 8
- `lr_g`: 1e-4 (Generator), `lr_d`: 1e-4 (Discriminator)
- `k_d`: 1 (D 업데이트 횟수/iter)
- `epsilon`: 0.05 (perturb L∞ 스케일)
- `alpha`: 0.1 (fairness entropy 가중치)
- `beta`: 1.0 (DETR loss 가중치)
- `max_norm`: 0.1 (G gradient clipping)
- 경로: `dataset_root=~/Desktop/faap_dataset`, `detr_repo=~/Desktop/detr`, `detr_checkpoint=~/Desktop/detr/detr-r50-e632da11.pth`, `output_dir=faap_outputs`
- 재시작: `--resume <checkpoint.pth>` (G/D/optimizer 상태와 epoch+1에서 이어서 학습)

## 학습 흐름
1. DETR 가중치를 동결해 로드하고, G/D 초기화.
2. men/women 데이터 로더를 성별 균형으로 구성.
3. 각 iteration:
   - D 업데이트(k_d회): 여성은 `x+G(x)`, 남성은 `x`로 feature 추출 → 성별 CE 최소화.
   - G 업데이트(여성만): `-(CE + α·entropy)`로 성별 정보 은닉 + `β·DETR loss`로 검출 유지 → 총손실 `g_total` 최소화, ε·tanh로 L∞ 제한.
4. 매 epoch마다 로그(`train_log.jsonl`)와 체크포인트(`output_dir/checkpoints/epoch_*.pth`) 저장.

## 평가 흐름
- `eval_faap.py --generator_checkpoint <ckpt>` 실행 시:
  - 남/녀 원본 AP/AR과 perturb 적용 AP/AR 모두 계산.
  - gap(AP/AR 차이) 정리 및 결과 JSON 저장(`faap_outputs/faap_metrics.json` 등).
  - generator를 주지 않으면 baseline만 계산.

## 기대 결과
- 여성 AP/AR 상승 및 남녀 gap 감소.
- 남성 AP/AR은 baseline과 동일하거나 근접(성능 유지).
- perturbation 크기는 ε=0.05 내에서 시각적으로 미미.

## 실행 예시
```bash
cd ~/Desktop/faap_gan
python train_faap.py \
  --dataset_root ~/Desktop/faap_dataset \
  --detr_repo ~/Desktop/detr \
  --detr_checkpoint ~/Desktop/detr/detr-r50-e632da11.pth \
  --output_dir faap_outputs \
  --device cuda \
  --batch_size 8 \
  --num_workers 8

# 이어서 학습
python train_faap.py --resume faap_outputs/checkpoints/epoch_0004.pth ...

# 평가
python eval_faap.py \
  --dataset_root ~/Desktop/faap_dataset \
  --detr_repo ~/Desktop/detr \
  --detr_checkpoint ~/Desktop/detr/detr-r50-e632da11.pth \
  --generator_checkpoint faap_outputs/checkpoints/epoch_0009.pth \
  --results_path faap_outputs/faap_metrics.json
```

## 출력 예시
- 학습 로그(스텝/누적 평균):  
  `Epoch 0 [ 200/3423] d_loss: 0.678 (0.688)  g_fair: -0.738 (-0.751)  g_det: 5.266 (5.608)  g_total: 4.605 (4.858)`
- 평가 결과 JSON (`faap_outputs/faap_metrics.json` 예시):
  ```json
  {
    "baseline": {
      "male": {"AP": 38.2, "AR": 55.1},
      "female": {"AP": 31.5, "AR": 48.0}
    },
    "perturbed": {
      "male": {"AP": 38.0, "AR": 55.0},
      "female": {"AP": 34.7, "AR": 51.2}
    },
    "gaps": {
      "AP": {"baseline": 6.7, "perturbed": 3.3},
      "AR": {"baseline": 7.1, "perturbed": 3.8}
    },
    "hyperparams": {
      "epsilon": 0.05,
      "generator_checkpoint": "faap_outputs/checkpoints/epoch_0009.pth",
      "detr_checkpoint": "detr-r50-e632da11.pth",
      "split": "test",
      "batch_size": 8
    }
  }
  ```
