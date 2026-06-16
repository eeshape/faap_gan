# train_faap_wgan_GD_6th_2 변경 요약 (vs 6th)

## 근거
- 짧은 “stress window”를 넣어서 교란과 공정성 가중치를 의도적으로 크게 올려 민감도를 확인.
- 공정성 압력이 강해졌을 때 detection 지표가 어떻게 흔들리는지 보기 쉬움.

## 변경 내용
1) Stress window 배율 추가
   - 신규 인자: stress_start_epoch, stress_epochs, stress_eps_scale, stress_fair_scale, stress_w_scale, stress_beta_scale
   - 기본 동작: epsilon warmup 직후부터 2 epoch 동안 epsilon/fairness/W 가중치를 상향

2) 로깅 확장
   - train_log.jsonl에 lambda_fair, lambda_w, stress 관련 스케일 값을 기록

## 사용 예시
python train_faap_wgan_GD_6th_2.py \
  --epochs 24 \
  --epsilon_final 0.10 \
  --epsilon_min 0.08 \
  --epsilon_hold_epochs 2 \
  --epsilon_cooldown_epochs 8 \
  --beta 0.5 \
  --beta_final 0.7 \
  --stress_epochs 2 \
  --stress_eps_scale 1.3 \
  --stress_fair_scale 1.8 \
  --stress_w_scale 2.0

## Eval
```bash
python eval_faap.py \
  --dataset_root /home/dohyeong/Desktop/faap_dataset \
  --detr_checkpoint /home/dohyeong/Desktop/detr/detr-r50-e632da11.pth \
  --generator_checkpoint /home/dohyeong/Desktop/faap_gan/faap_outputs/faap_outputs_gd_6th_2/checkpoints/epoch_0010.pth \
  --epsilon 0.13 \
  --split test \
  --results_path faap_outputs/faap_outputs_gd_6th_2/test_metrics_epoch_0010.json
```
