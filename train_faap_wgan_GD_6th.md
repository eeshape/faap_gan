# train_faap_wgan_GD_6th 변경 요약 (vs 4th)

## 근거 (epoch 23 관측)
- `faap_outputs_gd_4th/train_log.jsonl` 기준 epoch 23에서 `obj_score=0.1502`, `obj_frac=0.1135`로 초기보다 하락.
- epsilon 상승 구간과 `obj_score/obj_frac`의 음의 상관이 강함(관측 상관 약 -0.98).
- 해석: perturbation 강도가 커질수록 detection confidence가 떨어지는 경향 → 후반부에는 노이즈 강도를 낮추고 detection 보존을 강화하는 편이 합리적.

## 변경 내용
1) **Epsilon 스케줄 확장 (warmup → hold → cooldown)**
   - warmup 후 일정 기간 `epsilon_final` 유지, 이후 `epsilon_min`으로 선형 감소.
   - 기본값: `epsilon_final=0.10`, `epsilon_min=0.08`, `epsilon_hold_epochs=2`, `epsilon_cooldown_epochs=8`.
   - 목적: 공정성 신호 확보 후 detection 회복 구간 확보.

2) **Detection 가중치(beta) 스케줄 추가**
   - `beta`에서 `beta_final`로 epoch에 따라 선형 증가.
   - 기본값: `beta=0.5`, `beta_final=0.7`.
   - 목적: 후반부에 detection 보존 비중을 점진적으로 강화.

3) **로깅 확장**
   - 기존 `obj_score/obj_frac` 외에 성별별(`obj_score_f/m`, `obj_frac_f/m`) 로그 추가.
   - `beta` 값도 `train_log.jsonl`에 기록.

## 사용 예시
```bash
python train_faap_wgan_GD_6th.py \
  --epochs 24 \
  --epsilon_final 0.10 \
  --epsilon_min 0.08 \
  --epsilon_hold_epochs 2 \
  --epsilon_cooldown_epochs 8 \
  --beta 0.5 \
  --beta_final 0.7
```

## Eval
```bash
python eval_faap.py \
  --dataset_root /home/dohyeong/Desktop/faap_dataset \
  --detr_checkpoint /home/dohyeong/Desktop/detr/detr-r50-e632da11.pth \
  --generator_checkpoint /home/dohyeong/Desktop/faap_gan/faap_outputs/faap_outputs_gd_6th/checkpoints/epoch_0010.pth \
  --epsilon 0.10 \
  --split test \
  --results_path faap_outputs/faap_outputs_gd_6th/test_metrics_epoch_0010.json
```
