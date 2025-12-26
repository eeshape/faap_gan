# train_faap_wgan_GD_7th summary (vs 4th/5th/6th/6th_2)

## Observations
- 4th is the strongest overall (female AR and smallest gaps).
- 5th cap per gender did not help (likely reduced data).
- 6th cooldown + beta schedule did not beat 4th with short runs.
- 6th_2 stress window degraded AP/gaps, so it is avoided.

## 7th design goals
Keep the 4th core (one-way Wasserstein) and apply **conservative, late-phase stabilization**
so female gains stay while detection recovers.

### 1) Longer training + gentle epsilon cooldown
- Default `epochs=24` (match the best 4th run length).
- Epsilon schedule (softened 6th idea):
  - warmup 8 -> hold 6 -> cooldown 10
  - `epsilon_final=0.10`, `epsilon_min=0.09`
- Purpose: keep fairness signal early, then recover detection late.

### 2) Mild detection weight ramp
- `beta: 0.5 -> 0.6` linear increase (lighter than 6th).
- Purpose: preserve detection in later epochs.

### 3) Down-weight male fairness
- `fair_m_scale=0.5` (female stays 1.0)
- Purpose: reduce unnecessary male perturbation and protect male AP/AR.

### 4) Monitoring
- Keep per-gender `obj_score/obj_frac` logging (from 6th).

### 5) Optional per-gender cap
- `--max_train_per_gender` added (default 0: disabled).
- Use only if repeated sampling becomes a concern.

---

## Recommended run
```bash
python train_faap_wgan_GD_7th.py \
  --epochs 24 \
  --epsilon_final 0.10 \
  --epsilon_min 0.09 \
  --epsilon_warmup_epochs 8 \
  --epsilon_hold_epochs 6 \
  --epsilon_cooldown_epochs 10 \
  --beta 0.5 \
  --beta_final 0.6 \
  --fair_m_scale 0.5
```

## Eval example
```bash
python eval_faap.py \
  --dataset_root /home/dohyeong/Desktop/faap_dataset \
  --detr_checkpoint /home/dohyeong/Desktop/detr/detr-r50-e632da11.pth \
  --generator_checkpoint /home/dohyeong/Desktop/faap_gan/faap_outputs/faap_outputs_gd_7th/checkpoints/epoch_0023.pth \
  --epsilon 0.09 \
  --split test \
  --results_path faap_outputs/faap_outputs_gd_7th/test_metrics_epoch_0023.json
```

Note: set eval `--epsilon` to match the schedule value at the checkpoint epoch.
