# train_faap_wgan_GD_3rd 변경 요약 (vs 2nd)

- `k_d`: 2 → 4 (판별기 업데이트 스텝 증가)
- `epsilon_final`: 0.12 → 0.10 (목표 교란 크기 축소)
- `epsilon_warmup_epochs`: 5 → 10 (교란 크기 워밍업 기간 연장)
- `beta`: 0.3 → 0.5 (검출 유지 손실 가중치 강화)
- `lambda_fair`: 5.0 → 2.0 (공정성 손실 가중치 완화)
- `output_dir` 기본값: 스크립트 이름 기반 자동 설정(`faap_outputs/faap_outputs_<스크립트명>`), 예) 3rd → `faap_outputs/faap_outputs_gd_3rd`

[eval 001]
 python eval_faap.py   --dataset_root /home/dohyeong/Desktop/faap_dataset   --detr_checkpoint /home/dohyeong/Desktop/detr/detr-r50-e632da11.pth   --generator_checkpoint /home/dohyeong/Desktop/faap_gan/faap_outputs/faap_outputs_gd_3rd/checkpoints/epoch_0001.pth   --epsilon  0.0556   --split test  --results_path faap_outputs/faap_outputs_gd_3rd/test_metrics_epoch_0001.json


 epoch 0: 0.0500
epoch 1: 0.0556
epoch 2: 0.0611
epoch 3: 0.0667
epoch 4: 0.0722
epoch 5: 0.0778
epoch 6: 0.0833
epoch 7: 0.0889
epoch 8: 0.0944
epoch ≥9: 0.1000 (이후 에폭은 모두 0.10 유지)