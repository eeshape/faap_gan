## [val명령어]
cd /home/dohyeong/Desktop/faap_gan
python eval_faap.py \
  --dataset_root /home/dohyeong/Desktop/faap_dataset \
  --detr_checkpoint /home/dohyeong/Desktop/detr/detr-r50-e632da11.pth \
  --generator_checkpoint /home/dohyeong/Desktop/faap_gan/faap_outputs/checkpoints/epoch_0021.pth \
  --split val --batch_size 4 --num_workers 4 --device cuda \
  --results_path faap_outputs/faap_metrics_val_perturbed.json

[3rd]
 python eval_faap.py   --dataset_root /home/dohyeong/Desktop/faap_dataset   --detr_checkpoint /home/dohyeong/Desktop/detr/detr-r50-e632da11.pth   --generator_checkpoint /home/dohyeong/Desktop/faap_gan/faap_outputs/checkpoints_3rd/checkpoints/epoch_0004.pth  --split val --batch_size 16 --num_workers 16 --device cuda   --results_path faap_outputs/faap_metrics_val_perturbed_3rd.json

[wgan]

python eval_faap.py   --dataset_root /home/dohyeong/Desktop/faap_dataset   --detr_checkpoint /home/dohyeong/Desktop/detr/detr-r50-e632da11.pth   --generator_checkpoint /home/dohyeong/Desktop/faap_gan/faap_outputs_w/checkpoints/epoch_0010.pth  --split val --batch_size 16 --num_workers 16 --device cuda   --results_path faap_outputs/faap_metrics_val_perturbed_wgan_1st_0010.json

## [gen 명령어]

cd ~/Desktop/faap_gan
python gen_images.py \
  --generator_checkpoint faap_outputs/checkpoints/epoch_0021.pth \
  --dataset_root ~/Desktop/faap_dataset \
  --split val \
  --output_root faap_outputs/generated_images \
  --device cuda

cd ~/Desktop/faap_gan
python gen_images.py \
  --generator_checkpoint /home/dohyeong/Desktop/faap_gan/faap_outputs_w/checkpoints/epoch_0008.pth \
  --dataset_root ~/Desktop/faap_dataset \
  --split val \
  --output_root faap_outputs_w/generated_images \
  --device cuda

## [병렬 Train 처리 명령어]
cd /home/dohyeong/Desktop/faap_gan
CUDA_VISIBLE_DEVICES=1,2 \
torchrun --nproc_per_node=2 --master_port 29501 train_faap.py \
  --dataset_root /home/dohyeong/Desktop/faap_dataset \
  --detr_checkpoint /home/dohyeong/Desktop/detr/detr-r50-e632da11.pth \
  --output_dir faap_outputs/checkpoints_3rd

## [단일 Train 처리 명령어]
export CUDA_VISIBLE_DEVICES=1

cd /home/dohyeong/Desktop/faap_gan
python train_faap.py \
  --dataset_root /home/dohyeong/Desktop/faap_dataset \
  --detr_checkpoint /home/dohyeong/Desktop/detr/detr-r50-e632da11.pth \
  --output_dir faap_outputs/checkpoints_3rd

  [wgan Train 처리 명령어 -- resume]

python train_faap_wgan.py \
  --dataset_root /home/dohyeong/Desktop/faap_dataset \
  --detr_repo /home/dohyeong/Desktop/detr \
  --detr_checkpoint /home/dohyeong/Desktop/detr/detr-r50-e632da11.pth \
  --output_dir faap_outputs_w \
  --resume faap_outputs_w/checkpoints/epoch_0008.pth \
  --epochs 20 \
  --device cuda

