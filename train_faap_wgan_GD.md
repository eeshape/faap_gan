# train_faap_wgan_GD: 파이프라인 및 손실 정리

## 개요
- 기존 `train_faap_wgan.py`를 확장해 **여성/남성 모두** 생성기(G) 교란을 적용.
- 판별기(D)는 두 교란본을 입력받아 여=1, 남=0을 분류하도록 학습.
- 생성기(G)는 두 성별 모두 공정성(분류 혼란), 검출 유지, Wasserstein 분포 정렬을 동시에 최소화.

## 데이터 흐름
1) 배치 로딩: `train_loader` → `(samples, targets, genders)`.
2) 성별 분리: `genders`를 소문자화 후 여성/남성 인덱스로 배치 분할.
3) 생성기 교란:
   - 여성: `p_f = G(x_f)`, `x_f + p_f`
   - 남성: `p_m = G(x_m)`, `x_m + p_m`
4) 판별기 학습:
   - 입력: 교란된 여성/남성 특징.
   - 목표: 여성=1, 남성=0 크로스엔트로피 최소화.
5) 생성기 학습:
   - 공정성: D가 성별을 확신 못 하게 `(CE + α·H)`를 음수 부호로 합산(여+남).
   - 검출 유지: DETR 검출 손실을 여성+남성 합으로 더함.
   - 분포 정렬: 여성/남성 검출 점수의 1D Wasserstein 거리 최소화.
6) 로깅: 교란 크기(delta L∞/L2), 객체 점수 평균/비율, 손실들 기록.

## 손실 수식 (G)
\[
L_G
= -\Big( \mathrm{CE}(D(x_f + p_f),\,1) + \alpha \, \mathrm{H}(D(x_f + p_f))
        + \mathrm{CE}(D(x_m + p_m),\,0) + \alpha \, \mathrm{H}(D(x_m + p_m)) \Big)
\\ \quad
+ \beta \big( \mathcal{L}_{\mathrm{det}}(x_f + p_f) + \mathcal{L}_{\mathrm{det}}(x_m + p_m) \big)
+ \lambda_w \, W(s_f,\, s_m)
\]
- \(x_f, x_m\): 여성/남성 이미지, \(p_f, p_m = G(x)\) 교란.
- \(\mathrm{CE}\): 크로스엔트로피, \(\mathrm{H}\): 엔트로피.
- \(\mathcal{L}_{\mathrm{det}}\): DETR 검출 손실.
- \(W(s_f, s_m)\): 여성/남성 검출 점수 분포의 1D Wasserstein 거리.

## 손실 수식 (D)
\[
L_D = \mathrm{CE}(D(x_f + p_f), 1) + \mathrm{CE}(D(x_m + p_m), 0)
\]
- 두 성별 모두 교란본을 입력으로 사용.

## 입력 예시와 흐름
- 배치 내 8장 예: 여성 4장, 남성 4장 → `genders = ["female", ..., "male"]`.
- 분리 후:
  - `female_batch` (4장) → `p_f = G(female_batch)` → `D(female_batch + p_f)` → 공정성·검출·W.
  - `male_batch` (4장) → `p_m = G(male_batch)` → `D(male_batch + p_m)` → 공정성·검출·W.
- Wasserstein: `s_f`(여성 교란본 검출 점수) vs `s_m`(남성 교란본 검출 점수) 거리.

## 코드 포인트
- D 업데이트(교란본 사용): `train_faap_wgan_GD.py:245-265`
- G 업데이트(여+남 합산 손실): `train_faap_wgan_GD.py:269-318`
- Wasserstein 계산: `train_faap_wgan_GD.py:278-286`
- 교란 크기/객체성 로깅: `train_faap_wgan_GD.py:286-317`

## Train

export CUDA_VISIBLE_DEVICES=3



```bash
# 기본값 사용 (dataset_root=/home/dohyeong/Desktop/faap_dataset, output=faap_outputs)
python train_faap_wgan_GD.py --batch_size 8 --epochs 6 --device cuda

# 체크포인트/출력 경로 지정 예시
python train_faap_wgan_GD.py \
  --dataset_root /home/dohyeong/Desktop/faap_dataset \
  --detr_checkpoint /home/dohyeong/Desktop/detr/detr-r50-e632da11.pth \
  --output_dir faap_outputs/faap_outputs_gd \
  --batch_size 8 --epochs 6 --device cuda
```

## Val
```bash
# 학습된 생성기 체크포인트로 val 평가 (남/여 별도, 교란 적용/미적용 비교)

export CUDA_VISIBLE_DEVICES=2


python eval_faap.py \
  --dataset_root /home/dohyeong/Desktop/faap_dataset \
  --detr_checkpoint /home/dohyeong/Desktop/detr/detr-r50-e632da11.pth \
  --generator_checkpoint /home/dohyeong/Desktop/faap_gan/faap_outputs/faap_outputs_gd/checkpoints/epoch_0003.pth \
  --epsilon 0.12 \
  --split test \
  --batch_size 4 --device cuda \
  --results_path faap_outputs/faap_outputs_gd/test_metrics_epoch_0003.json
```

# 꼭 입실론을 0.12로 해야할까? 만약에 다른값이면? A : train_faap_wgan_GD.py에서 설정한 값과 동일하게 맞춰야 합니다. why? train시에 설정한 값과 다르면 교란 크기가 달라져서 평가가 일관되지 않기 때문입니다.


## Gen
python gen_images.py \
  --generator_checkpoint /home/dohyeong/Desktop/faap_gan/faap_outputs/faap_outputs_gd/checkpoints/epoch_0001.pth \
  --dataset_root ~/Desktop/faap_dataset \
  --split test \
  --output_root /home/dohyeong/Desktop/faap_gan/faap_outputs/faap_outputs_gd/generated_images/\
  --device cuda