# FAAP-GAN 프로젝트 노트 (Claude용 영구 메모리)

> RunPod 환경: `/root`는 재시작 시 휘발. 영구 기록은 이 파일(`/workspace/faap_gan/CLAUDE.md`)에 둔다.
> 이 파일은 매 세션 자동으로 Claude 컨텍스트에 로드됨.

## 진행 방향 결정 (2026-07-19)

- **GAN/adversarial FAAP 계열은 폐기.** `train_faap_wgan_GD_7th.py`(논문 최종본, discriminator + Wasserstein)는 더 이상 진행 안 함. 앞으로는 **contrastive 계열로만** 진행.
- **단일 baseline = `contrastive_learning_baseline.py`** (루트).
  - `train_faap_fix11_contrastive_gpu_ablation_no_l2_20260415.py`의 복제본. 원본은 `26.6월/`로 이동함.
  - 정체: fix11 contrastive에서 **L2 anchoring 제거 + anchor 없음** = 가장 순수한 contrastive 통제군.
  - 이후 모든 contrastive 실험(`26.6월/20260617_*`, `26.6월/20260708_*` anchor 3-way)이 이 파일에서 파생됨.

## "baseline" 용어 주의 (3가지가 섞임)

1. **평가 기준선** = perturbation 없는 **원본 DETR**. 그림/표의 "Baseline"(Male AP 0.511 / Female AP 0.404)이 이것. 학습 파일이 아니라 `eval_faap.py`를 generator 없이 돌린 값.
2. **원조 FAAP 방법** = GAN 버전(`train_faap_wgan_GD_7th.py`). **폐기됨.**
3. **우리 baseline** = `contrastive_learning_baseline.py` (no-L2 contrastive).

baseline은 **구조·loss 기준**(anchor 없는 contrastive)이고, 하이퍼파라미터는 실험마다 다름.
no-L2 파일 기본값은 `lambda_con=1.0, temperature=0.1`이지만 7/8 실험군은 스윕 반영해 `lambda_con=2.5` 사용.

## 작업 시 전제

- contrastive 실험/비교 얘기가 나오면 `contrastive_learning_baseline.py`를 기준점으로 전제하고 시작.
- GAN/WGAN/discriminator 방향 제안은 하지 말 것.
