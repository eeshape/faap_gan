#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# baseline vs fix1 (λ, τ 개입) 비교 파이프라인
#   STEP 1) fix1 학습 (epoch 8에서 자동 종료)
#   STEP 2) baseline 체크포인트 확인
#   STEP 3) eval  : baseline / fix1 각각 (동일 설정)
#   STEP 4) t-SNE : baseline vs fix1 비교 그림
#
# 전제조건:
#   - GPU가 비어 있어야 함 (현재 도는 baseline job이 끝났거나 epoch 8에서 멈춘 뒤 실행).
#     안 그러면 STEP1 fix1 학습이 OOM 날 수 있음.
#   - baseline 체크포인트 epoch_0008.pth 가 이미 존재해야 함 (도는 job이 epoch 8 도달).
#
# baseline 과 fix1 은 코드·하이퍼파라미터가 동일(λ,τ 만 다름)하며,
# 둘 다 epochs=15 스케줄의 epoch_0008 시점에서 비교 → 교란변수 통제됨.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

# faap_gan 패키지 import 가능하게 (tsne 스크립트가 구실험파일/ 로 옮겨져 경로가 깨진 상태 보정)
export PYTHONPATH="$(dirname "${PROJECT_DIR}"):${PROJECT_DIR}:${PYTHONPATH:-}"

# ---- 공통 설정 ----
DATASET_ROOT="${DATASET_ROOT:-/workspace/faap_dataset}"
DETR_CHECKPOINT="${DETR_CHECKPOINT:-/workspace/detr/detr-r50-e632da11.pth}"
EPSILON="${EPSILON:-0.10}"          # eval perturbation bound. epoch 8 학습 ε(=0.10)와 일치.
                                    #   (기존 ablation eval 은 0.05 사용 → 그쪽에 맞추려면 EPSILON=0.05 로 실행)
                                    #   ★ baseline·fix1 동일 값 사용이 핵심
EVAL_SPLIT="${EVAL_SPLIT:-test}"    # eval 은 test split
TSNE_SPLIT="${TSNE_SPLIT:-val}"     # t-SNE 는 val split (기존 관행)
EPOCH_TAG="epoch_0008"

# ---- wandb 설정 (끄려면 WANDB=0 으로 실행) ----
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-faap-baseline-vs-fix1-20260617}"
WANDB_ENTITY="${WANDB_ENTITY:-eeshape-incheon-national-university}"
TRAIN_WANDB_FLAG=""
EVAL_WANDB_FLAG=""
if [[ "${WANDB}" == "1" ]]; then
  TRAIN_WANDB_FLAG="--wandb --wandb_project ${WANDB_PROJECT} --wandb_entity ${WANDB_ENTITY} --wandb_run_name fix1_lambda2.5_tau0.05_epoch8"
  EVAL_WANDB_FLAG="--wandb --wandb_project ${WANDB_PROJECT} --wandb_entity ${WANDB_ENTITY}"
  echo "[wandb] ON  project=${WANDB_PROJECT}"
else
  echo "[wandb] OFF (로컬 파일만 기록)"
fi

# ---- 체크포인트 경로 ----
# baseline = 현재 도는 job 의 출력 폴더 (코드·하이퍼파라미터가 baseline 파일과 100% 동일)
BASELINE_DIR="${BASELINE_DIR:-${PROJECT_DIR}/faap_outputs/faap_outputs_20260617_contrastive_baseline_hyperparameter}"
BASELINE_CKPT="${BASELINE_DIR}/checkpoints/${EPOCH_TAG}.pth"

FIX1_SCRIPT="${PROJECT_DIR}/20260617_contrastive_baseline_hyperparameter_fix1.py"
FIX1_DIR="${PROJECT_DIR}/faap_outputs/faap_outputs_20260617_contrastive_baseline_hyperparameter_fix1"
FIX1_CKPT="${FIX1_DIR}/checkpoints/${EPOCH_TAG}.pth"

# ---- 결과 출력 폴더 ----
OUT_DIR="${PROJECT_DIR}/faap_outputs/compare_baseline_vs_fix1_20260617"
mkdir -p "${OUT_DIR}"

echo "PROJECT_DIR   = ${PROJECT_DIR}"
echo "BASELINE_CKPT = ${BASELINE_CKPT}"
echo "FIX1_CKPT     = ${FIX1_CKPT}"
echo "OUT_DIR       = ${OUT_DIR}"
echo "EPSILON=${EPSILON}  EVAL_SPLIT=${EVAL_SPLIT}  TSNE_SPLIT=${TSNE_SPLIT}"

# =============================================================================
# STEP 1) fix1 학습 (λ=2.5, τ=0.05) — epoch 8 에서 자동 종료
# =============================================================================
echo ""
echo "==================================================================="
echo " STEP 1) fix1 학습 (λ=2.5, τ=0.05) — stop_epoch=8 자동 종료"
echo "==================================================================="
if [[ -f "${FIX1_CKPT}" ]]; then
  echo "[skip] 이미 존재: ${FIX1_CKPT}"
else
  python "${FIX1_SCRIPT}" --output_dir "${FIX1_DIR}" ${TRAIN_WANDB_FLAG}
fi

if [[ ! -f "${FIX1_CKPT}" ]]; then
  echo "ERROR: fix1 체크포인트가 생성되지 않음: ${FIX1_CKPT}" >&2
  exit 1
fi

# =============================================================================
# STEP 2) baseline 체크포인트 확인
# =============================================================================
echo ""
echo "==================================================================="
echo " STEP 2) baseline 체크포인트 확인"
echo "==================================================================="
if [[ ! -f "${BASELINE_CKPT}" ]]; then
  echo "ERROR: baseline 체크포인트 없음: ${BASELINE_CKPT}" >&2
  echo "       현재 도는 baseline job 이 epoch 8 까지 도달했는지 확인하세요." >&2
  exit 1
fi
echo "OK: ${BASELINE_CKPT}"

# =============================================================================
# STEP 3) eval  (baseline / fix1, 동일 설정)
# =============================================================================
BASELINE_METRICS="${OUT_DIR}/baseline_${EVAL_SPLIT}_metrics_${EPOCH_TAG}.json"
FIX1_METRICS="${OUT_DIR}/fix1_${EVAL_SPLIT}_metrics_${EPOCH_TAG}.json"

echo ""
echo "==================================================================="
echo " STEP 3a) eval baseline (${EPOCH_TAG}, split=${EVAL_SPLIT})"
echo "==================================================================="
python "${PROJECT_DIR}/eval_faap.py" \
  --dataset_root "${DATASET_ROOT}" \
  --detr_checkpoint "${DETR_CHECKPOINT}" \
  --generator_checkpoint "${BASELINE_CKPT}" \
  --epsilon "${EPSILON}" \
  --split "${EVAL_SPLIT}" \
  --results_path "${BASELINE_METRICS}" ${EVAL_WANDB_FLAG}

echo ""
echo "==================================================================="
echo " STEP 3b) eval fix1 (${EPOCH_TAG}, split=${EVAL_SPLIT})"
echo "==================================================================="
python "${PROJECT_DIR}/eval_faap.py" \
  --dataset_root "${DATASET_ROOT}" \
  --detr_checkpoint "${DETR_CHECKPOINT}" \
  --generator_checkpoint "${FIX1_CKPT}" \
  --epsilon "${EPSILON}" \
  --split "${EVAL_SPLIT}" \
  --results_path "${FIX1_METRICS}" ${EVAL_WANDB_FLAG}

# =============================================================================
# STEP 4) t-SNE 비교 (baseline vs fix1)
# =============================================================================
TSNE_PNG="${OUT_DIR}/tsne_baseline_vs_fix1_${EPOCH_TAG}.png"
echo ""
echo "==================================================================="
echo " STEP 4) t-SNE 비교 (split=${TSNE_SPLIT})"
echo "==================================================================="
python "${PROJECT_DIR}/구실험파일/eval_tsne_features.py" \
  --checkpoint  "${BASELINE_CKPT}" \
  --checkpoint2 "${FIX1_CKPT}" \
  --title  "baseline (lambda=1.0, tau=0.1)" \
  --title2 "fix1 (lambda=2.5, tau=0.05)" \
  --dataset_root "${DATASET_ROOT}" \
  --split "${TSNE_SPLIT}" \
  --max_samples 500 \
  --output "${TSNE_PNG}"

# =============================================================================
# 완료
# =============================================================================
echo ""
echo "==================================================================="
echo " 완료!"
echo "   baseline metrics : ${BASELINE_METRICS}"
echo "   fix1 metrics     : ${FIX1_METRICS}"
echo "   t-SNE 비교 그림   : ${TSNE_PNG}"
echo "==================================================================="
