#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

DATASET_ROOT="${DATASET_ROOT:-/workspace/faap_dataset}"
DETR_CHECKPOINT="${DETR_CHECKPOINT:-/workspace/detr/detr-r50-e632da11.pth}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_DIR}/faap_outputs/faap_outputs_fix11_contrastive_gpu_ablation_no_l2_20260415}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${EXPERIMENT_DIR}/checkpoints}"
SPLIT="${SPLIT:-test}"
EPSILON="${EPSILON:-0.05}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

mapfile -t CHECKPOINTS < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'epoch_*.pth' | sort)

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "No checkpoints found in: ${CHECKPOINT_DIR}" >&2
  exit 1
fi

echo "Found ${#CHECKPOINTS[@]} checkpoints in ${CHECKPOINT_DIR}"

for CKPT in "${CHECKPOINTS[@]}"; do
  EPOCH_NAME="$(basename "${CKPT}" .pth)"
  RESULTS_PATH="${EXPERIMENT_DIR}/${SPLIT}_metrics_${EPOCH_NAME}.json"

  echo "=== Evaluating ${EPOCH_NAME} (${SPLIT}) ==="
  python "${PROJECT_DIR}/eval_faap.py" \
    --dataset_root "${DATASET_ROOT}" \
    --detr_checkpoint "${DETR_CHECKPOINT}" \
    --generator_checkpoint "${CKPT}" \
    --epsilon "${EPSILON}" \
    --split "${SPLIT}" \
    --results_path "${RESULTS_PATH}"
done

echo "All evaluations completed."
