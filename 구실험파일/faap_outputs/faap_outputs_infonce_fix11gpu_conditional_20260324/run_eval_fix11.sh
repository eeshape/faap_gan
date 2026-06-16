#!/bin/bash
# fix11 (infonce_fix11gpu_conditional_20260324) 평가 스크립트
# - checkpoints/epoch_*.pth 를 자동 탐지해서 순서대로 평가
# - checkpoints/best_model.pth 가 있으면 마지막에 추가 평가

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="${SCRIPT_DIR}/checkpoints"
EVAL_SCRIPT="/workspace/faap_gan/eval_perturb.py"

if [ ! -d "${CKPT_DIR}" ]; then
    echo "[ERROR] 체크포인트 디렉토리를 찾을 수 없습니다: ${CKPT_DIR}"
    exit 1
fi

# 사용자 지정 가능 옵션 (필요하면 export 해서 사용)
SPLIT="${SPLIT:-test}"
EPSILON="${EPSILON:-0.10}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/faap_dataset}"
DEVICE="${DEVICE:-cuda}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Workspace global PYTHONPATH can force loading incompatible wheels
# (e.g., cp311 numpy on Python 3.12). Run eval with a clean PYTHONPATH.
run_eval() {
    env -u PYTHONPATH PYTHONNOUSERSITE=1 "${PYTHON_BIN}" "${EVAL_SCRIPT}" "$@"
}

mapfile -t EPOCH_CKPTS < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name 'epoch_*.pth' | sort)
BEST_CKPT="${CKPT_DIR}/best_model.pth"

if [ ${#EPOCH_CKPTS[@]} -eq 0 ] && [ ! -f "${BEST_CKPT}" ]; then
    echo "[ERROR] 평가할 체크포인트가 없습니다: ${CKPT_DIR}"
    exit 1
fi

echo "========================================"
echo "  fix11 Eval 시작"
echo "  epoch 체크포인트 수: ${#EPOCH_CKPTS[@]}"
if [ ${#EPOCH_CKPTS[@]} -gt 0 ]; then
    echo "  범위: $(basename "${EPOCH_CKPTS[0]}") ~ $(basename "${EPOCH_CKPTS[-1]}")"
fi
if [ -f "${BEST_CKPT}" ]; then
    echo "  best 모델: $(basename "${BEST_CKPT}")"
fi
echo "  split=${SPLIT}, epsilon=${EPSILON}, device=${DEVICE}"
echo "========================================"
echo ""

for CKPT in "${EPOCH_CKPTS[@]}"; do
    NAME=$(basename "${CKPT}" .pth)
    echo "========================================"
    echo "  Evaluating ${NAME}"
    echo "========================================"
    run_eval \
        --generator_checkpoint "${CKPT}" \
        --split "${SPLIT}" \
        --epsilon "${EPSILON}" \
        --dataset_root "${DATASET_ROOT}" \
        --device "${DEVICE}"
    echo ""
done

if [ -f "${BEST_CKPT}" ]; then
    echo "========================================"
    echo "  Evaluating best_model"
    echo "========================================"
    run_eval \
        --generator_checkpoint "${BEST_CKPT}" \
        --split "${SPLIT}" \
        --epsilon "${EPSILON}" \
        --dataset_root "${DATASET_ROOT}" \
        --device "${DEVICE}"
    echo ""
fi

echo "========================================"
echo "  All done!"
echo "========================================"
