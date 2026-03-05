#!/bin/bash
set -e

CKPT_DIR="faap_outputs/faap_outputs_infonce_3rd_fix5_gpu_20260305/checkpoints"

for i in $(seq 10 14); do
    EPOCH=$(printf "%04d" $i)
    echo "========================================"
    echo "  Evaluating epoch_${EPOCH}"
    echo "========================================"
    python /workspace/faap_gan/eval_faap.py \
        --generator_checkpoint "${CKPT_DIR}/epoch_${EPOCH}.pth"
done

echo "========================================"
echo "  All done!"
echo "========================================"
