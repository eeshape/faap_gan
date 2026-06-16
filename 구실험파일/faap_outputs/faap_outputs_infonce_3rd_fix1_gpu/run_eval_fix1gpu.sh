#!/bin/bash
set -e

CKPT_DIR="faap_outputs/faap_outputs_infonce_3rd_fix1_gpu/checkpoints"

for i in $(seq 0 10); do
    EPOCH=$(printf "%04d" $i)
    echo "========================================"
    echo "  Evaluating epoch_${EPOCH}"
    echo "========================================"
    python /workspace/faap_gan/eval_perturb.py \
        --generator_checkpoint "${CKPT_DIR}/epoch_${EPOCH}.pth"
done

echo "========================================"
echo "  All done!"
echo "========================================"
