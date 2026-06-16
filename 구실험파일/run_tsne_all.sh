#!/bin/bash
# fix11 전체 epoch t-SNE 시각화 스크립트
# 사용법: bash run_tsne_all.sh

cd /home/dohyeong/Desktop/faap_gan

SCRIPT="eval_tsne_features.py"
DATASET_ROOT="/workspace/faap_dataset"

# 체크포인트 디렉토리
CKPT_DIR1="faap_outputs/faap_outputs_fix11_contrastive_gpu_20260410/checkpoints"
CKPT_DIR2="faap_outputs/faap_outputs_fix11_contrastive_gpu_ablation_no_l2_20260415/checkpoints"

# 결과 저장 디렉토리
OUT_DIR="faap_outputs/tsne_results_fix11"
mkdir -p "$OUT_DIR"

echo "============================================"
echo " fix11 t-SNE 전체 epoch 시각화"
echo "============================================"

for EPOCH in $(seq -w 0 14); do
    CKPT1="${CKPT_DIR1}/epoch_00${EPOCH}.pth"
    CKPT2="${CKPT_DIR2}/epoch_00${EPOCH}.pth"

    echo ""
    echo ">>> Epoch ${EPOCH} - 비교 모드 (w/ L2 vs w/o L2)"
    echo "    ckpt1: ${CKPT1}"
    echo "    ckpt2: ${CKPT2}"

    python "$SCRIPT" \
        --checkpoint "$CKPT1" \
        --checkpoint2 "$CKPT2" \
        --title "fix11 w/ L2" \
        --title2 "fix11 w/o L2 (ablation)" \
        --dataset_root "$DATASET_ROOT" \
        --split val \
        --max_samples 500 \
        --output "${OUT_DIR}/tsne_compare_epoch${EPOCH}.png"

    if [ $? -eq 0 ]; then
        echo "    [OK] epoch ${EPOCH} 완료"
    else
        echo "    [FAIL] epoch ${EPOCH} 실패"
    fi
done

echo ""
echo "============================================"
echo " 완료! 결과 위치: ${OUT_DIR}/"
echo "============================================"
ls -la "$OUT_DIR"
