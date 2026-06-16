#!/bin/bash
# fix6 (3rd_fix6_gpu_20260306) 전체 에폭 자동 평가 스크립트
# epoch_0000.pth ~ epoch_XXXX.pth 를 자동으로 탐지하여 순서대로 eval 실행

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_DIR="${SCRIPT_DIR}/checkpoints"

# 체크포인트 파일 목록 수집 (epoch_XXXX.pth 형식, 오름차순 정렬)
  mapfile -t CKPTS < <(ls "${CKPT_DIR}"/epoch_*.pth 2>/dev/null | sort | awk -F'epoch_|\.pth' '$2+0 >= 4')
if [ ${#CKPTS[@]} -eq 0 ]; then
    echo "[ERROR] 체크포인트를 찾을 수 없습니다: ${CKPT_DIR}"
    exit 1
fi

echo "========================================"
echo "  fix6 Eval 시작"
echo "  체크포인트 수: ${#CKPTS[@]}"
echo "  범위: $(basename "${CKPTS[0]}") ~ $(basename "${CKPTS[-1]}")"
echo "========================================"
echo ""

for CKPT in "${CKPTS[@]}"; do
    EPOCH_NAME=$(basename "${CKPT}" .pth)
    echo "========================================"
    echo "  Evaluating ${EPOCH_NAME}"
    echo "========================================"
    python /workspace/faap_gan/eval_perturb.py \
        --generator_checkpoint "${CKPT}"
    echo ""
done

echo "========================================"
echo "  All done!"
echo "========================================"
