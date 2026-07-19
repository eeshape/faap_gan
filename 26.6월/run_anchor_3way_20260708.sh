#!/usr/bin/env bash
# =============================================================================
# Anchor 3-Way 비교 + gamma 스윕 (2026-07-08 스크립트 기반)
#
#   과제 ② (ablation): loss 추가 vs 뺀 것 비교 — 동일 조건(λ_con=2.5, τ=0.1)
#     (A) 20260708_contrastive_baseline.py        anchor 없음 (통제군)
#     (B) 20260708_contrastive_l2_anchor.py       + γ·L2 anchor
#     (C) 20260708_contrastive_centroid_anchor.py + γ·centroid anchor
#
#   과제 ① (sweep): 온도(τ=0.1) 고정, 추가 loss 가중치 γ 스윕
#     L2:       γ ∈ {0.25, 0.5, 1.0}   (0.5 = ablation run 재사용)
#     Centroid: γ ∈ {1.0, 2.0, 4.0}    (2.0 = ablation run 재사용)
#
#   각 run: 학습(8 epoch) -> 전 체크포인트 eval -> 동일 wandb project 기록.
#   최종 체크포인트가 이미 있으면 학습 skip (재실행 안전).
#   ablation 3개가 먼저 돌도록 순서 배치 — 주간보고 마감 시 앞부분만이라도 확보.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
# faap_gan 패키지 import 보장 (스크립트가 26.6월/ 하위로 이동한 뒤에도 동작)
export PYTHONPATH="$(dirname "${REPO_ROOT}")${PYTHONPATH:+:$PYTHONPATH}"

SCRIPT_DIR="26.6월"
EVAL_SCRIPT="eval_faap.py"
PYTHON="${PYTHON:-python3}"

DATASET_ROOT="${DATASET_ROOT:-/workspace/faap_dataset}"
OUT_ROOT="${OUT_ROOT:-faap_outputs/anchor_3way_20260708}"
EPOCHS="${EPOCHS:-8}"                 # OFAT 스윕 프로토콜과 동일
EPSILON_EVAL="${EPSILON_EVAL:-0.10}"  # 학습 epsilon_final 과 일치

WANDB_PROJECT="${WANDB_PROJECT:-faap-anchor-3way-20260708}"
WANDB_ENTITY="${WANDB_ENTITY:-eeshape-incheon-national-university}"

# ---- 실험 정의: "variant gamma group" ----------------------------------------
#   variant: base | l2 | cent   /  gamma: base 는 "-" (인자 미전달)
#   순서 = 실행 순서. ablation(3-way) 먼저, gamma 스윕 나중.
RUNS=(
  # 과제 ②: 3-way ablation (γ = 각 스크립트 기본값)
  "base -    ablation_3way"
  "l2   0.5  ablation_3way"
  "cent 2.0  ablation_3way"
  # 과제 ①: γ 스윕 (τ=0.1 고정) — 중심점(0.5 / 2.0)은 위에서 이미 실행돼 skip
  "l2   0.25 gamma_sweep_l2"
  "l2   1.0  gamma_sweep_l2"
  "cent 1.0  gamma_sweep_cent"
  "cent 4.0  gamma_sweep_cent"
)

script_for() {
  case "$1" in
    base) echo "${SCRIPT_DIR}/20260708_contrastive_baseline.py" ;;
    l2)   echo "${SCRIPT_DIR}/20260708_contrastive_l2_anchor.py" ;;
    cent) echo "${SCRIPT_DIR}/20260708_contrastive_centroid_anchor.py" ;;
    *)    echo "unknown variant: $1" >&2; return 1 ;;
  esac
}

for spec in "${RUNS[@]}"; do
  read -r VARIANT GAMMA GROUP <<< "$spec"
  SCRIPT="$(script_for "${VARIANT}")"

  if [[ "${VARIANT}" == "base" ]]; then
    EXP="anchor_base"
    GAMMA_ARGS=()
    TAG_G="g-"
  else
    EXP="anchor_${VARIANT}_g${GAMMA}"
    GAMMA_ARGS=(--gamma "${GAMMA}")
    TAG_G="g${GAMMA}"
  fi

  OUT_DIR="${OUT_ROOT}/${EXP}"
  FINAL_CKPT="${OUT_DIR}/checkpoints/epoch_$(printf '%04d' "$((EPOCHS - 1))").pth"

  echo "============================================================"
  echo "[RUN] ${EXP}  (variant=${VARIANT}, gamma=${GAMMA}, group=${GROUP})"
  echo "      script=${SCRIPT}"
  echo "      out=${OUT_DIR}"
  echo "============================================================"

  # ---- 1) 학습 (최종 체크포인트가 이미 있으면 skip — 스윕 중심점 재사용) ----
  if [[ -f "${FINAL_CKPT}" ]]; then
    echo "[skip-train] ${FINAL_CKPT} 이미 존재"
  else
    "${PYTHON}" "${SCRIPT}" \
      --dataset_root "${DATASET_ROOT}" \
      --output_dir "${OUT_DIR}" \
      --epochs "${EPOCHS}" \
      "${GAMMA_ARGS[@]}" \
      --wandb \
      --wandb_project "${WANDB_PROJECT}" \
      --wandb_entity "${WANDB_ENTITY}" \
      --wandb_run_name "${EXP}" \
      --wandb_group "${GROUP}" \
      --wandb_tags "anchor3way,${GROUP},${VARIANT},${TAG_G}"
  fi

  # ---- 2) 평가 (모든 체크포인트 -> 같은 project 에 epoch별 시계열 기록) ----
  shopt -s nullglob
  CKPTS=( "${OUT_DIR}"/checkpoints/epoch_*.pth )
  shopt -u nullglob
  if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "[warn] ${OUT_DIR}/checkpoints 에 체크포인트가 없음 — eval skip"
    continue
  fi
  IFS=$'\n' CKPTS=( $(printf '%s\n' "${CKPTS[@]}" | sort) ); unset IFS

  # 이미 전 epoch eval 이 끝난 run 은 skip (eval 완료 마커 파일 사용)
  EVAL_DONE="${OUT_DIR}/.eval_done_${EPOCHS}ep_e${EPSILON_EVAL}"
  if [[ -f "${EVAL_DONE}" && ${#CKPTS[@]} -eq ${EPOCHS} ]]; then
    echo "[skip-eval] ${EXP}: 이미 평가 완료 (${EVAL_DONE})"
    continue
  fi

  echo "[eval] ${EXP}: ${#CKPTS[@]} 개 체크포인트 평가"
  for CKPT in "${CKPTS[@]}"; do
    echo "  -> $(basename "${CKPT}")"
    "${PYTHON}" "${EVAL_SCRIPT}" \
      --dataset_root "${DATASET_ROOT}" \
      --generator_checkpoint "${CKPT}" \
      --epsilon "${EPSILON_EVAL}" \
      --split test \
      --wandb \
      --wandb_project "${WANDB_PROJECT}" \
      --wandb_entity "${WANDB_ENTITY}" \
      --wandb_group "${GROUP}" \
      --wandb_tags "anchor3way,${GROUP},${VARIANT},${TAG_G},eval"
  done
  [[ ${#CKPTS[@]} -eq ${EPOCHS} ]] && touch "${EVAL_DONE}"
done

echo
echo "[DONE] anchor 3-way + gamma 스윕 완료. wandb project '${WANDB_PROJECT}' 에서:"
echo "  - ablation_3way group: base vs l2(γ0.5) vs cent(γ2.0) — final/AP_gap_perturbed 비교"
echo "  - gamma_sweep_* group: x=gamma, y=final/AP_gap_perturbed"
