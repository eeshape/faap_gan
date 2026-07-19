#!/usr/bin/env bash
# =============================================================================
# OFAT (One-Factor-At-A-Time) 하이퍼파라미터 스윕
#   - baseline 스크립트 하나로 lambda_con / temperature 를 "각각 개별로" 스윕
#   - anchor(중심점) = (lambda=1.0, tau=0.1) = baseline
#   - A. lambda 스윕 (tau=0.1 고정): 0.5, 1.0, 2.5, 5.0, 10.0
#   - B. temperature 스윕 (lambda=1.0 고정): 0.03, 0.05, 0.1, 0.2, 0.5
#   - 공통 중심 (1.0, 0.1)은 한 번만 실행
#   - 각 run: 학습 -> 최종 체크포인트 eval -> wandb 동일 project 에 기록
# =============================================================================
set -euo pipefail

# ---- 공통 설정 (모든 run 동일하게 고정해야 비교가 깨끗하다) -------------------
SCRIPT="20260617_contrastive_baseline_hyperparameter.py"
EVAL_SCRIPT="eval_faap.py"
PYTHON="${PYTHON:-python3}"

DATASET_ROOT="${DATASET_ROOT:-/workspace/faap_dataset}"
OUT_ROOT="${OUT_ROOT:-faap_outputs/sweep_ofat_20260617}"
EPOCHS="${EPOCHS:-8}"                          # 모든 실험 8 epoch (epoch_0000 ~ epoch_0007) 까지만
EPSILON_EVAL="${EPSILON_EVAL:-0.10}"          # eval 섭동 bound (학습 epsilon_final 과 일치)

WANDB_PROJECT="${WANDB_PROJECT:-faap-ofat-sweep-20260617}"
WANDB_ENTITY="${WANDB_ENTITY:-eeshape-incheon-national-university}"

# ---- 스윕 정의 ---------------------------------------------------------------
# 형식: "lambda tau group"
RUNS=(
  # A. lambda 스윕 (tau=0.1 고정)
  "0.5  0.1  lambda_sweep"
  "1.0  0.1  lambda_sweep"   # = 중심점 (baseline)
  "2.5  0.1  lambda_sweep"
  "5.0  0.1  lambda_sweep"
  "10.0 0.1  lambda_sweep"
  # B. temperature 스윕 (lambda=1.0 고정)
  "1.0  0.03 tau_sweep"
  "1.0  0.05 tau_sweep"
  "1.0  0.1  tau_sweep"      # = 중심점 (baseline) — 중복, 자동 skip
  "1.0  0.2  tau_sweep"
  "1.0  0.5  tau_sweep"
)

declare -A SEEN   # (lambda,tau) 중복 실행 방지 (중심점)

for spec in "${RUNS[@]}"; do
  read -r LAMBDA TAU GROUP <<< "$spec"
  key="${LAMBDA}_${TAU}"
  if [[ -n "${SEEN[$key]:-}" ]]; then
    echo "[skip-dup] lambda=${LAMBDA} tau=${TAU} (이미 ${SEEN[$key]} 에서 실행됨)"
    continue
  fi
  SEEN[$key]="$GROUP"

  EXP="ofat_l${LAMBDA}_t${TAU}"
  OUT_DIR="${OUT_ROOT}/${EXP}"
  FINAL_CKPT="${OUT_DIR}/checkpoints/epoch_$(printf '%04d' "$((EPOCHS - 1))").pth"

  echo "============================================================"
  echo "[RUN] ${EXP}  (lambda=${LAMBDA}, tau=${TAU}, group=${GROUP})"
  echo "      out=${OUT_DIR}"
  echo "============================================================"

  # ---- 1) 학습 (최종 체크포인트가 이미 있으면 skip) ----
  if [[ -f "${FINAL_CKPT}" ]]; then
    echo "[skip-train] ${FINAL_CKPT} 이미 존재"
  else
    "${PYTHON}" "${SCRIPT}" \
      --dataset_root "${DATASET_ROOT}" \
      --output_dir "${OUT_DIR}" \
      --epochs "${EPOCHS}" \
      --lambda_con "${LAMBDA}" \
      --temperature "${TAU}" \
      --wandb \
      --wandb_project "${WANDB_PROJECT}" \
      --wandb_entity "${WANDB_ENTITY}" \
      --wandb_run_name "${EXP}" \
      --wandb_group "${GROUP}" \
      --wandb_tags "ofat,${GROUP},l${LAMBDA},t${TAU}"
  fi

  # ---- 2) 평가 (모든 체크포인트 → 동일 eval run 에 epoch별 시계열로 누적) ----
  shopt -s nullglob
  CKPTS=( "${OUT_DIR}"/checkpoints/epoch_*.pth )
  shopt -u nullglob
  if [[ ${#CKPTS[@]} -eq 0 ]]; then
    echo "[warn] ${OUT_DIR}/checkpoints 에 체크포인트가 없음 — eval skip"
    continue
  fi
  # epoch 오름차순으로 평가 → wandb summary 는 최종(가장 큰) epoch 값으로 수렴
  IFS=$'\n' CKPTS=( $(printf '%s\n' "${CKPTS[@]}" | sort) ); unset IFS
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
      --wandb_tags "ofat,${GROUP},l${LAMBDA},t${TAU},eval"
  done
done

echo
echo "[DONE] OFAT 스윕 완료. wandb project '${WANDB_PROJECT}' 에서:"
echo "  - Scatter: x=lambda_con (또는 temperature), y=final/AP_gap_perturbed"
echo "  - group 으로 lambda_sweep / tau_sweep 필터링"
