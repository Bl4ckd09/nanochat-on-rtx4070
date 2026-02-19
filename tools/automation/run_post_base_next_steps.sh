#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-0}" # optional supervisor pid (recommended: auto_rerun watcher pid)

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
mkdir -p "${NOTES_DIR}"

# Candidate checkpoints to compare
CAND_A_GROUP="${CAND_A_GROUP:-d18_clean200k_lora_prod_2026-02-17_1209_s1024/best}"
CAND_A_STEP="${CAND_A_STEP:-1000}"
CAND_B_GROUP="${CAND_B_GROUP:-d18_20k/best}"
CAND_B_STEP="${CAND_B_STEP:-500}"

# Pipeline knobs
COMPARE_MAX="${COMPARE_MAX:-1000}"
ARC_MAX="${ARC_MAX:-1000}"
RUN_GAPS="${RUN_GAPS:-1}"            # run ARC/HumanEval gap fills
RUN_PHASE2="${RUN_PHASE2:-0}"        # optional phase-2 hook
PHASE2_CMD="${PHASE2_CMD:-}"         # user-supplied command
RUN_PASS1="${RUN_PASS1:-0}"          # pass@1 in confirm script (default off)
OBJECTIVE="${OBJECTIVE:-mmlu}"       # mmlu | spellingbee | gsm8k_pass8
FORCE_COMPARE="${FORCE_COMPARE:-0}"  # rerun compare even if summary exists
RUN_FULL_FINAL="${RUN_FULL_FINAL:-1}"
NO_TEST_PEEK="${NO_TEST_PEEK:-1}"    # 1 => select champion by fixed objective mapping, not test metrics
# Phase-1 failure policy (retry once -> degrade settings -> continue by default)
PH1_RETRY_COUNT="${PH1_RETRY_COUNT:-1}"
PH1_CONTINUE_ON_FAILURE="${PH1_CONTINUE_ON_FAILURE:-1}"
FALLBACK_COMPARE_MAX="${FALLBACK_COMPARE_MAX:-300}"
FALLBACK_ARC_MAX="${FALLBACK_ARC_MAX:-300}"
FALLBACK_HE_N="${FALLBACK_HE_N:-5}"
FALLBACK_HE_M="${FALLBACK_HE_M:-384}"
FALLBACK_HE_K="${FALLBACK_HE_K:-30}"

# Optional manual champion freeze (avoids objective selection from test metrics)
CHAMPION_GROUP="${CHAMPION_GROUP:-}"
CHAMPION_STEP="${CHAMPION_STEP:-}"

TS="$(date +%F_%H%M)"
PIPELINE_LOG="${NOTES_DIR}/post_base_next_steps_${TS}.log"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

echo "[start] post-base pipeline @ ${TS}"
echo "[info] log=${PIPELINE_LOG}"

if [[ "${WAIT_PID}" != "0" ]]; then
  echo "[wait] waiting for PID ${WAIT_PID}"
  while kill -0 "${WAIT_PID}" 2>/dev/null; do
    sleep 30
  done
  echo "[wait] PID ${WAIT_PID} finished"
fi

cd "${REPO_DIR}"
source .venv/bin/activate

latest_summary() {
  local group="$1"
  local step="$2"
  local max_problems="$3"
  local safe_group="${group//\//_}"
  ls -1t "${NOTES_DIR}/${safe_group}_s${step}_confirm_summary_${max_problems}_"*.txt 2>/dev/null | head -n 1 || true
}

run_with_retry() {
  local label="$1"
  shift
  local attempts=$((PH1_RETRY_COUNT + 1))
  local i rc
  for ((i=1; i<=attempts; i++)); do
    echo "[run] ${label} attempt ${i}/${attempts}"
    set +e
    "$@"
    rc=$?
    set -e
    if [[ "${rc}" == "0" ]]; then
      return 0
    fi
    echo "[warn] ${label} failed (rc=${rc})"
    sleep 2
  done
  return 1
}

run_confirm_once() {
  local group="$1"
  local step="$2"
  local max_problems="$3"
  RUN_PASS1="${RUN_PASS1}" WANDB_MODE="${WANDB_MODE:-online}" \
    "${HOME}/nanochat-learn/scripts/run_chat_eval_confirm_1k.sh" "${group}" "${step}" "${max_problems}"
}

run_confirm_if_needed() {
  local group="$1"
  local step="$2"
  local summary
  summary="$(latest_summary "${group}" "${step}" "${COMPARE_MAX}")"
  if [[ -n "${summary}" && "${FORCE_COMPARE}" != "1" ]]; then
    echo "[compare] reuse summary: ${summary}"
    echo "${summary}"
    return 0
  fi

  echo "[compare] running confirmation for ${group} step=${step} max=${COMPARE_MAX}"
  if run_with_retry "confirm ${group} s${step} max=${COMPARE_MAX}" run_confirm_once "${group}" "${step}" "${COMPARE_MAX}"; then
    summary="$(latest_summary "${group}" "${step}" "${COMPARE_MAX}")"
  fi

  if [[ -z "${summary}" && "${FALLBACK_COMPARE_MAX}" != "${COMPARE_MAX}" ]]; then
    echo "[fallback] confirmation degrade: max ${COMPARE_MAX} -> ${FALLBACK_COMPARE_MAX}"
    if run_with_retry "confirm ${group} s${step} fallback max=${FALLBACK_COMPARE_MAX}" \
      run_confirm_once "${group}" "${step}" "${FALLBACK_COMPARE_MAX}"; then
      summary="$(latest_summary "${group}" "${step}" "${FALLBACK_COMPARE_MAX}")"
    fi
  fi

  if [[ -z "${summary}" ]]; then
    if [[ "${PH1_CONTINUE_ON_FAILURE}" == "1" ]]; then
      echo "[warn] confirmation failed for ${group} step=${step}; continuing without this summary"
      echo "__FAILED__"
      return 0
    fi
    echo "[error] confirmation summary not found for ${group} step=${step}"
    exit 1
  fi

  echo "${summary}"
}

run_gap_evals_once() {
  local group="$1"
  local step="$2"
  local arc_max="$3"
  local he_n="$4"
  local he_m="$5"
  local he_k="$6"
  local safe_group="${group//\//_}"
  local ts_local
  ts_local="$(date +%F_%H%M)"
  local arc_log="${NOTES_DIR}/${safe_group}_s${step}_arc_${arc_max}_${ts_local}.log"
  local he_log="${NOTES_DIR}/${safe_group}_s${step}_humaneval_pass${he_n}_${ts_local}.log"
  echo "[gaps] ARC (x=${arc_max}) -> ${arc_log}"
  WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
    -i sft -g "${group}" -s "${step}" -a "ARC-Easy|ARC-Challenge" \
    -x "${arc_max}" -b 16 --device-type cuda \
    |& tee "${arc_log}"
  echo "[gaps] HumanEval (n=${he_n}, m=${he_m}, k=${he_k}) -> ${he_log}"
  WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
    -i sft -g "${group}" -s "${step}" -a HumanEval \
    -t 0.8 -n "${he_n}" -m "${he_m}" -k "${he_k}" --device-type cuda \
    |& tee "${he_log}"
}

run_gap_evals() {
  local group="$1"
  local step="$2"
  if run_with_retry "gap evals ${group} s${step} primary" \
    run_gap_evals_once "${group}" "${step}" "${ARC_MAX}" "10" "512" "50"; then
    return 0
  fi

  if [[ "${FALLBACK_ARC_MAX}" != "${ARC_MAX}" || "${FALLBACK_HE_N}" != "10" || "${FALLBACK_HE_M}" != "512" || "${FALLBACK_HE_K}" != "50" ]]; then
    echo "[fallback] gap evals degrade for ${group} s${step}: ARC ${ARC_MAX}->${FALLBACK_ARC_MAX}, HumanEval n/m/k 10/512/50 -> ${FALLBACK_HE_N}/${FALLBACK_HE_M}/${FALLBACK_HE_K}"
    if run_with_retry "gap evals ${group} s${step} fallback" \
      run_gap_evals_once "${group}" "${step}" "${FALLBACK_ARC_MAX}" "${FALLBACK_HE_N}" "${FALLBACK_HE_M}" "${FALLBACK_HE_K}"; then
      return 0
    fi
  fi

  if [[ "${PH1_CONTINUE_ON_FAILURE}" == "1" ]]; then
    echo "[warn] gap evals failed for ${group} s${step}; continuing"
    return 0
  fi
  return 1
}

metric_pct() {
  local summary="$1"
  local label="$2"
  if [[ "${summary}" == "__FAILED__" || ! -f "${summary}" ]]; then
    echo "0"
    return 0
  fi
  rg -n "^${label}:" "${summary}" | tail -n 1 | sed -E 's/.*= ([0-9.]+)%.*/\1/' || true
}

cmp_float_gt() {
  python3 - "$1" "$2" <<'PY'
import sys
a=float(sys.argv[1]); b=float(sys.argv[2])
print("1" if a>b else "0")
PY
}

echo "[stage] apples-to-apples compare"
SUMMARY_A="$(run_confirm_if_needed "${CAND_A_GROUP}" "${CAND_A_STEP}")"
SUMMARY_B="$(run_confirm_if_needed "${CAND_B_GROUP}" "${CAND_B_STEP}")"
echo "[compare] A=${SUMMARY_A}"
echo "[compare] B=${SUMMARY_B}"

if [[ "${RUN_GAPS}" == "1" ]]; then
  echo "[stage] fill ARC/HumanEval gaps on both candidates"
  run_gap_evals "${CAND_A_GROUP}" "${CAND_A_STEP}"
  run_gap_evals "${CAND_B_GROUP}" "${CAND_B_STEP}"
fi

MMLU_A="$(metric_pct "${SUMMARY_A}" "MMLU")"
MMLU_B="$(metric_pct "${SUMMARY_B}" "MMLU")"
SPELL_A="$(metric_pct "${SUMMARY_A}" "SpellingBee")"
SPELL_B="$(metric_pct "${SUMMARY_B}" "SpellingBee")"
PASS8_A="$(metric_pct "${SUMMARY_A}" "GSM8K pass@8")"
PASS8_B="$(metric_pct "${SUMMARY_B}" "GSM8K pass@8")"

FREEZE_FILE="${NOTES_DIR}/frozen_candidates_${TS}.txt"
{
  echo "timestamp=${TS}"
  echo "candidate_a=${CAND_A_GROUP} step=${CAND_A_STEP} summary=${SUMMARY_A}"
  echo "candidate_b=${CAND_B_GROUP} step=${CAND_B_STEP} summary=${SUMMARY_B}"
  echo "metrics:"
  echo "  A: MMLU=${MMLU_A}% SpellingBee=${SPELL_A}% GSM8K_pass8=${PASS8_A}%"
  echo "  B: MMLU=${MMLU_B}% SpellingBee=${SPELL_B}% GSM8K_pass8=${PASS8_B}%"
} > "${FREEZE_FILE}"
echo "[freeze] ${FREEZE_FILE}"

if [[ -n "${CHAMPION_GROUP}" && -n "${CHAMPION_STEP}" ]]; then
  SELECTED_GROUP="${CHAMPION_GROUP}"
  SELECTED_STEP="${CHAMPION_STEP}"
  echo "[champion] manual champion override: ${SELECTED_GROUP} step=${SELECTED_STEP}"
elif [[ "${NO_TEST_PEEK}" == "1" ]]; then
  echo "[champion] no-test-peek mode: selecting by fixed objective mapping"
  case "${OBJECTIVE}" in
    mmlu|gsm8k_pass8)
      SELECTED_GROUP="${CAND_A_GROUP}"
      SELECTED_STEP="${CAND_A_STEP}"
      ;;
    spellingbee)
      SELECTED_GROUP="${CAND_B_GROUP}"
      SELECTED_STEP="${CAND_B_STEP}"
      ;;
    *)
      echo "[error] unsupported OBJECTIVE=${OBJECTIVE}"
      exit 1
      ;;
  esac
else
  echo "[champion] selecting by objective=${OBJECTIVE} (test-driven; set CHAMPION_* to avoid this)"
  case "${OBJECTIVE}" in
    mmlu)
      pick="$(cmp_float_gt "${MMLU_A:-0}" "${MMLU_B:-0}")"
      ;;
    spellingbee)
      pick="$(cmp_float_gt "${SPELL_A:-0}" "${SPELL_B:-0}")"
      ;;
    gsm8k_pass8)
      pick="$(cmp_float_gt "${PASS8_A:-0}" "${PASS8_B:-0}")"
      ;;
    *)
      echo "[error] unsupported OBJECTIVE=${OBJECTIVE}"
      exit 1
      ;;
  esac
  if [[ "${pick}" == "1" ]]; then
    SELECTED_GROUP="${CAND_A_GROUP}"
    SELECTED_STEP="${CAND_A_STEP}"
  else
    SELECTED_GROUP="${CAND_B_GROUP}"
    SELECTED_STEP="${CAND_B_STEP}"
  fi
fi

CHAMP_FILE="${NOTES_DIR}/selected_champion_${TS}.txt"
{
  echo "objective=${OBJECTIVE}"
  echo "selected_group=${SELECTED_GROUP}"
  echo "selected_step=${SELECTED_STEP}"
  echo "selection_time=$(date -Is)"
} > "${CHAMP_FILE}"
echo "[champion] ${CHAMP_FILE}"

if [[ "${RUN_PHASE2}" == "1" ]]; then
  if [[ -n "${PHASE2_CMD}" ]]; then
    echo "[phase2] running user command"
    bash -lc "${PHASE2_CMD}"
  else
    echo "[phase2] RUN_PHASE2=1 but PHASE2_CMD is empty; skipping phase-2."
  fi
else
  echo "[phase2] skipped (RUN_PHASE2=${RUN_PHASE2})"
fi

echo "[stage] quick validation pack on champion"
VALID_TS="$(date +%F_%H%M)"
SAFE_SEL="${SELECTED_GROUP//\//_}"
QPASS8_LOG="${NOTES_DIR}/${SAFE_SEL}_s${SELECTED_STEP}_quick_pass8_${VALID_TS}.log"
QMMLU_LOG="${NOTES_DIR}/${SAFE_SEL}_s${SELECTED_STEP}_quick_mmlu_${VALID_TS}.log"
QSPELL_LOG="${NOTES_DIR}/${SAFE_SEL}_s${SELECTED_STEP}_quick_spellingbee_${VALID_TS}.log"

WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
  -i sft -g "${SELECTED_GROUP}" -s "${SELECTED_STEP}" -a GSM8K \
  -t 0.7 -n 8 -m 1024 -k 50 -x 300 --device-type cuda \
  |& tee "${QPASS8_LOG}"

WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
  -i sft -g "${SELECTED_GROUP}" -s "${SELECTED_STEP}" -a MMLU \
  -x 200 --device-type cuda \
  |& tee "${QMMLU_LOG}"

WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
  -i sft -g "${SELECTED_GROUP}" -s "${SELECTED_STEP}" -a SpellingBee \
  -x 200 --device-type cuda \
  |& tee "${QSPELL_LOG}"

if [[ "${RUN_FULL_FINAL}" == "1" ]]; then
  echo "[stage] full final eval (one pass, no reselection)"
  RUN_PASS1="${RUN_PASS1}" WANDB_MODE="${WANDB_MODE:-online}" \
    "${HOME}/nanochat-learn/scripts/run_chat_eval_confirm_1k.sh" "${SELECTED_GROUP}" "${SELECTED_STEP}" 1000

  FINAL_TS="$(date +%F_%H%M)"
  FARC_LOG="${NOTES_DIR}/${SAFE_SEL}_s${SELECTED_STEP}_final_arc1k_${FINAL_TS}.log"
  FHE_LOG="${NOTES_DIR}/${SAFE_SEL}_s${SELECTED_STEP}_final_humaneval_pass10_${FINAL_TS}.log"
  WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
    -i sft -g "${SELECTED_GROUP}" -s "${SELECTED_STEP}" -a "ARC-Easy|ARC-Challenge" \
    -x 1000 -b 16 --device-type cuda \
    |& tee "${FARC_LOG}"
  WANDB_MODE="${WANDB_MODE:-online}" .venv/bin/python -m scripts.chat_eval \
    -i sft -g "${SELECTED_GROUP}" -s "${SELECTED_STEP}" -a HumanEval \
    -t 0.8 -n 10 -m 512 -k 50 --device-type cuda \
    |& tee "${FHE_LOG}"
fi

echo "[done] post-base pipeline complete"
