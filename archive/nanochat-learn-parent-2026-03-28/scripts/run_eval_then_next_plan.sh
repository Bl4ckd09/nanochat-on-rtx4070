#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_TAG="${1:-d24_asp48_track}"
BASE_MODEL_STEP="${2:-820230}"
EVAL_MAX="${3:-200}"

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
mkdir -p "${NOTES_DIR}"

cd "${REPO_DIR}"
source .venv/bin/activate

TS="$(date +%F_%H%M)"
PLAN_TAG="d24_eval_then_next_${TS}"
CHAIN_LOG="${NOTES_DIR}/${PLAN_TAG}_chain.log"
RESULTS_TSV="${NOTES_DIR}/${PLAN_TAG}_results.tsv"
SUMMARY_MD="${NOTES_DIR}/${PLAN_TAG}_summary.md"

export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export CAT_BATCH_SIZE="${CAT_BATCH_SIZE:-1}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${CHAIN_LOG}"
}

extract_pct() {
  local pattern="$1"
  local file="$2"
  rg "^${pattern}" "${file}" | sed -E 's/.*= ([0-9.]+)%.*/\1/' | tail -n 1
}

score_of() {
  local gsm="$1"
  local mmlu="$2"
  local spelling="$3"
  awk -v g="$gsm" -v m="$mmlu" -v s="$spelling" 'BEGIN { printf "%.4f", (3.0*g + 1.0*m + 0.5*s) }'
}

run_eval_candidate() {
  local label="$1"
  local model_group="$2"
  local step="$3"
  local safe_group="${model_group//\//_}"

  log "EVAL start label=${label} group=${model_group} step=${step} max=${EVAL_MAX} cat_batch=${CAT_BATCH_SIZE}"
  set +e
  CAT_BATCH_SIZE="${CAT_BATCH_SIZE}" "${HOME}/nanochat-learn/scripts/run_chat_eval_confirm_1k.sh" "${model_group}" "${step}" "${EVAL_MAX}" |& tee -a "${CHAIN_LOG}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [ "${rc}" -ne 0 ]; then
    log "EVAL fail label=${label} rc=${rc}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${label}" "${model_group}" "${step}" "NA" "NA" "NA" "-1" "eval_failed" >> "${RESULTS_TSV}"
    return 0
  fi

  local summary_file
  summary_file="$(ls -1t "${NOTES_DIR}/${safe_group}_s${step}_confirm_summary_${EVAL_MAX}_"*.txt 2>/dev/null | head -n 1 || true)"
  if [ -z "${summary_file}" ]; then
    log "EVAL parse fail label=${label} reason=missing_summary"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${label}" "${model_group}" "${step}" "NA" "NA" "NA" "-1" "summary_missing" >> "${RESULTS_TSV}"
    return 0
  fi

  local gsm mmlu spelling score
  gsm="$(extract_pct 'GSM8K pass@8:' "${summary_file}")"
  mmlu="$(extract_pct 'MMLU:' "${summary_file}")"
  spelling="$(extract_pct 'SpellingBee:' "${summary_file}")"
  if [ -z "${gsm}" ] || [ -z "${mmlu}" ] || [ -z "${spelling}" ]; then
    log "EVAL parse fail label=${label} reason=bad_metrics summary=${summary_file}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${label}" "${model_group}" "${step}" "NA" "NA" "NA" "-1" "metric_parse_failed" >> "${RESULTS_TSV}"
    return 0
  fi
  score="$(score_of "${gsm}" "${mmlu}" "${spelling}")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${label}" "${model_group}" "${step}" "${gsm}" "${mmlu}" "${spelling}" "${score}" "ok" >> "${RESULTS_TSV}"
  log "EVAL done label=${label} gsm8k_pass8=${gsm}% mmlu=${mmlu}% spelling=${spelling}% score=${score}"
}

{
  echo -e "label\tmodel_group\tstep\tgsm8k_pass8\tmmlu\tspellingbee\tscore\tstatus"
} > "${RESULTS_TSV}"

log "PLAN start tag=${PLAN_TAG} base=${BASE_MODEL_TAG}@${BASE_MODEL_STEP} eval_max=${EVAL_MAX} cat_batch=${CAT_BATCH_SIZE}"

# Control + latest recovery checkpoints that already exist on disk.
run_eval_candidate "control_r32_s1000" "d24_r32_lora_nextbest_2026-03-19_1310_s1536_gc/best" "1000"
run_eval_candidate "recovery_s1536_s300" "d24_sft_recovery_2026-03-20_1216_s1536gc_lr3e5/best" "300"
run_eval_candidate "recovery_s1280_s50" "d24_sft_recovery_2026-03-20_1216_s1280_lr5e5/best" "50"
run_eval_candidate "recovery_s1024_s50" "d24_sft_recovery_2026-03-20_1216_s1024_lr5e5/best" "50"

best_line="$(tail -n +2 "${RESULTS_TSV}" | awk -F '\t' '$7 != "-1" {print $0}' | sort -t $'\t' -k7,7nr | head -n 1 || true)"
control_line="$(tail -n +2 "${RESULTS_TSV}" | awk -F '\t' '$1 == "control_r32_s1000" {print $0}' | head -n 1 || true)"
control_score="NA"
if [ -n "${control_line}" ]; then
  IFS=$'\t' read -r _ _ _ _ _ _ control_score _ <<< "${control_line}"
fi

{
  echo "# Eval Then Next Plan Summary"
  echo
  echo "- tag: \`${PLAN_TAG}\`"
  echo "- base: \`${BASE_MODEL_TAG}@${BASE_MODEL_STEP}\`"
  echo "- quick eval max problems: \`${EVAL_MAX}\`"
  echo "- categorical eval batch size: \`${CAT_BATCH_SIZE}\`"
  echo
  echo "## Quick Eval Results"
  echo
  echo "| label | model_group | step | gsm8k_pass8 | mmlu | spellingbee | score | status |"
  echo "|---|---|---:|---:|---:|---:|---:|---|"
  tail -n +2 "${RESULTS_TSV}" | while IFS=$'\t' read -r label group step gsm mmlu spelling score status; do
    echo "| ${label} | \`${group}\` | ${step} | ${gsm} | ${mmlu} | ${spelling} | ${score} | ${status} |"
  done
  echo
} > "${SUMMARY_MD}"

if [ -n "${best_line}" ]; then
  IFS=$'\t' read -r best_label best_group best_step best_gsm best_mmlu best_spelling best_score best_status <<< "${best_line}"
  log "BEST quick-eval label=${best_label} score=${best_score} group=${best_group} step=${best_step}"
  {
    echo "## Best Candidate"
    echo
    echo "- label: \`${best_label}\`"
    echo "- model_group: \`${best_group}\`"
    echo "- step: \`${best_step}\`"
    echo "- score: \`${best_score}\`"
    echo "- control score: \`${control_score}\`"
    echo
  } >> "${SUMMARY_MD}"

  if [ "${best_label}" != "control_r32_s1000" ]; then
    log "CONFIRM start best candidate on 1000 problems"
    set +e
    CAT_BATCH_SIZE="${CAT_BATCH_SIZE}" "${HOME}/nanochat-learn/scripts/run_chat_eval_confirm_1k.sh" "${best_group}" "${best_step}" "1000" |& tee -a "${CHAIN_LOG}"
    confirm_rc=${PIPESTATUS[0]}
    set -e
    log "CONFIRM end rc=${confirm_rc}"
    {
      echo "## Full Confirm (1000 problems)"
      echo
      echo "- candidate: \`${best_group}\` step \`${best_step}\`"
      echo "- rc: \`${confirm_rc}\`"
      echo
    } >> "${SUMMARY_MD}"
  else
    log "SKIP full confirm because control candidate remains best on quick eval"
    {
      echo "## Full Confirm (1000 problems)"
      echo
      echo "- skipped: control candidate remained best on quick eval"
      echo
    } >> "${SUMMARY_MD}"
  fi
else
  log "No successful quick-eval candidate found; skipping full confirm."
  {
    echo "## Best Candidate"
    echo
    echo "- none"
    echo
    echo "## Full Confirm (1000 problems)"
    echo
    echo "- skipped: no successful quick-eval candidate"
    echo
  } >> "${SUMMARY_MD}"
fi

log "NEXT plan: launch corrected recovery sweep v2"
"${HOME}/nanochat-learn/scripts/run_sft_recovery_sweep_v2.sh" "${BASE_MODEL_TAG}" "${BASE_MODEL_STEP}" "300" "${EVAL_MAX}" |& tee -a "${CHAIN_LOG}"
log "PLAN complete summary=${SUMMARY_MD} results=${RESULTS_TSV}"
