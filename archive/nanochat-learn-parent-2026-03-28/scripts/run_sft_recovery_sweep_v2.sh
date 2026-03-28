#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_TAG="${1:-d24_asp48_track}"
BASE_MODEL_STEP="${2:-820230}"
NUM_ITER="${3:-300}"
EVAL_MAX="${4:-200}"

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
mkdir -p "${NOTES_DIR}"

cd "${REPO_DIR}"
source .venv/bin/activate

TS="$(date +%F_%H%M)"
SWEEP_TAG="d24_sft_recovery_v2_${TS}"
CHAIN_LOG="${NOTES_DIR}/${SWEEP_TAG}_chain.log"
RESULTS_TSV="${NOTES_DIR}/${SWEEP_TAG}_results.tsv"
SUMMARY_MD="${NOTES_DIR}/${SWEEP_TAG}_summary.md"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat-sft}"
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

run_trial() {
  local label="$1"
  local seq="$2"
  local total_batch="$3"
  local lora_lr="$4"

  local out_tag="${SWEEP_TAG}_${label}"
  local run_name="${out_tag}"
  local train_log="${NOTES_DIR}/sft_${out_tag}.log"
  local eval_log="${NOTES_DIR}/eval_${out_tag}_quick.log"

  log "START trial=${label} out=${out_tag} seq=${seq} total_batch=${total_batch} lora_lr=${lora_lr} gc=1"

  local cmd=(
    python -u -m scripts.chat_sft
    --model-tag "${BASE_MODEL_TAG}" --model-step "${BASE_MODEL_STEP}"
    --output-tag "${out_tag}" --run "${run_name}"
    --device-type cuda --dtype bfloat16
    --lora --lora-rank 64 --lora-alpha 128 --lora-dropout 0.0 --lora-lr "${lora_lr}"
    --weight-decay 0.0
    --init-lr-frac 0.25 --warmup-ratio 0.2 --warmdown-ratio 0.3
    --max-seq-len "${seq}" --device-batch-size 1 --total-batch-size "${total_batch}"
    --num-iterations "${NUM_ITER}"
    --eval-every 50 --eval-tokens 524288
    --max-grad-norm 1.0
    --keep-best-k 5 --no-save-optimizer
    --gradient-checkpoint
  )

  set +e
  "${cmd[@]}" |& tee "${train_log}" | tee -a "${CHAIN_LOG}"
  local trc=${PIPESTATUS[0]}
  set -e
  if [ "${trc}" -ne 0 ]; then
    log "FAIL trial=${label} stage=train rc=${trc} log=${train_log}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${label}" "${out_tag}" "${seq}" "${lora_lr}" "NA" "NA" "NA" "-1" >> "${RESULTS_TSV}"
    return 0
  fi

  log "EVAL trial=${label} out=${out_tag} step=${NUM_ITER} max_problems=${EVAL_MAX} cat_batch=${CAT_BATCH_SIZE}"
  set +e
  CAT_BATCH_SIZE="${CAT_BATCH_SIZE}" "${HOME}/nanochat-learn/scripts/run_chat_eval_confirm_1k.sh" "${out_tag}/best" "${NUM_ITER}" "${EVAL_MAX}" |& tee "${eval_log}" | tee -a "${CHAIN_LOG}"
  local erc=${PIPESTATUS[0]}
  set -e

  if [ "${erc}" -ne 0 ]; then
    log "FAIL trial=${label} stage=eval rc=${erc} log=${eval_log}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${label}" "${out_tag}" "${seq}" "${lora_lr}" "NA" "NA" "NA" "-1" >> "${RESULTS_TSV}"
    return 0
  fi

  local summary_file
  summary_file="$(ls -1t "${NOTES_DIR}/${out_tag}_best_s${NUM_ITER}_confirm_summary_${EVAL_MAX}_"*.txt 2>/dev/null | head -n 1 || true)"
  if [ -z "${summary_file}" ]; then
    log "FAIL trial=${label} stage=parse reason=missing_summary"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${label}" "${out_tag}" "${seq}" "${lora_lr}" "NA" "NA" "NA" "-1" >> "${RESULTS_TSV}"
    return 0
  fi

  local gsm mmlu spelling score
  gsm="$(extract_pct 'GSM8K pass@8:' "${summary_file}")"
  mmlu="$(extract_pct 'MMLU:' "${summary_file}")"
  spelling="$(extract_pct 'SpellingBee:' "${summary_file}")"

  if [ -z "${gsm}" ] || [ -z "${mmlu}" ] || [ -z "${spelling}" ]; then
    log "FAIL trial=${label} stage=parse reason=bad_metrics summary=${summary_file}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${label}" "${out_tag}" "${seq}" "${lora_lr}" "NA" "NA" "NA" "-1" >> "${RESULTS_TSV}"
    return 0
  fi

  score="$(score_of "${gsm}" "${mmlu}" "${spelling}")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "${label}" "${out_tag}" "${seq}" "${lora_lr}" "${gsm}" "${mmlu}" "${spelling}" "${score}" >> "${RESULTS_TSV}"
  log "DONE trial=${label} gsm8k_pass8=${gsm}% mmlu=${mmlu}% spelling=${spelling}% score=${score}"
}

{
  echo -e "label\tout_tag\tseq\tlora_lr\tgsm8k_pass8\tmmlu\tspellingbee\tscore"
} > "${RESULTS_TSV}"

log "SWEEP start tag=${SWEEP_TAG} base=${BASE_MODEL_TAG}@${BASE_MODEL_STEP} num_iter=${NUM_ITER} eval_max=${EVAL_MAX} cat_batch=${CAT_BATCH_SIZE}"

# Safer trials for 12GB VRAM: gradient checkpoint always on, conservative LRs.
run_trial "s1024gc_lr3e5" 1024 7680 3e-5
run_trial "s1280gc_lr3e5" 1280 7680 3e-5
run_trial "s1536gc_lr2e5" 1536 7680 2e-5

best_line="$(tail -n +2 "${RESULTS_TSV}" | awk -F '\t' '$8 != "-1" {print $0}' | sort -t $'\t' -k8,8nr | head -n 1 || true)"

{
  echo "# SFT Recovery Sweep v2 Summary"
  echo
  echo "- tag: \`${SWEEP_TAG}\`"
  echo "- base: \`${BASE_MODEL_TAG}@${BASE_MODEL_STEP}\`"
  echo "- iterations per trial: \`${NUM_ITER}\`"
  echo "- eval max problems: \`${EVAL_MAX}\`"
  echo "- categorical eval batch size: \`${CAT_BATCH_SIZE}\`"
  echo
  echo "## Results"
  echo
  echo "| label | out_tag | seq | lora_lr | gsm8k_pass8 | mmlu | spellingbee | score |"
  echo "|---|---|---:|---:|---:|---:|---:|---:|"
  tail -n +2 "${RESULTS_TSV}" | while IFS=$'\t' read -r label out seq lr gsm mmlu spelling score; do
    echo "| ${label} | \`${out}\` | ${seq} | ${lr} | ${gsm} | ${mmlu} | ${spelling} | ${score} |"
  done
  echo
  if [ -n "${best_line}" ]; then
    IFS=$'\t' read -r best_label best_out _ _ _ _ _ best_score <<< "${best_line}"
    echo "## Winner"
    echo
    echo "- label: \`${best_label}\`"
    echo "- out_tag: \`${best_out}\`"
    echo "- score: \`${best_score}\`"
    echo
    echo "Promotion command:"
    echo '```bash'
    echo "CAT_BATCH_SIZE=${CAT_BATCH_SIZE} ~/nanochat-learn/scripts/run_chat_eval_confirm_1k.sh ${best_out}/best ${NUM_ITER} 1000"
    echo '```'
  else
    echo "## Winner"
    echo
    echo "- none (all trials failed)"
  fi
} > "${SUMMARY_MD}"

log "SWEEP complete results=${RESULTS_TSV} summary=${SUMMARY_MD}"
if [ -n "${best_line}" ]; then
  IFS=$'\t' read -r best_label best_out _ _ _ _ _ best_score <<< "${best_line}"
  log "WINNER label=${best_label} out_tag=${best_out} score=${best_score}"
else
  log "WINNER none"
fi
