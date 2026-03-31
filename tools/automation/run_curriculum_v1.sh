#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_TAG="${1:-d24_asp48_track}"
BASE_MODEL_STEP="${2:-820230}"
STAGE_A_ITERS="${STAGE_A_ITERS:-200}"
STAGE_B_ITERS="${STAGE_B_ITERS:-100}"
FREEZE_LAYERS="${FREEZE_LAYERS:-20}"
STAGE_A_PRESET="${STAGE_A_PRESET:-reasoning_focus_v4}"
STAGE_B_PRESET="${STAGE_B_PRESET:-curriculum_boost_v1}"
SEQ_LEN="${SEQ_LEN:-768}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-7680}"
STAGE_A_INIT_LR_FRAC="${STAGE_A_INIT_LR_FRAC:-0.05}"
STAGE_B_INIT_LR_FRAC="${STAGE_B_INIT_LR_FRAC:-0.03}"
QUICK_GATE_MMLU_MIN="${QUICK_GATE_MMLU_MIN:-27.0}"
FULL_CONFIRM_MAX_PROBLEMS="${FULL_CONFIRM_MAX_PROBLEMS:-1000}"
SEED="${SEED:-42}"
EVAL_SEED="${EVAL_SEED:-42}"
DETERMINISTIC="${DETERMINISTIC:-0}"

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
TS="$(date +%F_%H%M)"
RUN_BASE="d24_r32_adamw_curriculum_v1_${TS}"
MASTER_LOG="${NOTES_DIR}/${RUN_BASE}.log"
META_FILE="${NOTES_DIR}/${RUN_BASE}_meta.env"
DECISION_FILE="${NOTES_DIR}/${RUN_BASE}_decision.md"

mkdir -p "${NOTES_DIR}"
cd "${REPO_DIR}"
source .venv/bin/activate

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat-sft}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-nanochat-eval}"
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export CAT_BATCH_SIZE="${CAT_BATCH_SIZE:-1}"

log() {
  echo "$1" | tee -a "${MASTER_LOG}"
}

resolve_best_checkpoint() {
  local output_tag="$1"
  python3 - "${output_tag}" <<'PY'
import json
import math
import sys
from pathlib import Path

output_tag = sys.argv[1]
root = Path.home() / ".cache" / "nanochat" / "chatsft_checkpoints" / output_tag / "best"
records = []
if root.exists():
    for meta_path in root.glob("meta_*.json"):
        try:
            data = json.loads(meta_path.read_text())
            step = int(data.get("step"))
            val_bpb = float(data.get("val_bpb", math.inf))
        except Exception:
            continue
        records.append((val_bpb, step))
if not records:
    raise SystemExit(1)
val_bpb, step = min(records)
print(step, val_bpb)
PY
}

extract_summary_path() {
  local eval_log="$1"
  rg '^summary=' "${eval_log}" | tail -n 1 | sed 's/^summary=//'
}

read_eval_gate_metrics() {
  local summary_file="$1"
  python3 - "${summary_file}" <<'PY'
import math
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text() if Path(sys.argv[1]).exists() else ""

def extract(label):
    m = re.search(rf"{re.escape(label)}:\s+\d+/\d+\s+=\s+([0-9.]+)%", text)
    return float(m.group(1)) if m else math.nan

print(extract("GSM8K pass@1"), extract("GSM8K pass@8"), extract("MMLU"), extract("SpellingBee"))
PY
}

run_train() {
  local model_source="$1"
  local model_tag="$2"
  local model_step="$3"
  local output_tag="$4"
  local run_name="$5"
  local dataset_preset="$6"
  local num_iterations="$7"
  local init_lr_frac="$8"
  local log_file="$9"

  local -a cmd=(
    .venv/bin/python -u -m scripts.chat_sft
    --model-source "${model_source}"
    --model-tag "${model_tag}" --model-step "${model_step}"
    --output-tag "${output_tag}" --run "${run_name}"
    --device-type cuda --dtype bfloat16
    --seed "${SEED}"
    --adamw-only --optimizer paged_adamw8bit --weight-decay 0.1
    --embedding-lr 0.3 --unembedding-lr 0.004 --matrix-lr 0.02
    --init-lr-frac "${init_lr_frac}" --warmup-ratio 0.2 --warmdown-ratio 0.3
    --max-seq-len "${SEQ_LEN}" --device-batch-size 1 --total-batch-size "${TOTAL_BATCH_SIZE}"
    --num-iterations "${num_iterations}"
    --eval-every 25 --eval-tokens 524288
    --dataset-preset "${dataset_preset}"
    --freeze-layers "${FREEZE_LAYERS}" --freeze-embeddings --freeze-scalars
    --gradient-checkpoint --max-grad-norm 1.0
    --keep-best-k 1 --no-save-optimizer
  )
  if [[ "${DETERMINISTIC}" == "1" ]]; then
    cmd+=(--deterministic)
  fi

  set +e
  "${cmd[@]}" |& tee "${log_file}" | tee -a "${MASTER_LOG}"
  local rc=${PIPESTATUS[0]}
  set -e
  return ${rc}
}

run_quick_then_full_eval() {
  local model_group="$1"
  local step="$2"
  local eval_prefix="$3"
  local quick_log="${NOTES_DIR}/${eval_prefix}_quick.log"
  local full_log="${NOTES_DIR}/${eval_prefix}_full.log"

  set +e
  RUN_PASS1=1 SKIP_PASS1=0 SKIP_PASS8=1 SKIP_MMLU=0 SKIP_SPELLING=1 EVAL_SEED="${EVAL_SEED}" DETERMINISTIC="${DETERMINISTIC}" \
    "${HOME}/nanochat-learn/scripts/run_chat_eval_confirm_1k.sh" "${model_group}" "${step}" 250 |& tee "${quick_log}" | tee -a "${MASTER_LOG}"
  local quick_rc=${PIPESTATUS[0]}
  set -e
  local quick_summary
  quick_summary="$(extract_summary_path "${quick_log}")"
  if [[ -z "${quick_summary}" || ! -f "${quick_summary}" ]]; then
    log "[error] quick eval summary missing"
    return 1
  fi
  local pass1 pass8 mmlu spelling
  read -r pass1 pass8 mmlu spelling <<< "$(read_eval_gate_metrics "${quick_summary}")"
  log "[gate] curriculum quick metrics: pass1=${pass1}% mmlu=${mmlu}%"

  if ! python3 - "${mmlu}" "${QUICK_GATE_MMLU_MIN}" <<'PY'
import sys
mmlu=float(sys.argv[1])
threshold=float(sys.argv[2])
raise SystemExit(0 if mmlu >= threshold else 1)
PY
  then
    cat > "${DECISION_FILE}" <<EOF2
# ${RUN_BASE} decision

- status: quick_gate_failed
- quick_summary: ${quick_summary}
- gsm8k_pass1: ${pass1}%
- mmlu: ${mmlu}%
- reason: curriculum_v1 did not clear the quick MMLU gate
EOF2
    log "[decision] ${DECISION_FILE}"
    return 2
  fi

  set +e
  RUN_PASS1=0 SKIP_PASS1=1 SKIP_PASS8=0 SKIP_MMLU=0 SKIP_SPELLING=0 EVAL_SEED="${EVAL_SEED}" DETERMINISTIC="${DETERMINISTIC}" \
    "${HOME}/nanochat-learn/scripts/run_chat_eval_confirm_1k.sh" "${model_group}" "${step}" "${FULL_CONFIRM_MAX_PROBLEMS}" |& tee "${full_log}" | tee -a "${MASTER_LOG}"
  local full_rc=${PIPESTATUS[0]}
  set -e
  local full_summary
  full_summary="$(extract_summary_path "${full_log}")"
  local fpass1 fpass8 fmmlu fspell
  read -r fpass1 fpass8 fmmlu fspell <<< "$(read_eval_gate_metrics "${full_summary}")"
  cat > "${DECISION_FILE}" <<EOF2
# ${RUN_BASE} decision

- status: full_confirm_complete
- quick_summary: ${quick_summary}
- full_summary: ${full_summary}
- gsm8k_pass8: ${fpass8}%
- mmlu: ${fmmlu}%
- spellingbee: ${fspell}%
- full_rc: ${full_rc}
EOF2
  log "[decision] ${DECISION_FILE}"
  return ${full_rc}
}

{
  echo "RUN_BASE=${RUN_BASE}"
  echo "BASE_MODEL_TAG=${BASE_MODEL_TAG}"
  echo "BASE_MODEL_STEP=${BASE_MODEL_STEP}"
  echo "STAGE_A_PRESET=${STAGE_A_PRESET}"
  echo "STAGE_B_PRESET=${STAGE_B_PRESET}"
  echo "STAGE_A_ITERS=${STAGE_A_ITERS}"
  echo "STAGE_B_ITERS=${STAGE_B_ITERS}"
  echo "FREEZE_LAYERS=${FREEZE_LAYERS}"
  echo "SEQ_LEN=${SEQ_LEN}"
  echo "TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE}"
  echo "STAGE_A_INIT_LR_FRAC=${STAGE_A_INIT_LR_FRAC}"
  echo "STAGE_B_INIT_LR_FRAC=${STAGE_B_INIT_LR_FRAC}"
  echo "SEED=${SEED}"
  echo "EVAL_SEED=${EVAL_SEED}"
  echo "DETERMINISTIC=${DETERMINISTIC}"
  echo "MASTER_LOG=${MASTER_LOG}"
} > "${META_FILE}"

log "[info] meta: ${META_FILE}"
log "[plan] curriculum_v1 stageA=${STAGE_A_PRESET}/${STAGE_A_ITERS} -> stageB=${STAGE_B_PRESET}/${STAGE_B_ITERS}"

STAGE_A_OUT="${RUN_BASE}_stageA"
STAGE_A_RUN="${RUN_BASE}_stageA"
STAGE_A_LOG="${NOTES_DIR}/${STAGE_A_RUN}.log"
if run_train base "${BASE_MODEL_TAG}" "${BASE_MODEL_STEP}" "${STAGE_A_OUT}" "${STAGE_A_RUN}" "${STAGE_A_PRESET}" "${STAGE_A_ITERS}" "${STAGE_A_INIT_LR_FRAC}" "${STAGE_A_LOG}"; then
  read -r STAGE_A_STEP STAGE_A_VAL <<< "$(resolve_best_checkpoint "${STAGE_A_OUT}")"
  log "[stage_a] best_step=${STAGE_A_STEP} val_bpb=${STAGE_A_VAL}"
else
  log "[error] stage A training failed"
  exit 1
fi

STAGE_B_OUT="${RUN_BASE}_stageB"
STAGE_B_RUN="${RUN_BASE}_stageB"
STAGE_B_LOG="${NOTES_DIR}/${STAGE_B_RUN}.log"
if run_train sft "${STAGE_A_OUT}/best" "${STAGE_A_STEP}" "${STAGE_B_OUT}" "${STAGE_B_RUN}" "${STAGE_B_PRESET}" "${STAGE_B_ITERS}" "${STAGE_B_INIT_LR_FRAC}" "${STAGE_B_LOG}"; then
  read -r STAGE_B_STEP STAGE_B_VAL <<< "$(resolve_best_checkpoint "${STAGE_B_OUT}")"
  log "[stage_b] best_step=${STAGE_B_STEP} val_bpb=${STAGE_B_VAL}"
else
  log "[error] stage B training failed"
  exit 1
fi

run_quick_then_full_eval "${STAGE_B_OUT}/best" "${STAGE_B_STEP}" "${RUN_BASE}_stageB"
