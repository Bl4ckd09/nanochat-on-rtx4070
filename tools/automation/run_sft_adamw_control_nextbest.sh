#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL_TAG="${1:-d24_asp48_track}"
BASE_MODEL_STEP="${2:-820230}"
NUM_ITERATIONS="${3:-300}"
MODEL_SOURCE="${MODEL_SOURCE:-base}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
RUN_BASE_PREFIX="${RUN_BASE_PREFIX:-d24_r32_adamw_control}"
EVAL_EVERY="${EVAL_EVERY:-25}"
KEEP_BEST_K="${KEEP_BEST_K:-1}"
FREEZE_LAYERS="${FREEZE_LAYERS:-18}"
OPTIMIZER="${OPTIMIZER:-paged_adamw8bit}"
DATASET_PRESET="${DATASET_PRESET:-general_chat_reasoning}"
MANUAL_REASONING_JSONL="${MANUAL_REASONING_JSONL:-}"
EMBEDDING_LR="${EMBEDDING_LR:-0.3}"
UNEMBEDDING_LR="${UNEMBEDDING_LR:-0.004}"
MATRIX_LR="${MATRIX_LR:-0.02}"
INIT_LR_FRAC="${INIT_LR_FRAC:-0.25}"
WARMUP_RATIO="${WARMUP_RATIO:-0.2}"
WARMDOWN_RATIO="${WARMDOWN_RATIO:-0.3}"
VAL_BPB_GATE_MAX="${VAL_BPB_GATE_MAX:-1.20}"
QUICK_GATE_MMLU_MIN="${QUICK_GATE_MMLU_MIN:-27.0}"
QUICK_GATE_REQUIRE_PASS1_NONZERO="${QUICK_GATE_REQUIRE_PASS1_NONZERO:-0}"
FULL_CONFIRM_MAX_PROBLEMS="${FULL_CONFIRM_MAX_PROBLEMS:-1000}"
SEED="${SEED:-42}"
EVAL_SEED="${EVAL_SEED:-42}"
DETERMINISTIC="${DETERMINISTIC:-0}"

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"

mkdir -p "${NOTES_DIR}"
cd "${REPO_DIR}"
source .venv/bin/activate

TS="$(date +%F_%H%M)"
RUN_BASE="${RUN_BASE_PREFIX}_${TS}"
OUT_BASE="${RUN_BASE}"
MASTER_LOG="${NOTES_DIR}/sft_${RUN_BASE}_chain.log"
META_FILE="${NOTES_DIR}/sft_${RUN_BASE}_meta.env"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-nanochat-sft}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-nanochat-eval}"
export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:128}"
export CAT_BATCH_SIZE="${CAT_BATCH_SIZE:-1}"
export EVAL_MAX_PROBLEMS="${EVAL_MAX_PROBLEMS:-250}"
export RUN_PASS1="${RUN_PASS1:-1}"
export SKIP_PASS1="${SKIP_PASS1:-0}"
export SKIP_PASS8="${SKIP_PASS8:-1}"
export SKIP_MMLU="${SKIP_MMLU:-0}"
export SKIP_SPELLING="${SKIP_SPELLING:-1}"

ATTEMPT_ORDER="${ATTEMPT_ORDER:-s1024_gc:1024:8192,s768_gc:768:7680,s640_gc:640:7680,s512_gc:512:8192,s384_gc:384:7680}"
OOM_RE='out of memory|cuda out of memory|CUDNN_STATUS_ALLOC_FAILED|CUDA error: out of memory|CUDA driver error: out of memory'

{
  echo "RUN_BASE=${RUN_BASE}"
  echo "BASE_MODEL_TAG=${BASE_MODEL_TAG}"
  echo "BASE_MODEL_STEP=${BASE_MODEL_STEP}"
  echo "MODEL_SOURCE=${MODEL_SOURCE}"
  echo "NUM_ITERATIONS=${NUM_ITERATIONS}"
  echo "WEIGHT_DECAY=${WEIGHT_DECAY}"
  echo "RUN_BASE_PREFIX=${RUN_BASE_PREFIX}"
  echo "EVAL_EVERY=${EVAL_EVERY}"
  echo "KEEP_BEST_K=${KEEP_BEST_K}"
  echo "FREEZE_LAYERS=${FREEZE_LAYERS}"
  echo "OPTIMIZER=${OPTIMIZER}"
  echo "DATASET_PRESET=${DATASET_PRESET}"
  echo "MANUAL_REASONING_JSONL=${MANUAL_REASONING_JSONL}"
  echo "EMBEDDING_LR=${EMBEDDING_LR}"
  echo "UNEMBEDDING_LR=${UNEMBEDDING_LR}"
  echo "MATRIX_LR=${MATRIX_LR}"
  echo "INIT_LR_FRAC=${INIT_LR_FRAC}"
  echo "WARMUP_RATIO=${WARMUP_RATIO}"
  echo "WARMDOWN_RATIO=${WARMDOWN_RATIO}"
  echo "VAL_BPB_GATE_MAX=${VAL_BPB_GATE_MAX}"
  echo "QUICK_GATE_MMLU_MIN=${QUICK_GATE_MMLU_MIN}"
  echo "QUICK_GATE_REQUIRE_PASS1_NONZERO=${QUICK_GATE_REQUIRE_PASS1_NONZERO}"
  echo "FULL_CONFIRM_MAX_PROBLEMS=${FULL_CONFIRM_MAX_PROBLEMS}"
  echo "SEED=${SEED}"
  echo "EVAL_SEED=${EVAL_SEED}"
  echo "DETERMINISTIC=${DETERMINISTIC}"
  echo "MASTER_LOG=${MASTER_LOG}"
  echo "WANDB_PROJECT=${WANDB_PROJECT}"
  echo "EVAL_WANDB_PROJECT=${EVAL_WANDB_PROJECT}"
  echo "CAT_BATCH_SIZE=${CAT_BATCH_SIZE}"
  echo "EVAL_MAX_PROBLEMS=${EVAL_MAX_PROBLEMS}"
  echo "RUN_PASS1=${RUN_PASS1}"
  echo "SKIP_PASS1=${SKIP_PASS1}"
  echo "ATTEMPT_ORDER=${ATTEMPT_ORDER}"
} > "${META_FILE}"

echo "[info] meta: ${META_FILE}" | tee -a "${MASTER_LOG}"
echo "[info] source: ${MODEL_SOURCE} tag=${BASE_MODEL_TAG} step=${BASE_MODEL_STEP}" | tee -a "${MASTER_LOG}"
echo "[plan] partial full-tune control with ${OPTIMIZER}, freeze_layers=${FREEZE_LAYERS}, preset=${DATASET_PRESET}, OOM ladder: ${ATTEMPT_ORDER}" | tee -a "${MASTER_LOG}"
if [[ -n "${MANUAL_REASONING_JSONL}" ]]; then
  echo "[info] manual_reasoning_jsonl=${MANUAL_REASONING_JSONL}" | tee -a "${MASTER_LOG}"
fi

resolve_eval_checkpoint() {
  local output_tag="$1"
  python3 - "${output_tag}" <<'PY'
import json
import math
import sys
from pathlib import Path

output_tag = sys.argv[1]
root = Path.home() / ".cache" / "nanochat" / "chatsft_checkpoints" / output_tag

def best_record(path):
    records = []
    if not path.exists():
        return None
    for meta_path in path.glob("meta_*.json"):
        try:
            data = json.loads(meta_path.read_text())
        except Exception:
            continue
        step = data.get("step")
        if step is None:
            try:
                step = int(meta_path.stem.split("_")[1])
            except Exception:
                continue
        try:
            val_bpb = float(data.get("val_bpb", math.inf))
        except Exception:
            val_bpb = math.inf
        records.append((val_bpb, int(step)))
    if not records:
        return None
    return min(records)

best_dir = root / "best"
record = best_record(best_dir)
if record is not None:
    val_bpb, step = record
    print(f"best {step} {val_bpb}")
    raise SystemExit(0)

record = best_record(root)
if record is not None:
    val_bpb, step = record
    print(f"main {step} {val_bpb}")
    raise SystemExit(0)

raise SystemExit(1)
PY
}

read_best_val_bpb() {
  local output_tag="$1"
  python3 - "${output_tag}" <<'PY'
import json
import math
import sys
from pathlib import Path

output_tag = sys.argv[1]
best_dir = Path.home() / ".cache" / "nanochat" / "chatsft_checkpoints" / output_tag / "best"
records = []
if best_dir.exists():
    for meta_path in best_dir.glob("meta_*.json"):
        try:
            data = json.loads(meta_path.read_text())
            val_bpb = float(data.get("val_bpb", math.inf))
        except Exception:
            continue
        records.append(val_bpb)
if not records:
    raise SystemExit(1)
print(min(records))
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

text = Path(sys.argv[1]).read_text()

def extract(label):
    m = re.search(rf"{re.escape(label)}:\s+\d+/\d+\s+=\s+([0-9.]+)%", text)
    return float(m.group(1)) if m else math.nan

print(extract("GSM8K pass@1"), extract("MMLU"))
PY
}

quick_gate_passes() {
  local summary_file="$1"
  python3 - "${summary_file}" "${QUICK_GATE_MMLU_MIN}" "${QUICK_GATE_REQUIRE_PASS1_NONZERO}" <<'PY'
import math
import re
import sys
from pathlib import Path

summary_file = Path(sys.argv[1])
mmlu_min = float(sys.argv[2])
require_pass1 = sys.argv[3] == "1"
text = summary_file.read_text()

def extract(label):
    m = re.search(rf"{re.escape(label)}:\s+\d+/\d+\s+=\s+([0-9.]+)%", text)
    return float(m.group(1)) if m else math.nan

pass1 = extract("GSM8K pass@1")
mmlu = extract("MMLU")

if math.isnan(mmlu):
    raise SystemExit(1)
if mmlu < mmlu_min:
    raise SystemExit(1)
if require_pass1 and (math.isnan(pass1) or pass1 <= 0.0):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

LAST_SUMMARY_PATH=""

run_confirm_eval() {
  local profile="$1"
  local label="$2"
  local output_tag="$3"
  local run_name="$4"
  local max_problems="$5"
  local run_pass1="$6"
  local skip_pass1="$7"
  local skip_pass8="$8"
  local skip_mmlu="$9"
  local skip_spelling="${10}"
  local eval_log="${NOTES_DIR}/eval_${run_name}_${profile}.log"
  local resolved
  if ! resolved="$(resolve_eval_checkpoint "${output_tag}")"; then
    echo "[error] unable to resolve checkpoint for eval: ${output_tag}" | tee -a "${MASTER_LOG}"
    return 1
  fi

  local checkpoint_kind best_step best_val model_group
  read -r checkpoint_kind best_step best_val <<< "${resolved}"
  if [[ "${checkpoint_kind}" == "best" ]]; then
    model_group="${output_tag}/best"
  else
    model_group="${output_tag}"
  fi

  {
    echo "[eval] profile=${profile} label=${label} model_group=${model_group} step=${best_step} val_bpb=${best_val} max_problems=${max_problems} cat_batch=${CAT_BATCH_SIZE}"
    echo "[eval_log] ${eval_log}"
  } | tee -a "${MASTER_LOG}"

  set +e
  EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT}" \
  WANDB_ENTITY="${WANDB_ENTITY}" \
  EVAL_SEED="${EVAL_SEED}" \
  DETERMINISTIC="${DETERMINISTIC}" \
  EVAL_WANDB_RUN_PREFIX="${run_name}_${profile}_confirm_s${best_step}" \
  CAT_BATCH_SIZE="${CAT_BATCH_SIZE}" \
  RUN_PASS1="${run_pass1}" \
  SKIP_PASS1="${skip_pass1}" \
  SKIP_PASS8="${skip_pass8}" \
  SKIP_MMLU="${skip_mmlu}" \
  SKIP_SPELLING="${skip_spelling}" \
  "${HOME}/nanochat-learn/scripts/run_chat_eval_confirm_1k.sh" "${model_group}" "${best_step}" "${max_problems}" \
    |& tee "${eval_log}" | tee -a "${MASTER_LOG}"
  local rc=${PIPESTATUS[0]}
  set -e

  if [[ "${rc}" -ne 0 ]]; then
    echo "[error] confirm eval failed for ${run_name} (rc=${rc})" | tee -a "${MASTER_LOG}"
    return "${rc}"
  fi

  LAST_SUMMARY_PATH="$(extract_summary_path "${eval_log}")"
  if [[ -z "${LAST_SUMMARY_PATH}" ]]; then
    echo "[error] could not locate summary path in ${eval_log}" | tee -a "${MASTER_LOG}"
    return 1
  fi

  echo "[summary] ${LAST_SUMMARY_PATH}" | tee -a "${MASTER_LOG}"
  echo "[done] confirm eval finished for ${run_name}" | tee -a "${MASTER_LOG}"
  return 0
}

run_attempt() {
  local label="$1"
  local max_seq_len="$2"
  local total_batch_size="$3"

  local output_tag="${OUT_BASE}_${label}"
  local run_name="${RUN_BASE}_${label}"
  local log_file="${NOTES_DIR}/sft_${run_name}.log"

  {
    echo "[start] run=${run_name} output=${output_tag} max_seq_len=${max_seq_len} total_batch_size=${total_batch_size}"
    echo "[log] ${log_file}"
  } | tee -a "${MASTER_LOG}"

  local tokens_per_fwdbwd=$((max_seq_len))
  if (( total_batch_size % tokens_per_fwdbwd != 0 )); then
    echo "[error] invalid batch geometry for ${run_name}: total_batch_size=${total_batch_size} is not divisible by max_seq_len=${max_seq_len}" | tee -a "${MASTER_LOG}"
    return 1
  fi

  local cmd=(
    .venv/bin/python -u -m scripts.chat_sft
    --model-source "${MODEL_SOURCE}"
    --model-tag "${BASE_MODEL_TAG}" --model-step "${BASE_MODEL_STEP}"
    --output-tag "${output_tag}" --run "${run_name}"
    --device-type cuda --dtype bfloat16
    --seed "${SEED}"
    --adamw-only --optimizer "${OPTIMIZER}" --weight-decay "${WEIGHT_DECAY}"
    --embedding-lr "${EMBEDDING_LR}" --unembedding-lr "${UNEMBEDDING_LR}" --matrix-lr "${MATRIX_LR}"
    --init-lr-frac "${INIT_LR_FRAC}" --warmup-ratio "${WARMUP_RATIO}" --warmdown-ratio "${WARMDOWN_RATIO}"
    --max-seq-len "${max_seq_len}" --device-batch-size 1 --total-batch-size "${total_batch_size}"
    --num-iterations "${NUM_ITERATIONS}"
    --eval-every "${EVAL_EVERY}" --eval-tokens 524288
    --dataset-preset "${DATASET_PRESET}"
    --freeze-layers "${FREEZE_LAYERS}" --freeze-embeddings --freeze-scalars
    --gradient-checkpoint --max-grad-norm 1.0
    --keep-best-k "${KEEP_BEST_K}" --no-save-optimizer
  )
  if [[ -n "${MANUAL_REASONING_JSONL}" ]]; then
    cmd+=(--manual-reasoning-jsonl "${MANUAL_REASONING_JSONL}")
  fi
  if [[ "${DETERMINISTIC}" == "1" ]]; then
    cmd+=(--deterministic)
  fi

  set +e
  "${cmd[@]}" |& tee "${log_file}" | tee -a "${MASTER_LOG}"
  local rc=${PIPESTATUS[0]}
  set -e

  echo "[end] run=${run_name} rc=${rc}" | tee -a "${MASTER_LOG}"

  if [[ "${rc}" -eq 0 ]]; then
    local best_val_bpb
    if best_val_bpb="$(read_best_val_bpb "${output_tag}")"; then
      echo "[result] best_val_bpb=${best_val_bpb} for ${run_name}" | tee -a "${MASTER_LOG}"
      if ! python3 - "${best_val_bpb}" "${VAL_BPB_GATE_MAX}" <<'PY'
import sys
best = float(sys.argv[1])
gate = float(sys.argv[2])
raise SystemExit(0 if best <= gate else 1)
PY
      then
        echo "[gate] skipping external eval for ${run_name}: best_val_bpb=${best_val_bpb} exceeds gate ${VAL_BPB_GATE_MAX}" | tee -a "${MASTER_LOG}"
        return 1
      fi
    fi
    if ! run_confirm_eval quick "${label}" "${output_tag}" "${run_name}" "${EVAL_MAX_PROBLEMS}" "${RUN_PASS1}" "${SKIP_PASS1}" "${SKIP_PASS8}" "0" "${SKIP_SPELLING}"; then
      return 1
    fi

    local pass1_pct mmlu_pct
    read -r pass1_pct mmlu_pct <<< "$(read_eval_gate_metrics "${LAST_SUMMARY_PATH}")"
    echo "[gate] quick metrics for ${run_name}: gsm8k_pass1=${pass1_pct}% mmlu=${mmlu_pct}% thresholds: mmlu>=${QUICK_GATE_MMLU_MIN}% pass1_nonzero_required=${QUICK_GATE_REQUIRE_PASS1_NONZERO}" | tee -a "${MASTER_LOG}"

    if ! quick_gate_passes "${LAST_SUMMARY_PATH}"; then
      echo "[gate] quick gate failed for ${run_name}; skipping full confirm" | tee -a "${MASTER_LOG}"
      return 1
    fi

    echo "[gate] quick gate passed for ${run_name}; launching full confirm" | tee -a "${MASTER_LOG}"
    if run_confirm_eval full "${label}" "${output_tag}" "${run_name}" "${FULL_CONFIRM_MAX_PROBLEMS}" "0" "1" "0" "0" "0"; then
      return 0
    fi
    return 1
  fi

  if rg -qi "${OOM_RE}" "${log_file}"; then
    echo "[warn] OOM detected in ${run_name}" | tee -a "${MASTER_LOG}"
    return 2
  fi

  echo "[error] non-OOM failure in ${run_name}; check ${log_file}" | tee -a "${MASTER_LOG}"
  return 1
}

IFS=',' read -r -a attempts <<< "${ATTEMPT_ORDER}"
for spec in "${attempts[@]}"; do
  IFS=':' read -r label max_seq_len total_batch_size <<< "${spec}"
  if run_attempt "${label}" "${max_seq_len}" "${total_batch_size}"; then
    echo "[done] completed at ${label}" | tee -a "${MASTER_LOG}"
    exit 0
  fi
  rc=$?
  if [[ "${rc}" -eq 1 ]]; then
    exit 1
  fi
done

echo "[fail] all partial full-tune attempts exhausted without a promotable result: ${ATTEMPT_ORDER}" | tee -a "${MASTER_LOG}"
exit 2
