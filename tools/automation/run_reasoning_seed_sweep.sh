#!/usr/bin/env bash
set -euo pipefail

RUN_LABEL="${1:-reasoning_seed_sweep}"
DATASET_PRESET="${2:-reasoning_focus_v2}"
BASE_MODEL_TAG="${3:-d24_asp48_track}"
BASE_MODEL_STEP="${4:-820230}"
NUM_ITERATIONS="${5:-300}"
SEEDS_CSV="${SEEDS:-42,43,44}"
FREEZE_LAYERS="${FREEZE_LAYERS:-20}"
ATTEMPT_ORDER="${ATTEMPT_ORDER:-s768_gc:768:7680}"
INIT_LR_FRAC="${INIT_LR_FRAC:-0.05}"
QUICK_GATE_MMLU_MIN="${QUICK_GATE_MMLU_MIN:-27.0}"
QUICK_GATE_REQUIRE_PASS1_NONZERO="${QUICK_GATE_REQUIRE_PASS1_NONZERO:-0}"
EVAL_MAX_PROBLEMS="${EVAL_MAX_PROBLEMS:-250}"
FULL_CONFIRM_MAX_PROBLEMS="${FULL_CONFIRM_MAX_PROBLEMS:-1000}"
DETERMINISTIC="${DETERMINISTIC:-0}"
EVAL_SEED="${EVAL_SEED:-42}"
PROMOTE_PASS_COUNT_MIN="${PROMOTE_PASS_COUNT_MIN:-2}"
PROMOTE_GSM8K_MIN="${PROMOTE_GSM8K_MIN:-4.0}"
PROMOTE_MMLU_MIN="${PROMOTE_MMLU_MIN:-27.0}"

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
RUN_SCRIPT="${REPO_DIR}/tools/automation/run_sft_adamw_control_nextbest.sh"
TS="$(date +%F_%H%M)"
SWEEP_NAME="${RUN_LABEL}_${TS}"
MASTER_LOG="${NOTES_DIR}/${SWEEP_NAME}.log"
RESULTS_TSV="${NOTES_DIR}/${SWEEP_NAME}_results.tsv"
DECISION_MD="${NOTES_DIR}/${SWEEP_NAME}_decision.md"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-}"

mkdir -p "${NOTES_DIR}"
cd "${REPO_DIR}"
source .venv/bin/activate

log() {
  echo "$1" | tee -a "${MASTER_LOG}"
}

extract_last_summary() {
  local chain_log="$1"
  rg '^\[summary\]\s+' "${chain_log}" 2>/dev/null | tail -n 1 | sed 's/^\[summary\]\s\+//' || true
}

read_metrics() {
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

vals = [
    extract("GSM8K pass@1"),
    extract("GSM8K pass@8"),
    extract("MMLU"),
    extract("SpellingBee"),
]
print(*("nan" if math.isnan(v) else f"{v:.2f}" for v in vals))
PY
}

write_decision() {
  python3 - "${RESULTS_TSV}" "${DECISION_MD}" "${BASELINE_SUMMARY}" "${PROMOTE_PASS_COUNT_MIN}" "${PROMOTE_GSM8K_MIN}" "${PROMOTE_MMLU_MIN}" <<'PY'
import csv
import math
import re
import sys
from pathlib import Path

results_path = Path(sys.argv[1])
decision_path = Path(sys.argv[2])
baseline_path = Path(sys.argv[3]) if sys.argv[3] else None
promote_min = int(sys.argv[4])
promote_gsm_min = float(sys.argv[5])
promote_mmlu_min = float(sys.argv[6])
rows = list(csv.DictReader(results_path.open(), delimiter='\t'))
baseline = baseline_path.read_text() if baseline_path and baseline_path.exists() else ""

def metric(text, label):
    m = re.search(rf"{re.escape(label)}:\s+\d+/\d+\s+=\s+([0-9.]+)%", text)
    return float(m.group(1)) if m else math.nan

baseline_gsm = metric(baseline, "GSM8K pass@8")
baseline_mmlu = metric(baseline, "MMLU")
baseline_spell = metric(baseline, "SpellingBee")
full_rows = [r for r in rows if r.get("status") == "full_confirm_complete"]
threshold_rows = []
for r in full_rows:
    try:
        gsm = float(r["gsm8k_pass8_pct"])
        mmlu = float(r["mmlu_pct"])
    except Exception:
        continue
    if gsm >= promote_gsm_min and mmlu >= promote_mmlu_min:
        threshold_rows.append(r)

if len(threshold_rows) >= promote_min:
    decision = "promote_recipe"
    reason = f"{len(threshold_rows)} seeds met the replication thresholds (required {promote_min})"
elif len(full_rows) >= 1:
    decision = "hold_provisional"
    reason = "some seeds reached full confirm, but not enough met the promotion thresholds"
else:
    decision = "recipe_failed_quick_gate"
    reason = "no seed produced a full-confirm result that met the thresholds"

lines = [
    f"# {results_path.stem} decision",
    "",
    f"- decision: `{decision}`",
    f"- reason: {reason}",
    f"- seeds_total: `{len(rows)}`",
    f"- full_confirm_count: `{len(full_rows)}`",
    f"- threshold_pass_count: `{len(threshold_rows)}`",
    f"- threshold_required: `{promote_min}`",
    f"- promote_gsm8k_pass8_min: `{promote_gsm_min:.2f}%`",
    f"- promote_mmlu_min: `{promote_mmlu_min:.2f}%`",
]
if not math.isnan(baseline_gsm):
    lines.append(f"- baseline_gsm8k_pass8: `{baseline_gsm:.2f}%`")
if not math.isnan(baseline_mmlu):
    lines.append(f"- baseline_mmlu: `{baseline_mmlu:.2f}%`")
if not math.isnan(baseline_spell):
    lines.append(f"- baseline_spellingbee: `{baseline_spell:.2f}%`")
lines.append("")
lines.append("## Seed Results")
lines.append("")
for r in rows:
    lines.append(f"- seed {r['seed']}: status=`{r['status']}` gsm8k_pass8=`{r['gsm8k_pass8_pct']}` mmlu=`{r['mmlu_pct']}` summary=`{r['summary']}`")

decision_path.write_text("\n".join(lines) + "\n")
print(decision_path)
PY
}

printf 'seed\trc\tstatus\trun_base\tchain_log\tsummary\tgsm8k_pass1_pct\tgsm8k_pass8_pct\tmmlu_pct\tspellingbee_pct\n' > "${RESULTS_TSV}"

log "[info] sweep=${SWEEP_NAME}"
log "[info] dataset_preset=${DATASET_PRESET} seeds=${SEEDS_CSV}"
log "[info] deterministic=${DETERMINISTIC} eval_seed=${EVAL_SEED}"
log "[info] quick_gate: max_problems=${EVAL_MAX_PROBLEMS} mmlu>=${QUICK_GATE_MMLU_MIN}% pass1_nonzero_required=${QUICK_GATE_REQUIRE_PASS1_NONZERO}"
log "[info] promote_band: gsm8k_pass8>=${PROMOTE_GSM8K_MIN}% mmlu>=${PROMOTE_MMLU_MIN}%"
log "[info] baseline=${BASELINE_SUMMARY:-none}"
log "[info] results=${RESULTS_TSV}"

IFS=',' read -r -a seeds <<< "${SEEDS_CSV}"
for seed in "${seeds[@]}"; do
  prefix="${RUN_LABEL}_seed${seed}"
  log "[start] seed=${seed} prefix=${prefix}"
  set +e
  RUN_BASE_PREFIX="${prefix}" \
  FREEZE_LAYERS="${FREEZE_LAYERS}" \
  DATASET_PRESET="${DATASET_PRESET}" \
  INIT_LR_FRAC="${INIT_LR_FRAC}" \
  QUICK_GATE_MMLU_MIN="${QUICK_GATE_MMLU_MIN}" \
  QUICK_GATE_REQUIRE_PASS1_NONZERO="${QUICK_GATE_REQUIRE_PASS1_NONZERO}" \
  EVAL_MAX_PROBLEMS="${EVAL_MAX_PROBLEMS}" \
  FULL_CONFIRM_MAX_PROBLEMS="${FULL_CONFIRM_MAX_PROBLEMS}" \
  ATTEMPT_ORDER="${ATTEMPT_ORDER}" \
  SEED="${seed}" \
  EVAL_SEED="${EVAL_SEED}" \
  DETERMINISTIC="${DETERMINISTIC}" \
  WANDB_MODE="${WANDB_MODE:-online}" \
  WANDB_PROJECT="${WANDB_PROJECT:-nanochat-sft}" \
  EVAL_WANDB_PROJECT="${EVAL_WANDB_PROJECT:-nanochat-eval}" \
  bash "${RUN_SCRIPT}" "${BASE_MODEL_TAG}" "${BASE_MODEL_STEP}" "${NUM_ITERATIONS}" |& tee -a "${MASTER_LOG}"
  rc=${PIPESTATUS[0]}
  set -e

  chain_log="$(ls -1t "${NOTES_DIR}/sft_${prefix}"_*_chain.log 2>/dev/null | head -n 1 || true)"
  summary=""
  status="no_summary"
  pass1="nan"
  pass8="nan"
  mmlu="nan"
  spelling="nan"

  if [[ -n "${chain_log}" && -f "${chain_log}" ]]; then
    summary="$(extract_last_summary "${chain_log}")"
    if rg -q '^\[done\] completed at ' "${chain_log}"; then
      status="full_confirm_complete"
    elif rg -q '^\[gate\] quick gate failed ' "${chain_log}"; then
      status="quick_gate_failed"
    elif rg -q '^\[error\] non-OOM failure ' "${chain_log}"; then
      status="failed_non_oom"
    elif rg -q '^\[fail\] all partial full-tune attempts exhausted ' "${chain_log}"; then
      status="failed_attempts_exhausted"
    fi
  fi

  if [[ -n "${summary}" && -f "${summary}" ]]; then
    read -r pass1 pass8 mmlu spelling <<< "$(read_metrics "${summary}")"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${seed}" "${rc}" "${status}" "${prefix}" "${chain_log}" "${summary}" "${pass1}" "${pass8}" "${mmlu}" "${spelling}" \
    >> "${RESULTS_TSV}"
  log "[end] seed=${seed} rc=${rc} status=${status} pass8=${pass8} mmlu=${mmlu}"
done

DECISION_PATH="$(write_decision)"
log "[decision] ${DECISION_PATH}"
log "[done] sweep complete"
