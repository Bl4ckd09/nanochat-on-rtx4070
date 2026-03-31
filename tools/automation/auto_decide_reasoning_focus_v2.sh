#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <run_name> <chain_log>"
  exit 2
fi

RUN_NAME="$1"
CHAIN_LOG="$2"
NOTES_DIR="${HOME}/nanochat-learn/notes"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-${NOTES_DIR}/d24_r32_adamw_partial_fr20_mixv1_2026-03-29_0518_s1024_gc_best_s300_confirm_summary_1000_2026-03-29_0607.txt}"
DECISION_FILE="${NOTES_DIR}/${RUN_NAME}_decision.md"

if [[ ! -f "${CHAIN_LOG}" ]]; then
  alt_chain="${NOTES_DIR}/sft_${RUN_NAME}_chain.log"
  if [[ -f "${alt_chain}" ]]; then
    CHAIN_LOG="${alt_chain}"
  fi
fi

wait_for_terminal_state() {
  while true; do
    if [[ -f "${CHAIN_LOG}" ]]; then
      if rg -q '^\[done\] completed at ' "${CHAIN_LOG}"; then
        return 0
      fi
      if rg -q '^\[gate\] quick gate failed ' "${CHAIN_LOG}"; then
        return 0
      fi
      if rg -q '^\[error\] non-OOM failure ' "${CHAIN_LOG}"; then
        return 0
      fi
      if rg -q '^\[fail\] all partial full-tune attempts exhausted ' "${CHAIN_LOG}"; then
        return 0
      fi
    fi
    sleep 30
  done
}

write_decision() {
  python3 - "${RUN_NAME}" "${CHAIN_LOG}" "${BASELINE_SUMMARY}" "${DECISION_FILE}" <<'PY'
import math
import re
import sys
from pathlib import Path

run_name = sys.argv[1]
chain_path = Path(sys.argv[2])
baseline_path = Path(sys.argv[3])
decision_path = Path(sys.argv[4])
chain = chain_path.read_text() if chain_path.exists() else ""
baseline = baseline_path.read_text() if baseline_path.exists() else ""

def extract_summary(chain_text):
    m = re.findall(r'^\[summary\]\s+(.+)$', chain_text, flags=re.M)
    return Path(m[-1]) if m else None

def extract_metric(text, label):
    m = re.search(rf"{re.escape(label)}:\s+\d+/\d+\s+=\s+([0-9.]+)%", text)
    return float(m.group(1)) if m else math.nan

summary_path = extract_summary(chain)
summary_text = summary_path.read_text() if summary_path and summary_path.exists() else ""

baseline_gsm = extract_metric(baseline, "GSM8K pass@8")
baseline_mmlu = extract_metric(baseline, "MMLU")
baseline_spell = extract_metric(baseline, "SpellingBee")
run_gsm = extract_metric(summary_text, "GSM8K pass@8")
run_mmlu = extract_metric(summary_text, "MMLU")
run_spell = extract_metric(summary_text, "SpellingBee")

status = "unknown"
decision = "hold_mixv1"
reason = "run ended without a recognized terminal state"

if "[error] non-OOM failure" in chain:
    status = "failed_non_oom"
    decision = "retry_after_fix"
    reason = "training/eval ended with a non-OOM failure"
elif "[fail] all partial full-tune attempts exhausted" in chain:
    status = "failed_attempts_exhausted"
    decision = "hold_mixv1"
    reason = "the launcher exhausted its attempt ladder without producing a promotable result"
elif "[gate] quick gate failed" in chain:
    status = "quick_gate_failed"
    decision = "hold_mixv1"
    reason = "the quick gate did not justify a full confirm"
elif "[done] completed at " in chain and summary_text:
    status = "full_confirm_complete"
    if not math.isnan(run_gsm) and not math.isnan(run_mmlu) and run_gsm >= 3.0 and run_mmlu >= 27.0:
        decision = "promote_reasoning_focus_v2"
        reason = "v2 met the promotion thresholds: GSM8K pass@8 >= 3.0 and MMLU >= 27.0"
    else:
        decision = "hold_mixv1"
        reason = "v2 did not meet the promotion thresholds against the current mixv1 reference"

lines = [
    f"# {run_name} decision",
    "",
    f"- status: `{status}`",
    f"- decision: `{decision}`",
    f"- reason: {reason}",
    f"- chain_log: `{chain_path}`",
]
if summary_path:
    lines.append(f"- summary: `{summary_path}`")
if not math.isnan(run_gsm):
    lines.append(f"- run_gsm8k_pass8: `{run_gsm:.2f}%`")
if not math.isnan(run_mmlu):
    lines.append(f"- run_mmlu: `{run_mmlu:.2f}%`")
if not math.isnan(run_spell):
    lines.append(f"- run_spellingbee: `{run_spell:.2f}%`")
if not math.isnan(baseline_gsm):
    lines.append(f"- baseline_mixv1_gsm8k_pass8: `{baseline_gsm:.2f}%`")
if not math.isnan(baseline_mmlu):
    lines.append(f"- baseline_mixv1_mmlu: `{baseline_mmlu:.2f}%`")
if not math.isnan(baseline_spell):
    lines.append(f"- baseline_mixv1_spellingbee: `{baseline_spell:.2f}%`")

decision_path.write_text("\n".join(lines) + "\n")
print(decision_path)
PY
}

wait_for_terminal_state
DECISION_PATH="$(write_decision)"
echo "[decision] ${DECISION_PATH}"
