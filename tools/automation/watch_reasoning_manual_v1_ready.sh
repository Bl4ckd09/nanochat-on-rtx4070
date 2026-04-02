#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${HOME}/nanochat-learn/nanochat"
NOTES_DIR="${HOME}/nanochat-learn/notes"
MANUAL_REASONING_JSONL="${MANUAL_REASONING_JSONL:-${REPO_DIR}/data/manual_reasoning_chat_v1.jsonl}"
MIN_MANUAL_ROWS="${MIN_MANUAL_ROWS:-100}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TS="$(date +%F_%H%M)"
CHAIN_NAME="reasoning_manual_v1_ready_${TS}"
MASTER_LOG="${NOTES_DIR}/${CHAIN_NAME}.log"
DECISION_MD="${NOTES_DIR}/${CHAIN_NAME}_decision.md"
RUNNER="${REPO_DIR}/tools/automation/run_reasoning_manual_v1.sh"

mkdir -p "${NOTES_DIR}"
cd "${REPO_DIR}"
source .venv/bin/activate

log() {
  echo "$1" | tee -a "${MASTER_LOG}"
}

count_rows() {
  python3 - "${MANUAL_REASONING_JSONL}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
if not path.exists():
    print(0)
    raise SystemExit(0)
rows = 0
with path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows += 1
print(rows)
PY
}

latest_decision_file() {
  ls -1t "${NOTES_DIR}"/reasoning_manual_v1_*_decision.md 2>/dev/null | head -n 1 || true
}

write_decision() {
  local state="$1"
  local runner_decision="$2"
  cat > "${DECISION_MD}" <<EOF
# ${CHAIN_NAME} decision

- state: \`${state}\`
- manual_reasoning_jsonl: \`${MANUAL_REASONING_JSONL}\`
- min_manual_rows: \`${MIN_MANUAL_ROWS}\`
- runner_decision: \`${runner_decision}\`

If the manual branch does not promote, hold \`mixv2_s768_300\` as the provisional champion and treat the next productive move as a separate specialist branch or stronger hardware, not another same-family mix sweep.
EOF
}

log "[start] waiting for curated manual dataset readiness"
log "[info] manual_reasoning_jsonl=${MANUAL_REASONING_JSONL}"
log "[info] min_manual_rows=${MIN_MANUAL_ROWS} poll_seconds=${POLL_SECONDS}"
log "[info] runner=${RUNNER}"

while true; do
  rows="$(count_rows)"
  log "[state] waiting rows=${rows}/${MIN_MANUAL_ROWS}"
  if (( rows >= MIN_MANUAL_ROWS )); then
    break
  fi
  sleep "${POLL_SECONDS}"
done

log "[launch] curated dataset ready; launching reasoning_manual_v1"

set +e
bash "${RUNNER}" |& tee -a "${MASTER_LOG}"
rc=${PIPESTATUS[0]}
set -e

decision_file="$(latest_decision_file)"
runner_decision="missing"
if [[ -n "${decision_file}" && -f "${decision_file}" ]]; then
  runner_decision="$(rg '^- decision: ' "${decision_file}" | head -n 1 | sed 's/^- decision: `//; s/`$//')"
fi

if [[ "${rc}" -eq 0 && "${runner_decision}" == "promote_recipe" ]]; then
  write_decision "promoted" "${runner_decision}"
  log "[done] manual reasoning branch promoted"
  exit 0
fi

write_decision "hold_mixv2_provisional_and_stop" "${runner_decision}"
log "[done] manual reasoning branch did not promote; holding current provisional champion"
exit 0
