#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <r32_bs1_log_path> [start_step] [delta_steps] [poll_seconds]"
  exit 1
fi

LOG_PATH="$1"
START_STEP="${2:-615173}"
DELTA_STEPS="${3:-1000}"
POLL_SECONDS="${4:-60}"

if [[ ! -f "${LOG_PATH}" ]]; then
  echo "Log file not found: ${LOG_PATH}"
  exit 1
fi

NOTES_DIR="${HOME}/nanochat-learn/notes"
mkdir -p "${NOTES_DIR}"
TARGET_STEP=$((START_STEP + DELTA_STEPS))
WATCH_LOG="${NOTES_DIR}/d24_r32_gate_watch.log"

echo "[start] $(date '+%F %T') gate monitor log=${LOG_PATH} start_step=${START_STEP} target_step=${TARGET_STEP}" | tee -a "${WATCH_LOG}"

extract_latest_step() {
  rg -o '^step [0-9]+/[0-9]+' "${LOG_PATH}" 2>/dev/null \
    | tail -1 \
    | awk '{split($2,a,"/"); print a[1]}' || true
}

while true; do
  latest_step="$(extract_latest_step)"
  if [[ -n "${latest_step}" ]]; then
    echo "[poll] $(date '+%F %T') latest_step=${latest_step} target_step=${TARGET_STEP}" >> "${WATCH_LOG}"
    if (( latest_step >= TARGET_STEP )); then
      break
    fi
  fi
  sleep "${POLL_SECONDS}"
done

OUT_MD="${NOTES_DIR}/d24_r32_gate_decision_$(date +%F_%H%M%S).md"
WAND_B_URL="$(rg -o 'https://wandb.ai[^ ]+' "${LOG_PATH}" | tail -1 || true)"

python3 - "${LOG_PATH}" "${START_STEP}" "${TARGET_STEP}" "${OUT_MD}" "${WAND_B_URL}" <<'PY'
import re
import sys
import statistics
from datetime import datetime

log_path, start_s, target_s, out_md, wandb_url = sys.argv[1:]
start_step = int(start_s)
target_step = int(target_s)

step_re = re.compile(r"^step (\d+)/(\d+).*?loss: ([0-9]*\.?[0-9]+).*?tok/sec: ([0-9,]+)")
bad_re = re.compile(r"(out of memory|runtimeerror|nan|overflow)", re.IGNORECASE)

rows = []
bad_lines = []
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = step_re.search(line)
        if m:
            step = int(m.group(1))
            if start_step < step <= target_step:
                loss = float(m.group(3))
                tok_sec = int(m.group(4).replace(",", ""))
                rows.append((step, loss, tok_sec))
                continue
        if bad_re.search(line):
            bad_lines.append(line.strip())

rows.sort(key=lambda x: x[0])
n = len(rows)

if n == 0:
    decision = "NO_GO"
    reason = "No step records found in gate window."
    summary = {
        "window_start_step": start_step + 1,
        "window_end_step": target_step,
        "samples": 0,
    }
else:
    losses = [r[1] for r in rows]
    toksecs = [r[2] for r in rows]

    first_window = losses[: min(100, len(losses))]
    last_window = losses[-min(100, len(losses)) :]
    first_med = statistics.median(first_window)
    last_med = statistics.median(last_window)
    loss_delta = last_med - first_med
    max_loss = max(losses)
    min_loss = min(losses)
    med_tok = statistics.median(toksecs)
    min_tok = min(toksecs)
    max_tok = max(toksecs)

    # Gate heuristics for short early-stability check.
    checks = {
        "enough_samples": n >= min(300, target_step - start_step),
        "no_bad_lines": len(bad_lines) == 0,
        "loss_not_spiking": loss_delta <= 0.25,
        "toksec_healthy": med_tok >= 5500,
        "no_extreme_loss_jump": max_loss <= first_med + 1.0,
    }

    decision = "GO" if all(checks.values()) else "NO_GO"
    failed = [k for k, v in checks.items() if not v]
    reason = "All checks passed." if not failed else f"Failed checks: {', '.join(failed)}"

    summary = {
        "window_start_step": rows[0][0],
        "window_end_step": rows[-1][0],
        "samples": n,
        "first100_median_loss": round(first_med, 6),
        "last100_median_loss": round(last_med, 6),
        "loss_delta_last_minus_first": round(loss_delta, 6),
        "min_loss": round(min_loss, 6),
        "max_loss": round(max_loss, 6),
        "median_tok_per_sec": int(med_tok),
        "min_tok_per_sec": int(min_tok),
        "max_tok_per_sec": int(max_tok),
        "bad_line_count": len(bad_lines),
    }

lines = []
lines.append("# d24 r32 -> r40 Gate Decision")
lines.append("")
lines.append(f"- Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
lines.append(f"- Source log: `{log_path}`")
if wandb_url:
    lines.append(f"- W&B run: `{wandb_url}`")
lines.append(f"- Decision: **{decision}**")
lines.append(f"- Reason: {reason}")
lines.append("")
lines.append("## Metrics")
for k, v in summary.items():
    lines.append(f"- {k}: `{v}`")
lines.append("")
lines.append("## Next Command")
if decision == "GO":
    lines.append("```bash")
    lines.append("WANDB_MODE=online WANDB_PROJECT=nanochat ~/nanochat-learn/scripts/run_base_d24_asp48_maxcore_safe.sh d24_asp48_track 40")
    lines.append("```")
else:
    lines.append("```bash")
    lines.append("WANDB_MODE=online WANDB_PROJECT=nanochat EMBEDDING_LR=0.21 UNEMBEDDING_LR=0.0028 MATRIX_LR=0.014 SCALAR_LR=0.35 ~/nanochat-learn/scripts/run_base_d24_asp48_maxcore_safe.sh d24_asp48_track 32")
    lines.append("```")
lines.append("")
if bad_lines:
    lines.append("## Matched Warning/Error Lines")
    for x in bad_lines[-20:]:
        lines.append(f"- `{x}`")

with open(out_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
PY

cp -f "${OUT_MD}" "${NOTES_DIR}/d24_r32_gate_latest.md"
echo "[done] $(date '+%F %T') decision file: ${OUT_MD}" | tee -a "${WATCH_LOG}"
echo "[done] latest pointer: ${NOTES_DIR}/d24_r32_gate_latest.md" | tee -a "${WATCH_LOG}"
