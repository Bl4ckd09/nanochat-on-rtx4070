#!/usr/bin/env bash
# Monitor SFT training progress - extracts key metrics from the log file
# Usage:
#   bash ~/nanochat-learn/scripts/monitor_sft.sh
#   bash ~/nanochat-learn/scripts/monitor_sft.sh 30
#   bash ~/nanochat-learn/scripts/monitor_sft.sh /path/to/log.txt 30
set -euo pipefail

DEFAULT_LOG="$(ls -t ~/nanochat-learn/notes/sft-d18_20k_*.txt 2>/dev/null | head -1)"
LOG="$DEFAULT_LOG"
INTERVAL=60

if [ "${1:-}" != "" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        INTERVAL="$1"
    else
        LOG="$1"
    fi
fi

if [ "${2:-}" != "" ]; then
    INTERVAL="$2"
fi

if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
    echo "No SFT log file found."
    exit 1
fi

echo "Monitoring: $LOG (every ${INTERVAL}s)"
echo "Press Ctrl+C to stop."
echo ""

while true; do
    if [ -t 1 ]; then
        clear
    fi
    echo "=== SFT Monitor ($(date +%H:%M:%S)) ==="
    echo "Log: $LOG"
    echo ""

    # Latest step
    LAST_LINE=$(grep -E "^step " "$LOG" | tail -1)
    echo "Latest: $LAST_LINE"
    echo ""

    # All validation checkpoints
    echo "--- Validation History ---"
    grep -E "Validation bpb:" "$LOG" || echo "(no evals yet)"
    echo ""

    # Best checkpoint info
    echo "--- Best Checkpoint ---"
    grep -E "Saved best|Rotated old" "$LOG" | tail -2
    echo ""

    # Training loss trend (last 10 steps)
    echo "--- Loss Trend (last 10 steps) ---"
    grep -E "^step " "$LOG" | tail -10 | awk -F'|' '{
        split($1, a, " ");
        split($2, b, ": ");
        printf "%s loss=%s\n", a[2], b[2]
    }'
    echo ""

    # Quick ETA estimate from progress percentage
    LAST_PCT=$(echo "$LAST_LINE" | sed -n 's/.*(\([0-9.]\+\)%).*/\1/p')
    LAST_MIN=$(echo "$LAST_LINE" | sed -n 's/.*total time: \([0-9.]\+\)m.*/\1/p')
    if [ -n "$LAST_PCT" ] && [ -n "$LAST_MIN" ] && [ "$LAST_PCT" != "0.00" ]; then
        ETA_MIN=$(awk -v pct="$LAST_PCT" -v elapsed="$LAST_MIN" 'BEGIN {
            rem = elapsed * (100 - pct) / pct;
            if (rem < 0) rem = 0;
            printf "%.1f", rem
        }')
        echo "--- ETA (rough) ---"
        echo "Remaining: ${ETA_MIN}m"
    fi

    # Check if chain has moved past SFT
    CHAIN_LOG=~/nanochat-learn/notes/chain_log.txt
    if grep -q "Step 4: SFT done" "$CHAIN_LOG" 2>/dev/null; then
        echo ""
        echo ">>> SFT COMPLETED! Chain moving to evals <<<"
        grep -E "Step [45]:" "$CHAIN_LOG" | tail -5
    fi

    # Check if chain is fully done
    if grep -q "All steps complete" "$CHAIN_LOG" 2>/dev/null; then
        echo ""
        echo ">>> ALL CHAIN STEPS COMPLETE <<<"
        break
    fi

    # Check if process is still alive
    if [ -f /tmp/chain_steps.pid ]; then
        CHAIN_PID="$(cat /tmp/chain_steps.pid 2>/dev/null || true)"
        if [ -n "$CHAIN_PID" ] && ! ps -p "$CHAIN_PID" > /dev/null 2>&1; then
            echo ""
            echo ">>> WARNING: Chain process appears to have exited <<<"
            echo "Check chain_log.txt for errors."
            break
        fi
    fi

    sleep "$INTERVAL"
done
