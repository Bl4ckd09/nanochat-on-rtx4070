#!/usr/bin/env bash
# Chain Steps 3-5: base eval → SFT → GSM8K/benchmarks
# Usage: nohup bash ~/nanochat-learn/scripts/chain_post_training.sh > ~/nanochat-learn/notes/chain_log.txt 2>&1 &
set -euo pipefail

NANOCHAT=~/nanochat-learn/nanochat
PYTHON="$NANOCHAT/.venv/bin/python"
NOTES=~/nanochat-learn/notes
CKPT_DIR=~/.cache/nanochat/base_checkpoints/d18

echo "=== Chain script started at $(date) ==="

# Verify checkpoint exists
if [ ! -f "$CKPT_DIR/model_020000.pt" ]; then
    echo "ERROR: model_020000.pt not found in $CKPT_DIR"
    ls -la "$CKPT_DIR"/
    exit 1
fi
echo "Checkpoint model_020000.pt confirmed."

# ─────────────────────────────────────────────
# Step 3: Base eval at step 20000
# ─────────────────────────────────────────────
echo ""
echo "=== Step 3: Base eval (started $(date)) ==="
TS=$(date +%F_%H%M)
WANDB_MODE=disabled "$PYTHON" -m scripts.base_eval \
    --model-tag d18 --step 20000 \
    --eval bpb,sample \
    --split-tokens 524288 \
    --device-type cuda \
    --device-batch-size 8 \
    2>&1 | tee "$NOTES/base-d18-eval_20k_${TS}.txt"
echo "=== Step 3: Base eval done ($(date)) ==="

# ─────────────────────────────────────────────
# Step 4: SFT from step 20000 base
# ─────────────────────────────────────────────
echo ""
echo "=== Step 4: SFT (started $(date)) ==="
TS=$(date +%F_%H%M)
PYTHONUNBUFFERED=1 TORCH_COMPILE_DISABLE=1 WANDB_MODE=disabled "$PYTHON" -u -m scripts.chat_sft \
    --model-tag d18 --model-step 20000 \
    --output-tag d18_20k \
    --adamw-only --weight-decay 0.1 \
    --total-batch-size 8192 --device-batch-size 4 --max-seq-len 2048 \
    --num-iterations 500 \
    --eval-every 50 --eval-tokens 524288 \
    --max-grad-norm 1.0 --keep-best-k 1 --no-save-optimizer \
    --run sft_d18_20k \
    2>&1 | tee "$NOTES/sft-d18_20k_${TS}.txt"
echo "=== Step 4: SFT done ($(date)) ==="

# ─────────────────────────────────────────────
# Step 5: GSM8K + benchmark evals
# ─────────────────────────────────────────────
echo ""
echo "=== Step 5: Evals (started $(date)) ==="

# Determine SFT best step from checkpoint dir
SFT_DIR=~/.cache/nanochat/chatsft_checkpoints/d18_20k
BEST_STEP=$(ls "$SFT_DIR"/model_*.pt 2>/dev/null | grep -oP '\d+' | sort -n | tail -1)
if [ -z "$BEST_STEP" ]; then
    echo "WARNING: No SFT checkpoint found in $SFT_DIR, defaulting to step 500"
    BEST_STEP=500
fi
# Strip leading zeros for the -s flag
BEST_STEP_NUM=$((10#$BEST_STEP))
echo "Using SFT step: $BEST_STEP_NUM"

# GSM8K pass@1
TS=$(date +%F_%H%M)
echo "--- GSM8K pass@1 ---"
WANDB_MODE=disabled "$PYTHON" -m scripts.chat_eval \
    -i sft -g d18_20k -s "$BEST_STEP_NUM" -a GSM8K \
    -t 0 -n 1 -m 1024 -k 50 \
    -x 300 --device-type cuda \
    2>&1 | tee "$NOTES/sft-d18_20k-gsm8k-pass1_${TS}.txt"

# GSM8K pass@8
TS=$(date +%F_%H%M)
echo "--- GSM8K pass@8 ---"
WANDB_MODE=disabled "$PYTHON" -m scripts.chat_eval \
    -i sft -g d18_20k -s "$BEST_STEP_NUM" -a GSM8K \
    -t 0.7 -n 8 -m 1024 -k 50 \
    -x 300 --device-type cuda \
    2>&1 | tee "$NOTES/sft-d18_20k-gsm8k-pass8_${TS}.txt"

# MMLU + SpellingBee
TS=$(date +%F_%H%M)
echo "--- MMLU + SpellingBee ---"
WANDB_MODE=disabled "$PYTHON" -m scripts.chat_eval \
    -i sft -g d18_20k -s "$BEST_STEP_NUM" -a "MMLU|SpellingBee" \
    -x 200 --device-type cuda \
    2>&1 | tee "$NOTES/sft-d18_20k-benchmarks_${TS}.txt"

echo ""
echo "=== All steps complete at $(date) ==="
echo "Check results in $NOTES/"
