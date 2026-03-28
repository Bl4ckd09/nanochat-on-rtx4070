# d24 Ratio Ladder Plan (24 -> 32 -> 40) + W&B Monitoring
*Updated: 2026-03-13*

## Current Baseline
- Model tag: `d24_asp48_track`
- Completed step: `615173` (ratio `24`)
- Next targets:
  - ratio `32` -> step `820230` (remaining `205057`)
  - ratio `40` -> step `1025288` (remaining `410115`)

## Goal
1. Continue `24 -> 32` first, watch first `1k-3k` resumed steps.
2. If stable, continue `32 -> 40`.
3. If loss spikes, restart with lower LR using env overrides.
4. Keep run visible in W&B for live monitoring.

## Stage A: Launch Ratio 32 (W&B online)
Run in tmux:
```bash
tmux new-session -d -s d24_r32 \
  "bash -lc 'WANDB_MODE=online WANDB_PROJECT=nanochat ~/nanochat-learn/scripts/run_base_d24_asp48_maxcore_safe.sh d24_asp48_track 32; exec bash'"
```

Check session:
```bash
tmux ls | rg d24_r32
```

Get active log path:
```bash
ls -1t ~/nanochat-learn/notes/d24_asp48_track_train_r32_* | head -1
```

## Stage B: Monitor First 1k-3k Steps
Terminal monitor (step/eta/core/bpb/gpu + W&B URL line):
```bash
~/nanochat-learn/scripts/monitor_base.sh "$(ls -1t ~/nanochat-learn/notes/d24_asp48_track_train_r32_* | head -1)" 30
```

Hourly heartbeat log:
```bash
tmux new-session -d -s d24_r32_hourly \
  "bash -lc 'BASE_LOG=$(ls -1t ~/nanochat-learn/notes/d24_asp48_track_train_r32_* | head -1); ~/nanochat-learn/scripts/hourly_base_update.sh \"$BASE_LOG\" ~/nanochat-learn/notes/d24_r32_hourly.log 3600; exec bash'"
```

Extract W&B URL from log:
```bash
rg -n "wandb:.*(https://wandb.ai|View run at)" "$(ls -1t ~/nanochat-learn/notes/d24_asp48_track_train_r32_* | head -1)" | tail -1
```

Current live run (started 2026-03-13):
- Session: `d24_r32`
- Active log: `~/nanochat-learn/notes/d24_asp48_track_train_r32_p3_bs1_retry_2026-03-13_1557.log`
- W&B run: `https://wandb.ai/sunshines-gmail-com/nanochat/runs/ryf5hccj`
- Hourly monitor log: `~/nanochat-learn/notes/d24_r32_hourly.log`

## Stability Gate (after 1k-3k resumed steps)
Proceed to ratio 40 if all are true:
- No repeated OOM restarts after settling to the successful batch-size profile.
- No NaN/Inf/loss explosion lines in log.
- Train loss trend is not persistently diverging over the observed window.
- Throughput and memory are in normal range for your previous stable `bs=1` profile.

Quick checks:
```bash
LOG=$(ls -1t ~/nanochat-learn/notes/d24_asp48_track_train_r32_* | head -1)
rg -n "out of memory|nan|inf|overflow|CUDA error" "$LOG" | tail -20
rg -n "^step " "$LOG" | tail -120
```

## Stage C: Continue Ratio 32 -> 40
When stable:
```bash
tmux new-session -d -s d24_r40 \
  "bash -lc 'WANDB_MODE=online WANDB_PROJECT=nanochat ~/nanochat-learn/scripts/run_base_d24_asp48_maxcore_safe.sh d24_asp48_track 40; exec bash'"
```

## Stage D: Loss Spike Recovery (Lower LR Relaunch)
If spike/divergence appears, relaunch from latest checkpoint with reduced LRs:
```bash
tmux new-session -d -s d24_r32_lowlr \
  "bash -lc 'WANDB_MODE=online WANDB_PROJECT=nanochat EMBEDDING_LR=0.21 UNEMBEDDING_LR=0.0028 MATRIX_LR=0.014 SCALAR_LR=0.35 ~/nanochat-learn/scripts/run_base_d24_asp48_maxcore_safe.sh d24_asp48_track 32; exec bash'"
```

For ratio 40 recovery, same pattern with target `40`.

## Notes
- The launcher now uses `WANDB_MODE` from env (default `online`) and keeps run names unique automatically.
- Auto-resume remains enabled and continues from latest checkpoint in `d24_asp48_track`.
