# d24 r32 -> r40 Gate Decision

- Generated: `2026-03-13 16:37:36`
- Source log: `/home/sun0115/nanochat-learn/notes/d24_asp48_track_train_r32_p3_bs1_retry_2026-03-13_1557.log`
- W&B run: `https://wandb.ai/sunshines-gmail-com/nanochat/runs/ryf5hccj`
- Decision: **NO_GO**
- Reason: Failed checks: no_bad_lines

## Metrics
- window_start_step: `615174`
- window_end_step: `616173`
- samples: `1000`
- first100_median_loss: `2.615768`
- last100_median_loss: `2.620644`
- loss_delta_last_minus_first: `0.004876`
- min_loss: `2.407043`
- max_loss: `2.814033`
- median_tok_per_sec: `7146`
- min_tok_per_sec: `5626`
- max_tok_per_sec: `7446`
- bad_line_count: `6`

## Next Command
```bash
WANDB_MODE=online WANDB_PROJECT=nanochat EMBEDDING_LR=0.21 UNEMBEDDING_LR=0.0028 MATRIX_LR=0.014 SCALAR_LR=0.35 ~/nanochat-learn/scripts/run_base_d24_asp48_maxcore_safe.sh d24_asp48_track 32
```

## Matched Warning/Error Lines
- `2026-03-13 15:58:25,596 - nanochat.common - [32m[1mINFO[0m - Distributed world size: 1`
- `2026-03-13 15:58:25,597 - nanochat.common - [33m[1mWARNING[0m - Peak flops undefined for: NVIDIA GeForce RTX 4070, MFU will show as 0%`
- `wandb: Run data is saved locally in /home/sun0115/nanochat-learn/nanochat/wandb/run-20260313_155825-ryf5hccj`
- `wandb: ⭐️ View project at https://wandb.ai/sunshines-gmail-com/nanochat`
- `wandb: 🚀 View run at https://wandb.ai/sunshines-gmail-com/nanochat/runs/ryf5hccj`
- `Model torch.compile disabled by NANOCHAT_COMPILE_MODEL=0`
