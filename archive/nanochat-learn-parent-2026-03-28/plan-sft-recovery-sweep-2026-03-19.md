# SFT Recovery Sweep Plan (d24_r32 base)

Date: 2026-03-19
Base checkpoint: `d24_asp48_track` step `820230`

## Goal
Recover downstream chat eval quality after latest SFT regression by running short, gated SFT trials and selecting the best config for promotion.

## Sweep Trials (300 steps each)
1. `s1024_lr5e5`: `max_seq_len=1024`, `total_batch_size=8192`, `lora_lr=5e-5`
2. `s1280_lr5e5`: `max_seq_len=1280`, `total_batch_size=7680`, `lora_lr=5e-5`
3. `s1536gc_lr3e5`: `max_seq_len=1536`, `total_batch_size=7680`, `lora_lr=3e-5`, `--gradient-checkpoint`

Shared settings:
- `device_batch_size=1`, `num_iterations=300`
- `lora_rank=64`, `lora_alpha=128`, `lora_dropout=0.0`
- `eval_every=50`, `eval_tokens=524288`
- `max_grad_norm=1.0`, `keep_best_k=5`, `no_save_optimizer`

## Quick Eval Gate per trial
Run quick confirm eval on each trial's `best` checkpoint:
- script: `run_chat_eval_confirm_1k.sh`
- step: `300`
- max problems: `200`
- tasks: GSM8K pass@8, MMLU, SpellingBee

## Selection Rule
Rank trials by weighted score:
- `score = 3*GSM8K_pass8 + 1*MMLU + 0.5*SpellingBee`
- choose highest score as promotion candidate.

## Promotion
After sweep finishes, run a full 1000-problem confirm eval on the winning config before any longer training.
