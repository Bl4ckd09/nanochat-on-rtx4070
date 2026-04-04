# Fork Automation Scripts

This directory contains fork-specific automation scripts used in local experiments on a single RTX 4070 setup.

These scripts are **not part of upstream `karpathy/nanochat`**. They are included here so experiment orchestration is reproducible from this fork.

## Scope

- Base training/eval automation and fallback handling
- SFT run wrappers (including OOM-safe retries)
- Chat eval packs and confirmation runs
- Queue/watch/hourly monitor helpers

## Path assumptions

Most scripts currently assume this local layout:

- repo: `~/nanochat-learn/nanochat`
- notes/logs: `~/nanochat-learn/notes`
- helper scripts: `~/nanochat-learn/scripts`

If your layout differs, update these path variables inside each script before running.

## High-value scripts

- `run_base_eval_core_resilient.sh`
  Runs base eval, and if CORE crashes on long prompts, reruns with fixed overflow policy:
  `--core-overflow-policy skip --core-max-seq-len 5120`.

- `auto_rerun_baseeval_on_fail.sh`
  Watches a PID/log and triggers resilient base eval rerun when needed.

- `run_chat_eval_confirm_1k.sh`
  Runs 1k confirmation eval pack with Wilson CI summary output.

- `run_post_base_next_steps.sh`
  End-to-end post-base pipeline (compare candidates, optional gap fills, champion selection, optional phase 2).

- `run_additional_eval_suite.sh`
  Runs ARC/ChatCORE/HumanEval plus resilient base-side eval in one script.

- `run_reasoning_manual_v1.sh`
  Validates the manually curated reasoning JSONL and launches the final high-ROI single-4070 experiment only after the curated file is large enough to be meaningful.

- `run_reasoning_manual_v2.sh`
  Runs the broader `manual_reasoning_v2` branch with fixed `s768`, deterministic mode, W&B online, and a 2-seed promotion rule against the current provisional champion.

- `run_teacher_reasoning_v1b.sh`
  Runs the first teacher-generated reasoning branch with fixed `s768`, deterministic mode, W&B online, a 2-seed promotion rule, and the rebalanced `teacher_reasoning_v1b` dataset.

- `run_teacher_reasoning_v2.sh`
  Runs the next teacher-selected reasoning branch with fixed `s768`, deterministic mode, W&B online, a 2-seed promotion rule, and the `teacher_reasoning_v2` dataset built from `OpenThoughts-114k`, `OpenR1-Math-220k`, and filtered `Magpie-Ultra`.

- `run_teacher_reasoning_v3.sh`
  Runs the stricter teacher-selected reasoning branch with fixed `s768`, deterministic mode, W&B online, a 2-seed promotion rule, and the `teacher_reasoning_v3` dataset built with shorter-answer normalization and lower math dominance.

- `run_teacher_distilled_v1.sh`
  Validates a richer raw teacher-distilled JSONL, builds the flat training JSONL, enforces a minimum raw-row guard, and then launches the fixed-`s768`, deterministic, W&B-online 2-seed sweep against the current provisional champion.

- `run_teacher_distilled_v2.sh`
  Builds the broader `teacher_distilled_v2` raw pool, enforces source-balance launch criteria (`Magpie` minimum, math cap, category-share caps), builds the flat training JSONL, and launches the fixed-`s768`, deterministic, W&B-online 2-seed sweep.

- `watch_reasoning_manual_v1_ready.sh`
  Waits until the curated manual JSONL reaches the required row count, then launches `reasoning_manual_v1` automatically with OOM-safe geometry fallbacks and writes a final chain decision.

## Notes

- These scripts are intentionally shell-first and ops-focused.
- Logs are written under the `notes` directory in the expected local layout.
- For strict reproducibility, keep `WANDB_MODE`, model tags, and selection objective explicit in command invocations.
