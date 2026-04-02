# teacher_reasoning_v1b Result

Date: 2026-04-02

## Outcome

- status: `recipe_failed_quick_gate`
- dataset: `teacher_reasoning_v1b`
- base: `d24_asp48_track @ 820230`
- recipe: partial FT, `freeze_layers=20`, `paged_adamw8bit`, fixed `s768`, `300` steps, deterministic

## Seed Results

- seed `42`
  - run: `teacher_reasoning_v1b_seed42_2026-04-02_1621_s768_gc`
  - best val bpb: `1.9556`
  - external eval skipped because internal gate failed
- seed `43`
  - run: `teacher_reasoning_v1b_seed43_2026-04-02_1643_s768_gc`
  - best val bpb: `1.8408`
  - external eval skipped because internal gate failed

## Interpretation

This teacher-generated `v1b` branch did not clear the internal SFT gate on either seed, so it never reached the quick external eval stage. It does not challenge the current provisional champion `mixv2_s768_300`.

## Notes

- The original sweep wrapper aborted early when a run had no summary file because eval was skipped.
- `run_reasoning_seed_sweep.sh` was patched so skipped-eval runs are recorded correctly instead of terminating the whole sweep.
- The missing seed `43` leg was resumed and completed separately, then folded back into the final sweep decision.
