# Eval Then Next Plan Summary

- tag: `d24_eval_then_next_2026-03-20_1529`
- base: `d24_asp48_track@820230`
- quick eval max problems: `200`
- categorical eval batch size: `1`

## Quick Eval Results

| label | model_group | step | gsm8k_pass8 | mmlu | spellingbee | score | status |
|---|---|---:|---:|---:|---:|---:|---|
| control_r32_s1000 | `d24_r32_lora_nextbest_2026-03-19_1310_s1536_gc/best` | 1000 | 0.00 | 21.50 | 0.00 | 21.5000 | ok |
| recovery_s1536_s300 | `d24_sft_recovery_2026-03-20_1216_s1536gc_lr3e5/best` | 300 | 0.00 | 22.50 | 0.00 | 22.5000 | ok |
| recovery_s1280_s50 | `d24_sft_recovery_2026-03-20_1216_s1280_lr5e5/best` | 50 | 0.00 | 21.50 | 0.00 | 21.5000 | ok |
| recovery_s1024_s50 | `d24_sft_recovery_2026-03-20_1216_s1024_lr5e5/best` | 50 | 0.00 | 22.00 | 0.00 | 22.0000 | ok |

## Best Candidate

- label: `recovery_s1536_s300`
- model_group: `d24_sft_recovery_2026-03-20_1216_s1536gc_lr3e5/best`
- step: `300`
- score: `22.5000`
- control score: `21.5000`

## Full Confirm (1000 problems)

- candidate: `d24_sft_recovery_2026-03-20_1216_s1536gc_lr3e5/best` step `300`
- rc: `0`

