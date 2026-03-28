# SFT Recovery Sweep v2 Summary

- tag: `d24_sft_recovery_v2_2026-03-21_0105`
- base: `d24_asp48_track@820230`
- iterations per trial: `300`
- eval max problems: `200`
- categorical eval batch size: `1`

## Results

| label | out_tag | seq | lora_lr | gsm8k_pass8 | mmlu | spellingbee | score |
|---|---|---:|---:|---:|---:|---:|---:|
| s1024gc_lr3e5 | `d24_sft_recovery_v2_2026-03-21_0105_s1024gc_lr3e5` | 1024 | 3e-5 | NA | NA | NA | -1 |
| s1280gc_lr3e5 | `d24_sft_recovery_v2_2026-03-21_0105_s1280gc_lr3e5` | 1280 | 3e-5 | 0.00 | 22.50 | 0.00 | 22.5000 |
| s1536gc_lr2e5 | `d24_sft_recovery_v2_2026-03-21_0105_s1536gc_lr2e5` | 1536 | 2e-5 | 0.00 | 23.00 | 0.00 | 23.0000 |

## Winner

- label: `s1536gc_lr2e5`
- out_tag: `d24_sft_recovery_v2_2026-03-21_0105_s1536gc_lr2e5`
- score: `23.0000`

Promotion command:
```bash
CAT_BATCH_SIZE=1 ~/nanochat-learn/scripts/run_chat_eval_confirm_1k.sh d24_sft_recovery_v2_2026-03-21_0105_s1536gc_lr2e5/best 300 1000
```
