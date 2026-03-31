# reasoning_focus_v2 plan

Date: 2026-03-29
Base: d24_asp48_track @ step 820230
Goal: recover MMLU from mixv1 while preserving most of the GSM8K gain.

Backbone kept fixed:
- partial full-tune
- freeze_layers=20
- freeze embeddings and residual scalars
- paged_adamw8bit
- seq=1024
- total_batch_size=8192
- num_iterations=300
- init_lr_frac=0.05
- quick gate first, full confirm only on pass

Only changed knob:
- dataset_preset=reasoning_focus_v2

Dataset mix:
- SmolTalk(train, stop=120000)
- MMLU(auxiliary_train, train, stop=100000)
- GSM8K(main, train) x4

Reasoning:
- mixv1 increased GSM8K strongly but gave back some MMLU.
- v2 restores the full MMLU auxiliary pool while keeping the stronger GSM8K weighting and reduced SmolTalk weight.

Promotion rule:
- promote over mixv1 only if GSM8K pass@8 >= 3.0% and MMLU >= 27.0%
- otherwise keep mixv1 as the math-biased main branch winner
- keep spelling out of the main branch regardless
