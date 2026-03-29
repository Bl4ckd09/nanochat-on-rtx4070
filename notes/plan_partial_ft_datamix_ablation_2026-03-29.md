# Partial FT Data-Mix Ablation Plan (2026-03-29)

## Goal
Improve the current best `r32` partial full-tune recipe by changing only the SFT data mix.

## Current Winner
- Base: `d24_asp48_track @ 820230`
- Recipe: partial FT, `freeze_layers=20`, embeddings/scalars frozen
- Optimizer: `paged_adamw8bit`
- Sequence length: `1024`
- Total batch size: `8192`
- Horizon: `300` steps
- Current best confirm:
  - GSM8K pass@8: `1.20%`
  - MMLU: `27.40%`
  - SpellingBee: `0.00%`

## Hypothesis
The bottleneck is recipe alignment, not memory fit or base size.
The current `general_chat_reasoning` preset still overweights broad conversational data relative to reasoning/math targets.

## Fixed Knobs
Keep these unchanged for the first ablation:
- base checkpoint: `r32 @ 820230`
- `freeze_layers=20`
- `paged_adamw8bit`
- `seq=1024`
- `device_batch_size=1`
- `total_batch_size=8192`
- `num_iterations=300`
- `init_lr_frac=0.05`
- `warmup_ratio=0.2`
- `warmdown_ratio=0.3`
- same quick-gate / full-confirm automation

## Changed Knob
Only change the dataset mix.

### New Preset: `reasoning_focus_v1`
- `SmolTalk(train, stop=120000)`
- `MMLU(auxiliary_train, train, stop=50000)`
- `GSM8K(main, train)` x4

### Rationale
- Reduce general chat weight without removing it.
- Keep MMLU auxiliary constant to preserve broad reasoning pressure.
- Increase GSM8K weight materially without making the run math-only.
- Avoid changing multiple recipe knobs at once.

## Expected Effect
- Target: higher GSM8K without losing MMLU.
- Acceptable: similar MMLU with better GSM8K.
- Failure mode: lower MMLU and flat GSM8K, indicating the new mix is too narrow or noisy.

## Gate Logic
### Quick gate (250 problems)
- Run `GSM8K pass@1` and `MMLU`.
- Escalate to full confirm if `MMLU >= 27.0%`.
- Do not require non-zero `pass@1`, because the current winner had `pass@1 = 0` but still improved on full `pass@8`.

### Full confirm (1000 problems)
- Run `GSM8K pass@8`, `MMLU`, `SpellingBee`.

## Decision Rule
Promote `reasoning_focus_v1` over current winner only if:
- GSM8K pass@8 improves materially, and
- MMLU does not regress meaningfully.

Practical tie-break:
- Prefer higher GSM8K if MMLU stays within about `0.3-0.5` points.
- Reject if MMLU drops clearly below current winner.

## If This Fails
Next ablation should still stay in the same backbone recipe family and change only one axis:
1. `reasoning_focus_v2`: reduce `SmolTalk` further
2. `reasoning_focus_v1` + lower effective LR / longer warmdown
3. only after that, try a freeze-boundary change again
