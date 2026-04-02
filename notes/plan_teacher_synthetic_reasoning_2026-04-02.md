# Teacher-Synthetic Reasoning Plan

## Goal

Use a materially different lever from the exhausted local mix-sweep family: teacher-generated reasoning data with targeted selection.

## Why this is next

Local evidence now says:

- `mixv2_s768_300` is the official provisional champion because it is the best stable-enough confirmed run so far.
- `manual_reasoning_v1` can spike above it on GSM8K in a single seed, but it does not replicate reliably.
- `manual_reasoning_v2` broadened the manual set and lowered oversampling pressure, but both seeds failed the `500`-problem quick gate.

That means the remaining bottleneck is not memory fit. It is data quality, breadth, and stability.

## Source ideas

Use these methods as the design pattern:

- `Self-Instruct` for teacher-driven instruction generation: https://arxiv.org/abs/2212.10560
- `Orca 2` for diverse reasoning strategies in small models: https://arxiv.org/abs/2311.11045
- `Magpie` for synthetic alignment-style data generation with minimal seed prompting: https://arxiv.org/abs/2406.08464
- `LESS` for targeted data selection instead of training on everything: https://arxiv.org/abs/2402.04333

Preference optimization is not the next step here. Capability/data instability is the current problem, not style alignment.

## Recommendation

Build `teacher_reasoning_v1` in two phases:

1. generate a candidate pool
2. select a compact subset that best matches the target capabilities

## Candidate pool

Generate `2k-5k` candidate conversations with a stronger teacher.

Preferred teacher order:

1. strongest available external teacher model
2. your best observed unstable local reasoning artifact only as a weak auxiliary source
3. never use the weaker local branches as the main teacher

Candidate categories:

- GSM8K-style arithmetic and word problems
- MMLU-style short academic QA with concise reasoning
- logic / evidence choice / causal-claims analysis
- calibration / metric interpretation / error analysis
- small amount of format-discipline and safe refusal

Do not include spelling-specialist data in the main branch.

## Generation policy

For each prompt family, generate multiple answer styles:

- direct concise solution
- short step-by-step solution
- answer-then-justify
- compare-options-then-answer

This follows the main useful lesson from `Orca 2`: strategy diversity matters for smaller models.

## Selection policy

Do not train on the entire synthetic pool.

Use a LESS-style targeted selection policy:

- create a small seed set of desired capabilities
- rank candidate items by relevance to that capability set
- keep the best `300-800` examples, not all generated data

Practical local approximation if you do not implement full LESS:

- score items manually or heuristically for:
  - reasoning clarity
  - correctness
  - subject breadth
  - format discipline
  - non-redundancy
- remove near-duplicates aggressively
- keep the final set balanced across task families

## Initial dataset shape

Target `teacher_reasoning_v1.jsonl` with roughly:

- `200` quantitative / GSM8K-like
- `200` MMLU-style breadth examples
- `100` logic / evidence / causal reasoning
- `50` metric interpretation / error analysis
- `25` format-discipline examples
- `25` safe refusal examples

Total target: `600` rows.

## Training recipe

Keep the backbone fixed to avoid attribution drift.

- base: `d24_asp48_track @ 820230`
- partial FT
- `freeze_layers=20`
- `paged_adamw8bit`
- fixed `s768`
- `300` steps
- deterministic mode on
- same W&B split: `nanochat-sft` and `nanochat-eval`

Suggested mixture:

- `CustomJSON(teacher_reasoning_v1)` x2
- `GSM8K(train)` x1
- `MMLU(auxiliary_train)` x1
- optional `SmolTalk(train, stop=5000)` x1

Avoid manual oversampling beyond `x2`.

## Evaluation rule

Quick gate first:

- `500` problems
- require non-zero `GSM8K pass@1`
- require `MMLU >= 27.0%`

Full confirm only on pass.

Promotion rule stays strict:

- `2` seeds must clear
- `GSM8K pass@8 >= 4.60%`
- `MMLU >= 27.40%`

## Stop rule

If this teacher-synthetic branch still fails to produce a stable improvement, stop local recipe churn on the 4070 and treat the current project state as saturated on this hardware/base combination.

At that point, only three high-ROI moves remain:

1. stronger teacher and better synthetic data
2. stronger base model
3. stronger hardware
