# teacher_distilled_v2 Plan

## Why `teacher_distilled_v1` failed

`teacher_distilled_v1` failed before external eval on both seeds.

Observed issues:
- raw pool was too narrow: `160` OpenThoughts, `40` OpenR1, only `3` Magpie rows
- internal validation stayed far above the eval gate:
  - seed 42 best val bpb: `2.6745`
  - seed 43 best val bpb: `1.9363`
- the stricter short-answer filtering likely over-pruned broad non-math teacher data
- the resulting dataset was dominated by academic/problem-style rows and lacked enough broad reasoning/chat glue

Practical conclusion:
- `v1` was too narrow and too teacher-style-homogeneous
- the next branch should widen broad reasoning coverage, not add more math

## Goal

Build a better-balanced teacher-distilled branch that preserves short-answer discipline but restores broader reasoning coverage and enough conversational glue to avoid the `v1` collapse.

## Backbone

Keep fixed:
- base: `d24_asp48_track @ 820230`
- partial FT
- `freeze_layers=20`
- `paged_adamw8bit`
- fixed `s768`
- `300` steps
- deterministic mode on
- 2 seeds

## Raw dataset target

Target raw rows: `260-320`

Target source mix:
- `OpenThoughts-114k`: `120-140`
- `OpenR1-Math-220k`: `30-40`
- `Magpie-Ultra`: `80-120`
- optional small manual anchors: `10-20`

Target category mix:
- `science`: `45-55`
- `puzzle`: `35-45`
- `logic`: `25-35`
- `grounded_qa`: `25-35`
- `reasoning`: `35-50`
- `data_analysis`: `20-30`
- `information_seeking`: `15-25`
- `math`: `25-35`

Constraint:
- math cap around `15%`
- MCQ-like prompts should stay low

## Main changes vs `v1`

1. Increase Magpie contribution materially
- lower Magpie reward floor from `0.24` to about `0.16-0.18`
- keep short-answer normalization
- continue excluding code and long CoT spillover

2. Reduce OpenThoughts dominance
- lower OpenThoughts quota from `160` to around `130`
- keep science/puzzle/logic, but trim grounded QA if it starts dominating

3. Keep OpenR1 as a small math booster only
- do not exceed `40` rows

4. Allow a small manual anchor slice
- optional `10-20` best rows from `manual_reasoning_v1`
- only short, high-confidence rows
- no spelling branch data

## Filtering rules

Keep:
- no code tasks
- no long chain-of-thought style spillover
- assistant max words around `110-130`
- short answer plus brief justification format

Loosen slightly from `v1`:
- Magpie reward threshold
- assistant max chars if needed
- allow more broad reasoning/data-analysis rows through

Still reject:
- answer-letter-only rows
- heavy MCQ prompts
- long templated rubric outputs
- multi-turn dialogues longer than 4 messages

## Launch criteria

Do not launch until all are true:
- raw rows >= `240`
- Magpie rows >= `60`
- math rows <= `20%`
- no category over `30%` except science may touch low `30s`
- validator passes cleanly
- built training JSONL emits at least `240` conversations

## Gating and promotion

Keep current sweep policy:
- quick gate on `500`
- require non-zero `GSM8K pass@1`
- require `MMLU >= 27.0%`
- full confirm only on pass

Promotion band remains:
- `GSM8K pass@8 >= 4.60%`
- `MMLU >= 27.40%`
- require `2` seeds

## Success criteria

`teacher_distilled_v2` is worth continuing only if:
- at least one seed reaches full confirm
- and at least one seed lands near or above the current provisional champion band

If both seeds fail internal or quick gate again:
- stop the distilled branch family on this base
- keep `mixv2_s768_300` as official provisional champion
- treat stronger improvement as requiring a better teacher source or a stronger base/hardware change

## Implementation order

1. patch the raw builder quotas and Magpie threshold
2. regenerate `teacher_distilled_v2_raw.jsonl`
3. validate category/source mix against the launch criteria
4. build `teacher_distilled_v2.jsonl`
5. launch the fixed-`s768` 2-seed sweep under W&B
6. update `report_v3.md` after the branch settles
