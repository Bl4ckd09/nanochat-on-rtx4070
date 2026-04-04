# teacher_distilled_v1 Dataset

## Build result

- raw rows: `203`
- emitted training conversations: `203`
- source mix:
  - `OpenThoughts-114k`: `160`
  - `OpenR1-Math-220k`: `40`
  - `Magpie-Ultra`: `3`

## Category mix

- `science`: `60`
- `puzzle`: `50`
- `logic`: `30`
- `grounded_qa`: `20`
- `math`: `40`
- `reasoning`: `3`

## Filters

- low MCQ bias
- code tasks rejected
- short-answer normalization
- assistant max words: `110`
- Magpie minimum reward: `0.24`

## Practical read

This is the first real `teacher_distilled_v1` launchable pool. It clears the row guard, but it is still narrower than ideal because Magpie survived the strict filter only `3` times. That means the branch is dominated by `OpenThoughts` plus a smaller `OpenR1` math booster.

The value of launching this branch is to test whether stricter short-answer teacher filtering beats the failed `teacher_reasoning_v2/v3` mixtures before spending more time on another data pass.
