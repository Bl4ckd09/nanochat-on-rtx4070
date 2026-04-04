# Teacher Reasoning v3 Plan

## Goal

Build a teacher-selected dataset that is materially tighter than `teacher_reasoning_v2`:

- shorter answers
- less math dominance
- less latent chain-of-thought style spillover
- better match to the current quick-gate targets

This is a data-construction change, not a training-recipe change.

## Why `v2` failed

`teacher_reasoning_v2` reran cleanly after the disk fix and still failed:

- seed `42`: `GSM8K pass@1 0.00%`, `MMLU 26.40%`
- seed `43`: `GSM8K pass@1 0.40%`, `MMLU 26.40%`

That means the failure is in the dataset shape, not infrastructure.

Likely causes:

1. too much math relative to broad knowledge and reasoning
2. teacher answers are too long and stylistically distant from the current best branch
3. not enough filtering for concise answer-first behavior
4. insufficient rejection of verbose or low-signal synthetic rows

## Design Principles

1. Keep the training backbone fixed:
- base `d24_asp48_track @ 820230`
- partial FT
- `freeze_layers=20`
- `paged_adamw8bit`
- fixed `s768`
- `300` steps
- deterministic

2. Change only the dataset construction.

3. Bias toward short, high-signal answers.

4. Prefer selected teacher rows over raw synthetic volume.

## Source Mix

Use the same source families, but with stricter quotas:

- `OpenThoughts-114k`
- `OpenR1-Math-220k`
- filtered `Magpie-Ultra`

Target total: `420-480` rows, not `520-600`.

Target source split:

- `OpenThoughts-114k`: `180`
- `OpenR1-Math-220k`: `120`
- `Magpie-Ultra`: `120`

## Category Quotas

Target category balance:

- math: `150`
- science: `70`
- puzzle/logic: `70`
- general reasoning: `60`
- data-analysis: `50`
- information-seeking: `40`

Hard rule: math should be at most about `35%` of the final dataset.

## Filtering Rules

### Global rules

Reject any row if:

- assistant answer exceeds `900` characters
- assistant answer exceeds `220` words
- conversation has more than one user turn or one assistant turn
- response contains code blocks
- response is mostly list boilerplate
- response is obviously chain-of-thought formatted rather than concise explanation

### OpenThoughts rules

Accept only if:

- category is `math`, `science`, `puzzle`, or `general-reasoning`
- solution section can be extracted cleanly
- extracted assistant answer is `120-700` characters
- prompt is not code-centric

Reject if prompt contains obvious code-task markers such as:

- `python`
- `javascript`
- `write a function`
- `implement`
- `leetcode`
- `algorithm complexity`

### OpenR1 rules

Accept only if:

- problem is short enough to fit the current style
- solution is not proof-like or essay-length
- final answer is explicit

Normalize by:

- stripping long prefatory text
- keeping short derivation plus final answer
- rejecting very long multi-paragraph traces

### Magpie rules

Accept only categories:

- `reasoning`
- `data-analysis`
- `information-seeking`

Accept only shorter conversations where the assistant answer is:

- concise
- specific
- not roleplay-heavy
- not generic lifestyle chatter

## Formatting Policy

Every assistant answer should be normalized toward:

1. short direct answer
2. brief justification
3. no hidden-thought style markers

Preferred style:

- `Answer: ...`
- `Why: ...`

or a short two-paragraph equivalent.

Avoid:

- long rambling explanations
- “let’s think step by step” phrasing
- excessive bullet nesting
- verbose hedging

## Deduping

Apply aggressive dedupe:

- exact prompt dedupe
- normalized text dedupe
- remove near-duplicate math templates with trivial number changes when the reasoning pattern is identical

## Training Preset

Suggested `teacher_reasoning_v3` training mix:

- `CustomJSON(teacher_reasoning_v3)` x2
- `GSM8K(train)` x1
- `MMLU(auxiliary_train)` x1
- `SmolTalk(train, stop=3000)` x1

SmolTalk glue is reduced further from `v2`.

## Gate

Quick gate:

- `500` problems
- require non-zero `GSM8K pass@1`
- require `MMLU >= 27.0%`

Full confirm only on pass.

Promotion rule remains:

- `2` seeds
- `GSM8K pass@8 >= 4.60%`
- `MMLU >= 27.40%`

## Stop Rule

If `teacher_reasoning_v3` also fails quick gate or produces another unstable one-off spike without replication, stop local recipe churn on this backbone.

At that point, the remaining meaningful levers are:

1. stronger external teacher generation and filtering
2. stronger base model
3. stronger hardware
