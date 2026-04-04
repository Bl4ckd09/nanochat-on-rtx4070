# teacher_distilled_v2 Dataset

## Build result

- raw rows: `278`
- emitted training conversations: `278`
- source mix:
  - `OpenThoughts-114k`: `130`
  - `OpenR1-Math-220k`: `35`
  - `Magpie-Ultra`: `100`
  - `manual_reasoning_v1` anchors: `13`

## Category mix

- `science`: `52`
- `reasoning`: `51`
- `puzzle`: `40`
- `math`: `39`
- `data_analysis`: `30`
- `logic`: `28`
- `information_seeking`: `20`
- `grounded_qa`: `18`

## Launch criteria check

- raw rows >= `240`: pass
- `Magpie` rows >= `60`: pass (`100`)
- math <= `20%`: pass (`14.03%`)
- no non-science category > `30%`: pass
- science <= `35%`: pass
- built training JSONL >= `240`: pass

## Practical read

This is materially better balanced than `teacher_distilled_v1`.

Key changes that landed:
- Magpie contribution is now real instead of negligible
- OpenThoughts no longer dominates the pool alone
- OpenR1 remains a small math booster
- a small set of manual anchors survived into the pool

One caveat remains: the manual-anchor slice did not contribute any `information_seeking` rows after filtering, so the broad reasoning/data-analysis coverage is doing more of the conversational glue work than the manual anchors are.
