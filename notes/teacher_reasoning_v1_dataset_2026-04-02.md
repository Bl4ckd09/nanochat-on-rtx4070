# teacher_reasoning_v1 dataset build

## Status

Built but not trained yet.

## Files

- dataset: `/home/sun0115/nanochat-learn/nanochat/data/teacher_reasoning_v1.jsonl`
- metadata: `/home/sun0115/nanochat-learn/nanochat/data/teacher_reasoning_v1_metadata.json`
- builder: `/home/sun0115/nanochat-learn/nanochat/data/build_teacher_reasoning_v1.py`

## Sources

- `argilla/magpie-ultra-v1.0` config `top_300k_shorter_conversations`
- `microsoft/orca-math-word-problems-200k`

## Build policy

- deterministic selection with `seed=42`
- `300` rows from Magpie-Ultra
- `300` rows from Orca-Math
- Magpie categories targeted:
  - `reasoning`
  - `data-analysis`
  - `information-seeking`
  - `math`
- category filtering excludes coding/roleplay/creative-writing style drift

## Result

- total rows: `600`
- source mix:
  - `magpie-ultra`: `300`
  - `orca-math`: `300`
- category mix:
  - `reasoning`: `110`
  - `data-analysis`: `90`
  - `information-seeking`: `50`
  - `math`: `350`

## Caveat

This first version is still math-heavy because `Orca-Math` contributes half the dataset. If this becomes the actual next training branch, reduce the math share or add a broader teacher source before training.
