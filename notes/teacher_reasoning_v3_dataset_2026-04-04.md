# Teacher Reasoning v3 Dataset

Built from stricter teacher-selected sources for the next fixed-`s768` branch.

Files:

- `/home/sun0115/nanochat-learn/nanochat/data/teacher_reasoning_v3.jsonl`
- `/home/sun0115/nanochat-learn/nanochat/data/teacher_reasoning_v3_metadata.json`
- `/home/sun0115/nanochat-learn/notes/build_teacher_reasoning_v3_2026-04-04.log`

Final shape:

- rows: `420`
- source mix:
  - `OpenThoughts-114k`: `180`
  - `OpenR1-Math-220k`: `120`
  - `Magpie-Ultra`: `120`
- category mix:
  - math: `148`
  - science: `60`
  - puzzle: `50`
  - general-reasoning: `42`
  - reasoning: `45`
  - data-analysis: `45`
  - information-seeking: `30`

Formatting/selection profile:

- max user chars: `700`
- max assistant chars: `900`
- max assistant words: `220`
- style: `short-answer normalized`
- reward threshold for Magpie rows: `0.18`

Observed content profile:

- average user chars: about `280`
- average assistant chars: about `153`
- average assistant words: about `26`
- p95 assistant chars: `471`
- p95 assistant words: `81`
- MCQ-style prompts detected: `100/420`
- answer-letter rows detected: `106/420`

Interpretation:

- This is materially shorter and more normalized than `teacher_reasoning_v2`.
- It is still somewhat MCQ-heavy, which may help MMLU-style behavior but could also blunt open-ended reasoning gains.
- The branch is worth one disciplined 2-seed test because it is a true data-construction change, not another minor mix sweep.
