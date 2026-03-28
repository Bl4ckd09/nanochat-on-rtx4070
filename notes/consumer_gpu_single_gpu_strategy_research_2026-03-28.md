# Consumer Single-GPU Strategy Research

Date: 2026-03-28
Repo: `/home/sun0115/nanochat-learn/nanochat`
Commit referenced: `be76cdf`

## Bottom Line

For the current `1x RTX 4070 12GB` setup, the most feasible path is:

1. Keep `d24_asp48_track @ 820230 (r32)` as the frozen base.
2. Stop spending more time on plain base continuation and plain LoRA LR sweeps.
3. Change the SFT recipe first:
   - smaller, higher-quality dataset
   - fewer mixed objectives in one run
   - cheaper eval gating
4. If moving beyond LoRA, do it with a memory-efficient optimizer path (`Adam8bit`, `GaLore`, or `APOLLO`), not vanilla full AdamW.

This is the best fit for the local evidence and current single-GPU research.

## Why The Current Path Stalled

Local results already show:

- `r32` is still the best base checkpoint; `r40` regressed on the comparable eval path in `/home/sun0115/nanochat-learn/nanochat/report_v3.md`.
- The `d24` LoRA runs improved SFT loss but not downstream capability.
- Full AdamW SFT is currently infeasible on the 4070 with the current optimizer path.

The latest AdamW control run OOMed in `optimizer.step()` even at `seq=384`, which means the bottleneck is optimizer state, not just activations:

- `/home/sun0115/nanochat-learn/notes/sft_d24_r32_adamw_control_2026-03-28_1453_chain.log`

There is also one local code blocker:

- `--freeze-layers` exists in `/home/sun0115/nanochat-learn/nanochat/scripts/chat_sft.py`
- The freeze is applied before training starts.
- But `setup_optimizer()` in `/home/sun0115/nanochat-learn/nanochat/nanochat/gpt.py` still builds optimizer groups from all transformer parameters without filtering `requires_grad`.

So partial freezing does not currently buy optimizer-memory savings. That needs to be fixed before any partial full-tuning experiment is meaningful.

## What Current Research Says

### 1. High-quality SFT data beats large noisy mixtures

- LIMA: https://arxiv.org/abs/2305.11206
- SHED: https://arxiv.org/abs/2405.00705

Takeaway:

- small, high-quality instruction data can outperform much larger mixed datasets
- data selection can matter more than dataset size

This matches the current case: the local SFT mix is broad and heterogeneous, but external eval is not improving.

### 2. QLoRA is a memory enabler, not a guaranteed quality fix

- QLoRA: https://arxiv.org/abs/2305.14314

Takeaway:

- 4-bit frozen-base finetuning with NF4 and paged optimizers is a strong way to fit larger or more expressive adapter runs on limited VRAM
- this mainly solves memory, not recipe quality by itself

### 3. LoRA can fit the SFT task while still hurting broader behavior

- LoRA vs Full Fine-tuning: https://arxiv.org/abs/2410.21228

Takeaway:

- LoRA and full finetuning can reach different solutions
- LoRA can degrade pretraining-distribution behavior even when downstream task fit looks good

This matches the current local pattern: better SFT `val_bpb`, weak MMLU/GSM8K.

### 4. Better adapter variants exist

- PEFT LoRA docs: https://huggingface.co/docs/peft/developer_guides/lora
- DoRA: https://huggingface.co/papers/2402.09353

Takeaway:

- `target_modules="all-linear"` is a stronger default than narrow attention-only targeting
- DoRA and rsLoRA are practical next-step adapter upgrades
- PEFT also supports better initialization variants such as PiSSA and LoftQ-style workflows

### 5. Memory-efficient full or near-full tuning is now realistic on constrained GPUs

- bitsandbytes optimizers: https://huggingface.co/docs/bitsandbytes/optimizers
- Transformers optimizer docs: https://huggingface.co/docs/transformers/en/optimizers
- GaLore: https://arxiv.org/abs/2403.03507
- GaLore repo: https://github.com/jiaweizzhao/GaLore
- APOLLO: https://arxiv.org/abs/2412.05270
- APOLLO repo: https://github.com/zhuhanqing/APOLLO

Takeaway:

- `Adam8bit` is the easiest memory reduction to graft into the current codebase
- `GaLoreAdamW8bit` and `APOLLO` are stronger memory-efficient full-parameter alternatives
- these methods target the exact bottleneck seen locally: optimizer-state memory

## Most Feasible Strategy For This Use Case

Recommended path: data-first SFT retune plus memory-efficient partial or full tuning.

### 1. Fix the local optimizer path first

Patch `setup_optimizer()` to exclude `requires_grad=False` params.

Add one low-memory optimizer path before anything else:

- easiest: `bitsandbytes Adam8bit`
- stronger: `APOLLO-Mini`
- strongest but more work: `GaLoreAdamW8bit`

Why this is first:

- current OOMs happen in optimizer step, not forward pass
- this directly attacks the actual bottleneck

### 2. Stop using one monolithic SFT mix

Current SFT training mix combines:

- SmolTalk
- MMLU auxiliary train
- 6x GSM8K
- identity JSON
- SimpleSpelling
- SpellingBee

in one objective.

Inference from the local results:

- the run is over-mixing unrelated behaviors and damaging transfer

Recommended split:

- `general_chat_reasoning`
- optional `narrow_skill` branch for spelling or identity

### 3. Run smaller curated SFT first

Use a much smaller high-quality SFT set.

Do not optimize for more rows. Optimize for cleaner behavior and better transfer.

On this hardware, this is the cheapest lever with the highest likely upside.

### 4. If staying adapter-based, increase adapter quality instead of only tuning LR

Current adapter coverage is not truly all-linear.

From local code:

- current LoRA definitely hits `c_q`, `c_k`, `c_v`, and any module named `c_proj`
- that means both attention `c_proj` and MLP `c_proj`
- it does not hit `mlp.c_fc`

So the next adapter step is not another LR sweep. It is:

- add `c_fc`
- consider `rsLoRA`
- consider `DoRA`
- optionally use 4-bit base if more memory headroom is needed

## Concrete Experiment Order

### 1. Patch the optimizer path

- filter frozen params in `nanochat/gpt.py`
- add `Adam8bit` option first

### 2. Run a partial full-tune control

- freeze bottom `18` layers
- train top `6` layers plus `lm_head`
- `seq=1024`
- `300` steps
- low-memory optimizer
- W&B on, quick eval gate first

### 3. Run a data-clean SFT ablation

- small curated set
- remove or drastically reduce spelling and identity from the main assistant run
- keep `300`-step probes
- only run full confirm pack on the winner

### 4. If partial FT still loses, run better adapters

- `rsLoRA` or `DoRA`
- extend target coverage to include `mlp.c_fc`
- if memory gets tight, move to QLoRA-style 4-bit base

## What I Would Not Recommend

- more base continuation beyond `r32`
- another plain LoRA LR sweep
- full AdamW retries without changing optimizer memory
- using the full expensive confirm pack on every weak candidate

## Best Single Recommendation

If only one next step is chosen:

- patch `requires_grad` filtering and add `Adam8bit`, then run a top-6-layer partial full-tune on a smaller curated SFT set from `r32`

This is the most feasible consumer-GPU strategy for the exact current bottleneck.

## Local References

- `/home/sun0115/nanochat-learn/nanochat/report_v3.md`
- `/home/sun0115/nanochat-learn/nanochat/README.md`
- `/home/sun0115/nanochat-learn/nanochat/scripts/chat_sft.py`
- `/home/sun0115/nanochat-learn/nanochat/nanochat/gpt.py`
- `/home/sun0115/nanochat-learn/notes/sft_d24_r32_adamw_control_2026-03-28_1453_chain.log`

## External Sources

- LIMA: https://arxiv.org/abs/2305.11206
- SHED: https://arxiv.org/abs/2405.00705
- QLoRA: https://arxiv.org/abs/2305.14314
- LoRA vs Full Fine-tuning: https://arxiv.org/abs/2410.21228
- bitsandbytes optimizers: https://huggingface.co/docs/bitsandbytes/optimizers
- HF optimizers docs: https://huggingface.co/docs/transformers/en/optimizers
- PEFT LoRA docs: https://huggingface.co/docs/peft/developer_guides/lora
- DoRA: https://huggingface.co/papers/2402.09353
- GaLore: https://arxiv.org/abs/2403.03507
- GaLore repo: https://github.com/jiaweizzhao/GaLore
- APOLLO: https://arxiv.org/abs/2412.05270
- APOLLO repo: https://github.com/zhuhanqing/APOLLO
