# nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat is the simplest experimental harness for training LLMs. It is designed to run on a single GPU node, the code is minimal/hackable, and it covers all major LLM stages including tokenization, pretraining, finetuning, evaluation, inference, and a chat UI. Upstream nanochat is currently tuned around an 8xH100 GPT-2 speedrun; this fork documents what happens when the same codebase is pushed on a single consumer GPU instead.

For questions about the repo, I recommend either using [DeepWiki](https://deepwiki.com/karpathy/nanochat) from Devin/Cognition to ask questions about the repo, or use the [Discussions tab](https://github.com/karpathy/nanochat/discussions), or come by the [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) channel on Discord.

## Updates

- (Jan 31 2026) Major revamp of all scripts/README ongoing, deleting midtraining stage, might be a bit messy briefly...
- (Jan 30 2026) With all the latest improvements we're able to train GPT-2 grade LLM in about $73. The [runs/speedrun.sh](runs/speedrun.sh) script will become the refernece way to train GPT-2 grade model and talk to it.

## Fork Status (Upstream vs Local Work)

This repository is a working fork of upstream nanochat:
- Upstream source: `karpathy/nanochat` (`upstream` remote)
- Fork remote: `Bl4ckd09/nanochat-experiments` (`origin` remote)
- Canonical local checkpoint report: [`report_v3.md`](report_v3.md)
- Fork-only automation and recovery scripts: [`tools/automation/README.md`](tools/automation/README.md)

The important distinction is operating point, not code ownership:

| Track | Hardware target | Objective | Typical wall clock | Data / recipe emphasis | Current status |
|---|---|---|---|---|---|
| Upstream `karpathy/nanochat` | `8xH100` node | Beat GPT-2 CORE as fast as possible | `~2 hours` for the current upstream speedrun path | datacenter-scale speedrun recipe and leaderboard tuning | best upstream leaderboard entry is now well below the original 3-hour mark |
| This fork (`origin/master`) | `1x RTX 4070 12GB` | Push the strongest end-to-end nanochat workflow that still fits on a consumer GPU | `~8-15 min` SFT probes, `~4-5 h` confirm eval packs, `~5-8 day` base continuation segments | checkpoint resilience, OOM-safe fallbacks, W&B-connected train/eval, smaller-batch recipes | promoted base is `d24_asp48_track @ step 820230` (`r32`) |

Local code added/changed vs upstream:
- Added `nanochat/lora.py` (LoRA modules + merge/load helpers)
- Extended `scripts/chat_sft.py` with LoRA training flags, better checkpointing, assistant-only masking improvements, safer step-based stopping, and W&B-project overrides
- Extended `nanochat/gpt.py` with chunked cross-entropy and optional gradient checkpointing
- Extended `scripts/chat_eval.py` with W&B logging and a lower-memory categorical path used for MMLU on 12GB GPUs
- Extended CORE eval handling with explicit overflow policies and transparent skipped/evaluated counters
- Added fork-only experiment automation under `tools/automation/` for watchers, retries, eval packs, OOM ladders, and post-run pipelines

## Local Consumer-GPU Setup

All results below were produced on a single local workstation, not on a multi-GPU node:

| Component | Local setup |
|---|---|
| OS | Windows 11 + WSL2 (Ubuntu) |
| CPU | Intel i7-13700K |
| RAM | 64GB |
| GPU | NVIDIA RTX 4070 12GB |
| Base pretraining data | [`karpathy/fineweb-edu-100b-shuffle`](nanochat/dataset.py) parquet shards for the `d24` branch |
| Current SFT data mix | `SmolTalk + MMLU auxiliary train + 6x GSM8K + identity JSON + SimpleSpelling + SpellingBee` in [`scripts/chat_sft.py`](scripts/chat_sft.py) |
| Monitoring | W&B projects: [`nanochat`](https://wandb.ai/sunshines-gmail-com/nanochat), [`nanochat-sft`](https://wandb.ai/sunshines-gmail-com/nanochat-sft), [`nanochat-eval`](https://wandb.ai/sunshines-gmail-com/nanochat-eval) |

## Consumer-GPU Results History

### Base Training History

| Date | Dataset | Model | Params | Recipe | Compute time | Key metrics | Notes |
|---|---|---|---:|---|---|---|---|
| 2026-02-04 | default base smoke | `d6` | `73.5M` | 500-step pretraining smoke test | `0.86m` | `val bpb 1.56`, `~160k tok/s`, `1.1 GB` peak VRAM | first fit check on the 4070 |
| 2026-02-04 | default base smoke | `d12` | `~280M` | 500-step pretraining smoke test | `3.36m` | `val bpb 1.50`, `~40k tok/s`, `3.9 GB` peak VRAM | GPT-1-scale still easy to fit |
| 2026-02-04 | default base smoke | `d16` | `537M` | 500-step pretraining smoke test | `6.83m` | `val bpb 1.493`, `~19k tok/s`, `7.1 GB` peak VRAM | first reasonably large stable model |
| 2026-02-04 | default base smoke | `d18` | `~710M` | 500-step pretraining smoke test | `9.80m` | `val bpb 1.478`, `~13k tok/s`, `9.3 GB` peak VRAM | close to the practical 12GB ceiling without recipe changes |
| 2026-02-04 | default base smoke | `d20 aspect=48` | `599M` | 500-step pretraining smoke test, narrower width | `7.62m` | `val bpb 1.487`, `~17.7k tok/s`, `7.9 GB` peak VRAM | aspect-ratio reduction traded width for depth successfully |
| 2026-02-18 | local `clean200k` branch | `d18_clean200k_safe @ 200000` | `701.9M` | safe long-run base pretraining on the older `d18` branch | `3932.40m` (`65.54h`) | `train bpb 0.847593`, `val bpb 0.903612`, `CORE 0.1255` | first solid long-run local base checkpoint |
| 2026-03-12 | `karpathy/fineweb-edu-100b-shuffle` | `d24_asp48_track @ r24` | `910.7M` | `depth=24`, `aspect=48`, `seq=2048`, grad checkpointing, total batch `16384`, OOM ladder `bs 4 -> 2 -> 1` | `11955.08m` continuation (`199.25h`, `307584 -> 615173`) | `train bpb 0.799226`, `val bpb 0.908073`, `CORE 0.1494` | big jump in base capability on the same 4070 |
| 2026-03-19 | `karpathy/fineweb-edu-100b-shuffle` | `d24_asp48_track @ r32` | `910.7M` | same `d24 aspect=48` recipe continued to target ratio `32` | `7964.55m` continuation (`132.74h`, `615173 -> 820230`) | `train bpb 0.801367`, `val bpb 0.919863`, `CORE 0.1514` | promoted base checkpoint; best local CORE so far |
| 2026-03-27 | `karpathy/fineweb-edu-100b-shuffle` | `d24_asp48_track @ r40` | `910.7M` | same recipe continued to target ratio `40` with frequent checkpoint saves and W&B resume | `7967.71m` continuation (`132.80h`, `820230 -> 1025288`) | `train bpb 0.784647`, `val bpb 0.924377`, `CORE 0.1440` | completed cleanly, but regressed on the comparable base-eval metrics |

For the `r24/r32/r40` rows, "compute time" is the continuation segment measured from the trainer's cumulative clock between saved checkpoints, not a single uninterrupted run from step 0.

### SFT History

All current `d24` chat runs use the consumer-safe SFT mixture in [`scripts/chat_sft.py`](scripts/chat_sft.py): `SmolTalk + MMLU auxiliary train + 6x GSM8K + identity JSON + SimpleSpelling + SpellingBee`.

| Date | Base checkpoint | Trainable params | Recipe | Compute time | Best val bpb | Confirm eval | Notes |
|---|---|---:|---|---|---:|---|---|
| 2026-02-19 | `d18_clean200k_safe @ 200000` | `17.25M / 719.15M` (`2.40%`) | LoRA `r64`, `alpha=128`, `seq=1024`, `1000` steps | `9.75m` | `0.6423` | GSM8K pass@8 `0.80%`, MMLU `26.20%`, SpellingBee `3.52%` | best early local chat result on the `d18` branch |
| 2026-03-27 | `d24_asp48_track @ 820230` | `23.00M / 933.69M` (`2.46%`) | LoRA `r64`, `alpha=128`, `lr=1e-4`, `seq=1536`, grad checkpointing, `1000` steps | `15.41m` | `0.6151` | GSM8K pass@8 `0.20%`, MMLU `21.40%`, SpellingBee `0.00%` | strong SFT loss, weak downstream capability; not promoted |
| 2026-03-27 | `d24_asp48_track @ 820230` | `23.00M / 933.69M` (`2.46%`) | LoRA `r64`, `alpha=128`, `lr=3e-5`, `seq=2048`, grad checkpointing, `500` steps, max-first OOM ladder | `7.96m` | `0.6470` | GSM8K pass@8 `0.00%`, MMLU `21.60%`, SpellingBee `0.00%` | max geometry fit on the 4070, but did not improve chat evals |
| 2026-03-28 | `d24_asp48_track @ 820230` | `23.00M / 933.69M` (`2.46%`) | LoRA `r64`, `alpha=128`, `lr=5e-5`, `seq=2048`, grad checkpointing, `500` steps, same OOM ladder | `7.96m` | `0.6386` | GSM8K pass@8 `0.00%`, MMLU `21.60%`, SpellingBee `0.00%` | confirmed that LR alone was not the bottleneck |

On this hardware, the SFT training probe itself is cheap; the expensive part is the full confirm pack (`GSM8K pass@8 + MMLU + SpellingBee`), which typically adds another `~4-5 hours` of GPU time per candidate.

## Current Local Recommendation

- Best local base checkpoint on this machine is still `d24_asp48_track @ step 820230` (`r32`).
- `r40` is an archive/ablation checkpoint, not the promotion branch.
- The latest `d24` LoRA recipe improved SFT validation loss but not downstream capability, so the next stage should be a recipe/data change, not a longer rerun of the same LR sweep.
- All current training/eval automation is connected to W&B and uses lower-memory eval or OOM fallback paths so that the single-4070 setup can resume and recover cleanly.

## Leaderboard

| # | time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | bump total batch size to 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |
| 4 | 2.02 | 0.71854 | 0.2571 | change dataset to NVIDIA ClimbMix | Mar 4 2026 | 324e69c | @ddudek @karpathy |
| 5 | 1.80 | 0.71808 | 0.2690 | autoresearch round 1 | Mar 9 2026 | 6ed7d1d | @karpathy |
| 6 | 1.65 | 0.71800 | 0.2626 | autoresearch round 2 | Mar 14 2026 | a825e63 | @karpathy |

The primary metric we care about is "time to GPT-2" - the wall clock time needed to outperform the GPT-2 (1.6B) CORE metric on an 8XH100 GPU node. The GPT-2 CORE score is 0.256525. In 2019, the training of GPT-2 cost approximately $43,000, so it is remarkable that current nanochat recipes can now do this in about 2 hours and for well below $100 (e.g. at roughly $3/GPU/hr, an 8XH100 node is about $24/hr, so 2 hours is about $48).

See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for more docs on how to interpret and contribute to the leaderboard.

## Getting started

### Reproduce and talk to GPT-2

The most fun you can have is to train your own GPT-2 and talk to it. The entire pipeline to do so is contained in the single file [runs/speedrun.sh](runs/speedrun.sh), which is designed to be run on an 8XH100 GPU node. Currently, at about ~$24/hour for these nodes, pretraining a GPT-2-grade model takes roughly 2 hours and costs about $48. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash runs/speedrun.sh
```

You may wish to do so in a screen session as this will take about 2 hours to run. Once it's done, you can talk to it via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

A few more notes:

- The code will run just fine on the Ampere 8XA100 GPU node as well, but a bit slower.
- All code will run just fine on even a single GPU by omitting `torchrun`, and will produce ~identical results (code will automatically switch to gradient accumulation), but you'll have to wait 8 times longer.
- If your GPU(s) have less than 80GB, you'll have to tune some of the hyperparameters or you will OOM / run out of VRAM. Look for `--device_batch_size` in the scripts and reduce it until things fit. E.g. from 32 (default) to 16, 8, 4, 2, or even 1. Less than that you'll have to know a bit more what you're doing and get more creative.
- Most of the code is fairly vanilla PyTorch so it should run on anything that supports that - xpu, mps, or etc, but I haven't personally exercised all of these code paths so there might be sharp edges.

## Research

If you are a researcher and wish to help improve nanochat, two scripts of interest are [runs/scaling_laws.sh](runs/scaling_laws.sh) and [runs/miniseries.sh](runs/miniseries.sh). See [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) for related documentation. For quick experimentation (~5 min pretraining runs) my favorite scale is to train a 12-layer model (GPT-1 sized), e.g. like this:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

This uses wandb (run name "d12"), only runs the CORE metric on last step, and it doesn't sample and save intermediate checkpoints. I like to change something in the code, re-run a d12 (or a d16 etc) and see if it helped, in an iteration loop.

The overall approach is to treat the depth of the model as the single dial of complexity. By sweeping out the depth, we get increasingly more powerful models. We determine the scaling laws, set the data budget to a compute optimal setting, train a whole miniseries of models of increasing sizes, and compare them to the GPT-2 and GPT-3 miniseries. Right now, beating GPT-2 specifically faster and faster is the most interesting target.

## Running on CPU / MPS

The script [runs/runcpu.sh](runs/runcpu.sh) shows a very simple example of running on CPU or Apple Silicon. It dramatically shrinks the LLM tha tis being trained to make things fit into a reasonable time interval of a few ten minutes of training. You will not get strong results in this way.

## Guides

I've published a number of guides that might contain helpful information:

- [Oct 13 2025 original nanochat post](https://github.com/karpathy/nanochat/discussions/1) introducing nanochat, though now it contains some deprecated information and the model is a lot older (with worse results) than current master.
- [Jan 7 miniseries v1](https://github.com/karpathy/nanochat/discussions/420) documents the first nanochat miniseries of models.
- To customize your nanochat, see [Guide: infusing identity to your nanochat](https://github.com/karpathy/nanochat/discussions/139) in Discussions, which describes how you can tune your nanochat's personality through synthetic data generation and mixing that data into the SFT stage.
- To add new abilities to nanochat, see [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164).

## File structure

```
.
├── LICENSE
├── README.md
├── dev
│   ├── gen_synthetic_data.py       # Example synthetic data for identity
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py # Pretraining data shard generation
├── nanochat
│   ├── __init__.py                 # empty
│   ├── checkpoint_manager.py       # Save/Load model checkpoints
│   ├── common.py                   # Misc small utilities, quality of life
│   ├── core_eval.py                # Evaluates base model CORE score (DCLM paper)
│   ├── dataloader.py               # Tokenizing Distributed Data Loader
│   ├── dataset.py                  # Download/read utils for pretraining data
│   ├── engine.py                   # Efficient model inference with KV Cache
│   ├── execution.py                # Allows the LLM to execute Python code as tool
│   ├── gpt.py                      # The GPT nn.Module Transformer
│   ├── logo.svg
│   ├── loss_eval.py                # Evaluate bits per byte (instead of loss)
│   ├── optim.py                    # AdamW + Muon optimizer, 1GPU and distributed
│   ├── report.py                   # Utilities for writing the nanochat Report
│   ├── tokenizer.py                # BPE Tokenizer wrapper in style of GPT-4
│   └── ui.html                     # HTML/CSS/JS for nanochat frontend
├── pyproject.toml
├── runs
│   ├── miniseries.sh               # Miniseries training script
│   ├── runcpu.sh                   # Small example of how to run on CPU/MPS
│   ├── scaling_laws.sh             # Scaling laws experiments
│   └── speedrun.sh                 # Train the ~$100 nanochat d20
├── scripts
│   ├── base_eval.py                # Base model: CORE score, bits per byte, samples
│   ├── base_train.py               # Base model: train
│   ├── chat_cli.py                 # Chat model: talk to over CLI
│   ├── chat_eval.py                # Chat model: eval tasks
│   ├── chat_rl.py                  # Chat model: reinforcement learning
│   ├── chat_sft.py                 # Chat model: train SFT
│   ├── chat_web.py                 # Chat model: talk to over WebUI
│   ├── tok_eval.py                 # Tokenizer: evaluate compression rate
│   └── tok_train.py                # Tokenizer: train it
├── tasks
│   ├── arc.py                      # Multiple choice science questions
│   ├── common.py                   # TaskMixture | TaskSequence
│   ├── customjson.py               # Make Task from arbitrary jsonl convos
│   ├── gsm8k.py                    # 8K Grade School Math questions
│   ├── humaneval.py                # Misnomer; Simple Python coding task
│   ├── mmlu.py                     # Multiple choice questions, broad topics
│   ├── smoltalk.py                 # Conglomerate dataset of SmolTalk from HF
│   └── spellingbee.py              # Task teaching model to spell/count letters
├── tests
│   └── test_engine.py
└── uv.lock
```

## Contributing

The goal of nanochat is to improve the state of the art in micro models that are accessible to work with end to end on budgets of < $1000 dollars. Accessibility is about overall cost but also about cognitive complexity - nanochat is not an exhaustively configurable LLM "framework"; there are no giant configuration objects, model factories, or if-then-else monsters in the code base. It is a single, cohesive, minimal, readable, hackable, maximally-forkable "strong baseline" codebase designed to run start to end and produce a ChatGPT model you can talk to. Currently, the most interesting part personally is speeding up the latency to GPT-2 (i.e. getting a CORE score above 0.256525). Currently this takes ~3 hours, but by improving the pretraining stage we can improve this further.

Current AI policy: disclosure. When submitting a PR, please declare any parts that had substantial LLM contribution and that you have not written or that you do not fully understand.

## Acknowledgements

- The name (nanochat) derives from my earlier project [nanoGPT](https://github.com/karpathy/nanoGPT), which only covered pretraining.
- nanochat is also inspired by [modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt), which gamified the nanoGPT repo with clear metrics and a leaderboard, and borrows a lot of its ideas and some implementation for pretraining.
- Thank you to [HuggingFace](https://huggingface.co/) for fineweb and smoltalk.
- Thank you [Lambda](https://lambda.ai/service/gpu-cloud) for the compute used in developing this project.
- Thank you to chief LLM whisperer 🧙‍♂️ Alec Radford for advice/guidance.
- Thank you to the repo czar Sofie [@svlandeg](https://github.com/svlandeg) for help with managing issues, pull requests and discussions of nanochat.

## Cite

If you find nanochat helpful in your research cite simply as:

```bibtex
@misc{nanochat,
  author = {Andrej Karpathy},
  title = {nanochat: The best ChatGPT that \$100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/karpathy/nanochat}
}
```

## License

MIT
