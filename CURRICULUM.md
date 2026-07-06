# Bookmark-Driven LLM Internals Curriculum

Every unit below is motivated by X/Twitter bookmarks I actually saved — mined from my
second-brain vault (5,728 exported bookmarks → 1,301 keyword candidates → **838
classified as LLM-internals relevant** by LLM triage). Each unit maps to the files in
this fork that implement (or will implement) the concept, and ends with a concrete
experiment with a compute target: **local** (RTX 4070 12GB), **Modal** (serverless
H100s, ~$10–30 per d20-scale ablation), or **Prime Intellect** (H100 pods / verifiers
environments).

Full machine-readable index of all 838 scored bookmarks: `notes/bookmark_index.json`
(generated locally, kept untracked). Regenerate with [`tools/curriculum/`](tools/curriculum/).

**Legend** — ✅ already in this fork (study + ablate it) · 🔨 build from scratch (the
point of this curriculum) · Units are ordered as a build path: tokenizer → attention →
architecture → training → inference → post-training → evals.

| Unit | Topic | Bookmarks | Status |
|---|---|---:|---|
| 0 | Tokenization | 13 | ✅ rustbpe exists · 🔨 SuperBPE |
| 1 | Attention internals | 43 | ✅ QK-norm, softcap · 🔨 gated attention |
| 2 | Efficient attention (GQA/MLA/SWA) | 63 | ✅ GQA knob, SWA patterns · 🔨 MLA |
| 3 | Position embeddings (RoPE) | 6 | ✅ RoPE · 🔨 YaRN long-context |
| 4 | Mixture-of-experts | 45 | 🔨 all of it |
| 5 | Optimizers & scaling laws | 90 | ✅ Muon+AdamW · 🔨 CWD, tokens/param sweep |
| 6 | Pretraining data & recipes | 17 | ✅ FineWeb-edu · 🔨 ClimbMix swap |
| 7 | KV-cache & inference | 127 | ✅ KV cache · 🔨 INT8 KV, speculative decoding |
| 8 | SFT & distillation | 58 | ✅ LoRA, masking · 🔨 LoRA recipe ablations, OPD |
| 9 | RL: GRPO → verifiers | 130 | ✅ simplified GRPO · 🔨 Clip-Higher, verifiers envs |
| 10 | Evals & benchmarks | 33 | ✅ CORE/MMLU/GSM8K packs · 🔨 cloud eval port |
| — | Reference shelf (META) | 213 | reading list |

---

## Unit 0 — Tokenization

**Repo:** `nanochat/tokenizer.py`, `scripts/tok_train.py`, `scripts/tok_eval.py`
**Have:** rustbpe byte-level BPE training. **Build:** cross-word (SuperBPE-style) merges; retokenization-drift awareness (feeds Unit 9).

Motivating bookmarks:
- **@iamgrigorev** — Byte-level BPE from scratch (Rust pretok, LRU+linked-list); SuperBPE cross-word merges give ~20% more sample-efficiency ([x.com](https://x.com/iamgrigorev/status/1975562834793607464))
- **@vllm_project** — Retokenization drift: string round-trips in RL agents change token splits, causing off-policy bugs ([x.com](https://x.com/vllm_project/status/1981017184769061153))
- **@athleticKoder** — A good tokenizer minimizes tokens while staying reversible, maximizing usable context per unit of compute ([x.com](https://x.com/athleticKoder/status/2008516490484482231))
- **@jeremyphoward** — BPE book chapter derived from Karpathy's "Let's build the GPT tokenizer" ([x.com](https://x.com/jeremyphoward/status/1983984624784306640))
- **@sukjun_hwang** — H-Net replaces tokenization with dynamic chunking learned inside the model (frontier reading) ([x.com](https://x.com/sukjun_hwang/status/1943703574908723674))

**Experiment (local, ~free):** add cross-word merge support to `tok_train.py`, retrain
32k vocab on a FineWeb-edu sample, compare bytes/token + downstream CORE on an existing
checkpoint retokenized eval. Metric: tokens-per-byte and effective context gain.

---

## Unit 1 — Attention internals

**Repo:** `nanochat/gpt.py` (`CausalSelfAttention`, `norm`, softcap in `chunked_cross_entropy`)
**Have:** QK-norm, logit softcap, value embeddings, param-free RMSNorm. **Build:** gated attention; attention visualization notebook.

Motivating bookmarks:
- **@Alibaba_Qwen** — Gated Attention adds non-linearity/sparsity and is attention-sink-free (NeurIPS 2025 best paper) ([x.com](https://x.com/Alibaba_Qwen/status/1993854188171006453))
- **@Kimi_Moonshot** — Attention Residuals: replace uniform depth-wise residuals with learned attention over preceding layers ([x.com](https://x.com/Kimi_Moonshot/status/2033378587878072424))
- **@gu_xiangming** — Learnable key bias on attention logits (zero value bias) removes activation outliers → easy quantization (feeds Unit 7) ([x.com](https://x.com/gu_xiangming/status/1952811057673642227))
- **@ProfTomYeh** — Self vs cross attention by hand: S=KᵀQ, F=VA ([x.com](https://x.com/ProfTomYeh/status/2048041421970743588))
- **@docmilanfar** — Attention as Nadaraya-Watson kernel regression ([x.com](https://x.com/docmilanfar/status/1974328880564752525))
- **@1Punch_Girl** — Head dim 64 is the sweet spot: 32 too weak, 128 too costly ([x.com](https://x.com/1Punch_Girl/status/1961406026336346117))

**Experiment (Modal, ~$15):** add `gated_attention: bool` to `GPTConfig` (sigmoid gate
on attention output per Qwen paper), d12 ablation vs baseline at matched params/FLOPs,
compare CORE + loss curves in W&B. Also sweep head_dim 32/64/128 at fixed n_embd.

---

## Unit 2 — Efficient attention: GQA → MLA → sparse

**Repo:** `nanochat/gpt.py` (`n_kv_head`, `window_pattern`), `nanochat/flash_attention.py`, `nanochat/engine.py`
**Have:** GQA (config knob, default 6=6 i.e. MHA), sliding-window patterns (`"SSSL"`), FlashAttention. **Build:** MLA from scratch; KV-bytes/token accounting.

Motivating bookmarks:
- **@rasbt** — From-scratch Multi-Head Latent Attention code + GQA-vs-MHA memory-savings estimator ([x.com](https://x.com/rasbt/status/1977372829973651853))
- **@rasbt** — From-scratch sliding-window attention implementation ([x.com](https://x.com/rasbt/status/1977733802660155875))
- **@cHHillee** — The 3 ways to cut long-context attention cost: local/linear attn, interleave global+local, smaller per-token KV (GQA/MLA/tying/quant) ([x.com](https://x.com/cHHillee/status/2048756662845022655))
- **@HeMuyu0327** — DeepSeek DSA: indexer heads select top-k key tokens per query → sparse attention ([x.com](https://x.com/HeMuyu0327/status/2001163509225415160))
- **@danielhanchen** — DeepSeek V3.2 sparse attention: 128K decode ~10x cheaper than dense ([x.com](https://x.com/danielhanchen/status/1972613546119991791))
- **@cHHillee** — FlexAttention: fused kernels for many attention variants in a few lines ([x.com](https://x.com/cHHillee/status/1821253769147118004))
- **@pranay5255** — FlashAttention-2's Q-outer loop keeps Q/O in SRAM → 2–3x via online softmax ([x.com](https://x.com/pranay5255/status/2040442681625813162))

**Experiments:**
1. *(Modal, ~$20)* GQA sweep: `n_kv_head` 6→2→1 at d20, plot CORE vs KV-bytes/token (add the accounting to `engine.py`).
2. *(Modal, ~$30)* Implement MLA as an alternative `CausalSelfAttention` (rasbt's reference), iso-param d20 ablation vs GQA-2.
3. *(local)* Window-pattern ablation `"SSSL"` vs `"SL"` vs `"L"` on a d12 probe.

---

## Unit 3 — Position embeddings & long context

**Repo:** `nanochat/gpt.py` (`apply_rotary_emb`, cos/sin precompute)
**Have:** RoPE, no learned positions. **Build:** NTK/YaRN scaling for post-hoc context extension.

Motivating bookmarks:
- **@YouJiacheng** — Applying RoPE rotation to V leaks positional info into values; DeepSeek-V4 design nuance ([x.com](https://x.com/YouJiacheng/status/2052467872409203138))
- **@curlysaarthak** — Jane Street: group theory view of positional encodings and attention ([x.com](https://x.com/curlysaarthak/status/2062909849508487202))
- **@himanshustwts** — Kazemnejad's intuition builder for why/how transformers use positional encodings ([x.com](https://x.com/himanshustwts/status/1876334818851696713))
- **@byebyescaling** — Square/sawtooth positional encodings inject spectral leakage; their BLEU gains are artifacts ([x.com](https://x.com/byebyescaling/status/2003504773421773295))

**Experiment (local + Modal eval, ~$5):** implement YaRN-style RoPE rescaling in
`apply_rotary_emb`, extend the promoted d24 checkpoint 2048→8192 with a short LoRA
anneal, eval with a needle-in-haystack task added to `scripts/chat_eval.py`.

---

## Unit 4 — Mixture-of-experts 🔨 (biggest build)

**Repo:** new `nanochat/moe.py` + `MLP` swap in `gpt.py`
**Have:** nothing — dense relu² MLP only. **Build:** top-k router, DeepSeek-style fine-grained + shared experts, sigmoid gating, load-balancing loss.

Motivating bookmarks:
- **@rasbt** — From-scratch Mixture-of-Experts implementation (LLMs-from-scratch ch04/07_moe) ([x.com](https://x.com/rasbt/status/1980269760043446725))
- **@gm8xx8** — DeepSeekMoE ≠ Mixtral: fine-grained + shared experts and sigmoid gating, not just GShard top-k ([x.com](https://x.com/gm8xx8/status/2014676490797842907))
- **@tri_dao** — Rewritten MoE backward halves stored activations (same gradients, no recompute) → ~2x faster training ([x.com](https://x.com/tri_dao/status/2001785266873499875))
- **@scaling01** — Trinity-Large recipe: 400B@13B MoE, gated attn mix, Muon, new load balancing ([x.com](https://x.com/scaling01/status/2016297766809375203))
- **@novasarc01** — nano-moe: detailed from-scratch MoE repo ([x.com](https://x.com/novasarc01/status/1899851725852606658))
- **@cHHillee** — MoE's low arithmetic intensity is why deployment shapes differ from dense ([x.com](https://x.com/cHHillee/status/1973469947889422539))
- **@TheAhmadOsman** — 88-page reference on training MoE models ([x.com](https://x.com/TheAhmadOsman/status/2031309306507694525))

**Experiment (Modal, ~$40–60; too slow on 4070):** `moe.py` with 8 experts × top-2 +
1 shared expert, sigmoid gating, aux-free load-balancing bias. Iso-FLOP ablation: dense
d20 vs MoE-d20-A(active-params-matched), same token budget, CORE + expert-utilization
histograms to W&B. Stretch: Tri Dao's activation-halving backward.

---

## Unit 5 — Optimizers & scaling laws

**Repo:** `nanochat/optim.py` (`MuonAdamW`, `DistMuonAdamW`), `runs/scaling_laws.sh`, `runs/miniseries.sh`
**Have:** Muon+AdamW hybrid, fused steps, scaling-law harness. **Build:** one-line optimizer ablations; a tokens/param study on your own hardware curve.

Motivating bookmarks:
- **@andrewgwils** — nanochat's optimal ~8 tokens/param (vs Chinchilla's 20) explained by better optimization + data quality ([x.com](https://x.com/andrewgwils/status/2021243841836167345))
- **@cwolferesearch** — Cautious Weight Decay: decay only coords whose sign matches the update; one line, consistent gains ([x.com](https://x.com/cwolferesearch/status/2011661378281414930))
- **@percyliang** — Hyperball: normalize updates+parameters instead of weight decay; 33% speedup atop Muon ([x.com](https://x.com/percyliang/status/2014066930164928897))
- **@realsigridjin** — Muon kills ~25% of MLP neurons early (orthogonalization ignores weak grads); Aurora fixes it ([x.com](https://x.com/realsigridjin/status/2053534995424395549))
- **@lilianweng** — Scaling-laws deep dive: Kaplan vs Chinchilla, extrapolation pitfalls ([x.com](https://x.com/lilianweng/status/2070237256070389897))
- **@tanishqkumar07** — Pretraining loss L(N,D) isn't truly a power law; the form is partly an artifact ([x.com](https://x.com/tanishqkumar07/status/1983267357217972491))
- **@HeMuyu0327** — Jianlin Su's first-principles derivation of Muon's SVD computation ([x.com](https://x.com/HeMuyu0327/status/2062018759725076889))
- **@JingyuanLiu123** — Frontier-lab stability toolkit: soft-cap, muP, spectral-norm control, eval-loss prediction ([x.com](https://x.com/JingyuanLiu123/status/1966887747622453560))

**Experiments:**
1. *(Modal, ~$10)* Cautious Weight Decay in `optim.py` — literally a sign-mask line — d12 A/B.
2. *(Modal, ~$25)* Dead-neuron audit of your promoted d24 (activation stats over FineWeb sample); if confirmed, test the Aurora-style fix on a d12 run.
3. *(Modal, ~$50)* Rerun `runs/scaling_laws.sh` miniseries at 4/8/12/20 tokens-per-param to reproduce the ~8 t/p claim on your own curve.

---

## Unit 6 — Pretraining data & recipes

**Repo:** `nanochat/dataset.py`, `nanochat/dataloader.py`
**Have:** FineWeb-edu-100b shards. **Build:** ClimbMix swap (the single biggest gain upstream found), domain-repetition study.

Motivating bookmarks:
- **@karpathy** — nanochat hits GPT-2 in 2h on 8xH100; switching FineWeb-edu → NVIDIA ClimbMix + fp8 was the biggest gain ([x.com](https://x.com/karpathy/status/2029701092347630069))
- **@_christinabaek** — Repeating a small domain dataset 10–50x in pretraining beats larger models on downstream tasks ([x.com](https://x.com/_christinabaek/status/2034285795071205737))
- **@pratyushmaini** — Finetuner's Fallacy: data seen early in pretraining imprints representations that are hard to undo ([x.com](https://x.com/pratyushmaini/status/2034653569706811782))
- **@nrehiew_** — Map of open pretraining datasets (~12–15T tokens across FineWeb-Edu/DCLM/Zyda-2/Dolma) ([x.com](https://x.com/nrehiew_/status/1955109618528456954))
- **@ZeyuanAllenZhu** — Physics of LM: skill-pure synthetic pretraining playgrounds eliminate academic-scale noise ([x.com](https://x.com/ZeyuanAllenZhu/status/2005840089709224260))

**Experiment (Modal, ~$73 = the speedrun):** point `dataset.py` at ClimbMix shards and
run the upstream speedrun operating point on 8xH100 — this simultaneously (a) validates
the biggest known data win and (b) gives the fork its first datacenter-side data point
for the two-operating-point story.

---

## Unit 7 — KV-cache & inference

**Repo:** `nanochat/engine.py` (KV cache, prefill/decode), `scripts/chat_web.py`, `scripts/chat_cli.py`
**Have:** working KV-cache engine. **Build:** INT8 KV quantization, speculative decoding (you have the perfect draft/target pair: d12 + d24), KV accounting.

Motivating bookmarks:
- **@samwhoo** — From-scratch Rust+CUDA: KV caching → 35x throughput; INT8 KV quant → 3.78x smaller cache at same speed ([x.com](https://x.com/samwhoo/status/2036845101561835968))
- **@prajdabre** — Step-by-step speculative decoding build: pair big+small models from one family, benchmark acceptance/speedup ([x.com](https://x.com/prajdabre/status/1986933580564713695))
- **@bnjmn_marie** — KV-cache math per token across architectures (why GQA/MLA choices dominate serving cost) ([x.com](https://x.com/bnjmn_marie/status/2031821490916905089))
- **@AdrianLancucki** — KVTC: 20–40x KV compression via PCA decorrelation + adaptive quantization instead of eviction ([x.com](https://x.com/AdrianLancucki/status/2019748151209476587))
- **@p_nawrot** — DMS distills 8x KV-cache compression end-to-end, beating token-importance eviction ([x.com](https://x.com/p_nawrot/status/2014770473289019709))
- **@TheAhmadOsman** — mini-sglang (~5k LOC): scheduler/tokenizer/detokenizer processes, prefill vs decode batching ([x.com](https://x.com/TheAhmadOsman/status/2020451094665494901))
- **@karpathy** — LLM efficiency hinges on two memory pools: SRAM vs DRAM through a straw ([x.com](https://x.com/karpathy/status/2026452488434651264))

**Experiments (all local — inference fits the 4070):**
1. INT8 KV quant in `engine.py`; measure tok/s + max batch + MMLU delta on the promoted d24.
2. Speculative decoding: d12 drafts, d24 verifies; report acceptance rate and wall-clock speedup in `chat_cli.py`.

---

## Unit 8 — SFT & distillation

**Repo:** `scripts/chat_sft.py`, `nanochat/lora.py`
**Have:** LoRA (+merge), assistant-only masking, your SmolTalk+GSM8K+identity mix. **Build:** the LoRA recipe from the literature; repetition-vs-diversity study; on-policy distillation.

Motivating bookmarks:
- **@gm8xx8** — LoRA matches full FT when applied to *all* layers with ~10x higher LR; RL needs only rank 1 ([x.com](https://x.com/gm8xx8/status/1972744080800309369))
- **@dawkopi** — For long-CoT SFT under a fixed update budget, repeating a small dataset beats more unique samples ([x.com](https://x.com/dawkopi/status/2023496387271553088))
- **@agarwl_** — On-policy distillation = sequence-level reverse-KL minus sampling term → supervised, DAGGER-like ([x.com](https://x.com/agarwl_/status/2065849135383589141))
- **@N8Programs** — KL-regularized SFT (90/10 blend with min-KL-from-base) adds behaviors while preserving the base ([x.com](https://x.com/N8Programs/status/2032871095947219351))
- **@srush_nlp** — On-policy distillation discourages specific rollout mistakes (bad tool calls) ([x.com](https://x.com/srush_nlp/status/2062359839783657816))
- **@cwolferesearch** — SFT minimizes forward KL, RL minimizes reverse KL — why their learning mechanics differ ([x.com](https://x.com/cwolferesearch/status/2012551263099949143))

**Experiments (local — SFT probes already run in 8–15 min):**
1. LoRA all-layers + 10x LR vs your current config; W&B A/B on the standard eval pack.
2. Fixed update budget: 1 epoch × N samples vs 4 epochs × N/4 on the GSM8K slice.
3. *(stretch, Modal)* On-policy distillation from a small open teacher into d24.

---

## Unit 9 — RL: GRPO → verifiers 🔨 (capstone)

**Repo:** `scripts/chat_rl.py`, `tasks/gsm8k.py`
**Have:** simplified GRPO (on-policy, no ratio/clip, mean-baseline advantage, GSM8K only). **Build:** the modern GRPO toolkit, then verifiers environments on Prime Intellect.

Motivating bookmarks:
- **@cwolferesearch** — GRPO tricks: Clip-Higher (decoupled upper clip) and Dynamic Sampling curb entropy collapse ([x.com](https://x.com/cwolferesearch/status/2008245160883208214))
- **@rasbt** — From-scratch GRPO chapter: 0.6B base on MATH-12k lifts MATH-500 15%→47% ([x.com](https://x.com/rasbt/status/2012897755916579278))
- **@DimitrisPapail** — Tinker drops GRPO for plain REINFORCE: advantage = reward − mean, no clipping (validates nanochat's current choice!) ([x.com](https://x.com/DimitrisPapail/status/1973470706135605534))
- **@maxrumpf** — Retokenizing chat messages in RL creates rare tokens that dominate gradients → collapse; go tokens-in/tokens-out ([x.com](https://x.com/maxrumpf/status/2001128217030304240))
- **@guohao_li** — Dense reward: unit-test pass *ratio* (+ full-pass bonus) instead of sparse 0/1 ([x.com](https://x.com/guohao_li/status/2010480985570250861))
- **@natolambert** — RLHF Book single-GPU minimal impls: REINFORCE, RLOO, PPO, GRPO, GSPO, CISPO, reward models ([x.com](https://x.com/natolambert/status/2015473455530225939))
- **@iScienceLuvr** — Apply policy gradient only to the ~20% high-entropy "fork" tokens in CoT ([x.com](https://x.com/iScienceLuvr/status/1929750117927797143))
- **@askalphaxiv** — MaxRL: scale advantages by 1/μ not 1/σ to lift low pass rates ([x.com](https://x.com/askalphaxiv/status/2024594970339090683))

**Experiments:**
1. *(local probe → Modal confirm, ~$20)* Add `--advantage {mean,zscore,maxrl}`, `--clip-higher`, and dynamic sampling flags to `chat_rl.py`; ablate on GSM8K pass@1 with entropy curves in W&B.
2. *(local)* Audit the rollout path for retokenization drift (Unit 0 tie-in); make it strictly tokens-in/tokens-out.
3. *(Prime Intellect pod, ~$30–80)* New `tasks/verifiers_env.py` adapter: wrap a [verifiers](https://github.com/willccbb/verifiers) environment as a nanochat task (reward fn + rollout format), train d24-SFT against a math/code env from the Environments Hub. This is the fork's headline feature: **nanochat × verifiers**.

---

## Unit 10 — Evals & benchmarks

**Repo:** `nanochat/core_eval.py`, `nanochat/loss_eval.py`, `scripts/chat_eval.py`, `scripts/base_eval.py`, `tools/automation/` eval packs
**Have:** CORE + MMLU + GSM8K with overflow policies, W&B eval logging, 4–5h eval packs. **Build:** cloud eval port (biggest quality-of-life win), variance-aware reporting, one contamination-safe private eval.

Motivating bookmarks:
- **@iamgrigorev** — nanochat re-implements lm-eval-harness in 2 Python files, easy to adapt to custom evals ([x.com](https://x.com/iamgrigorev/status/1979583139828650423))
- **@ArtificialAnlys** — Repeat benchmarks many times, report median + percentiles, not single runs ([x.com](https://x.com/ArtificialAnlys/status/1955102409044398415))
- **@kenziyuliu** — Eval on organic *unsolved* problems via reference-free validation to avoid contamination ([x.com](https://x.com/kenziyuliu/status/1960388584567136762))
- **@abhijitwt** — Cautionary tale: a model gamed BrowseComp by finding the benchmark's answers on GitHub ([x.com](https://x.com/abhijitwt/status/2030574688640901274))
- **@clefourrier** — LLM Evaluation Guidebook v2 ([x.com](https://x.com/clefourrier/status/1996250279033839918))

**Experiment (Modal, ~$5–10 per pack):** wrap the existing eval pack in a Modal
function (A100 is plenty) — 4–5h local → ~20 min cloud. Add `--repeats N` to
`chat_eval.py` and report median ± IQR. Then every ablation above gets cheap,
variance-aware evals.

---

## Reference shelf (META, 213 bookmarks)

The from-scratch canon your bookmarks keep returning to — not units, but the reading
spine for the whole curriculum:

- **@karpathy** — nanochat itself: ~8k-line full-stack pipeline ([x.com](https://x.com/karpathy/status/1977755427569111362)) · the 243-line dependency-free GPT ([x.com](https://x.com/karpathy/status/2021694437152157847)) · autoresearch: agent-driven nanochat tweaks, d12→d24 transfer cutting time-to-GPT-2 to 1.80h ([x.com](https://x.com/karpathy/status/2031135152349524125)) · fp8 training → 2.91h/$20-spot ([x.com](https://x.com/karpathy/status/2018804068874064198)) · scaling-law science via a single compute dial ([x.com](https://x.com/karpathy/status/2009037707918626874))
- **@rasbt** — big LLM architecture comparison: MLA, SWA, Post/Pre-Norm, NoPE, shared-expert MoE ([x.com](https://x.com/rasbt/status/1946549778319339931))
- **@_djdumpling** — distillation of 7 frontier open-weight reports: architecture → RL ([x.com](https://x.com/_djdumpling/status/2024203932709552352))
- **@GoSailGlobal** — 2026 open-LLM architecture template ([x.com](https://x.com/GoSailGlobal/status/2051494907974697144))
- **@iamgrigorev** — from-scratch PyTorch parallelism: DDP, ZeRO, FSDP, TP ([x.com](https://x.com/iamgrigorev/status/1984394295428649136))
- **@jacobaustin132** — JAX scaling book, GPU chapter ([x.com](https://x.com/jacobaustin132/status/1957447351011840336))

…plus 207 more in the index.

---

## Suggested execution order

Cheapest-first, each step de-risking the next:

1. **Unit 10** Modal eval port — makes everything else measurable for ~$5/run
2. **Unit 5.1** Cautious Weight Decay A/B — one line, first cloud ablation (~$10)
3. **Unit 2.1** GQA sweep + KV accounting (~$20)
4. **Unit 7** INT8 KV + speculative decoding — local, big demo value
5. **Unit 9.1–9.2** GRPO toolkit + tokens-in/tokens-out (~$20)
6. **Unit 4** MoE from scratch (~$50)
7. **Unit 6** ClimbMix speedrun — the $73 datacenter data point
8. **Unit 9.3** verifiers × nanochat on Prime Intellect — capstone

Total budget for the full pass: **~$250–350** of cloud compute.

---

*Generated 2026-07-06 from the second-brain vault (`ingest/twitter` exporter → Obsidian).
Pipeline: keyword extraction (1,301/5,728) → 6-way parallel LLM classification into 12
units with 0–3 implementability scores (838 relevant) → curated here. Full scored index:
`notes/bookmark_index.json` (local, untracked). Scripts to regenerate: `tools/curriculum/`.*
