# Insights Log

Collected during development sessions. Each insight is tagged with the context it arose from.

---

## Gradient Checkpointing (2026-02-19)

**Context:** Creating `base_train_gc.py` to enable d20 pretraining on 12GB VRAM.

- **How it works:** Gradient checkpointing trades compute for memory. Instead of storing all intermediate activations during the forward pass, it recomputes them during backward. In `gpt.py`, when `self.gradient_checkpointing=True` and `self.training`, each transformer block uses `torch.utils.checkpoint.checkpoint()` to wrap the forward — reducing peak VRAM from O(layers) activations to O(1).

- **Memory savings:** d20 at seq_len=512, batch_size=4 goes from OOM (>12GB) to ~11.5 GB peak. Without GC, activations for all 20 transformer blocks are held simultaneously; with GC, only ~1 block's activations are stored at a time.

- **Throughput cost:** ~30-50% slowdown is typical. In this test: ~5.3s/step with GC vs an estimated ~3-4s without. For VRAM-constrained training, this tradeoff is well worth it.

- **Placement matters:** The `gradient_checkpointing` flag must be set on the raw `GPT` instance *before* `torch.compile()`. The compiler traces the forward graph including checkpoint boundaries. Setting it after compile would not propagate into the already-traced graph.

---

## Memory Feasibility: d24 asp24 vs asp48 (2026-02-19)

**Context:** Checking whether d24 with smaller aspect ratios fits on 12GB (RTX 4070). All tests at seq_len=2048, device_batch_size=4.

| Config | Params | No GC | With GC | Fits 12GB? |
|--------|--------|-------|---------|------------|
| d24 asp24 | 412M | 9,753 MiB (9.5 GB) | 5,181 MiB (5.1 GB) | Yes, even without GC |
| d24 asp48 | 911M | 17,497 MiB (17.1 GB) | 11,406 MiB (11.1 GB) | Only with GC |
| d20 asp64 (ref) | 897M | OOM (>12GB) | 11,514 MiB* | Only with GC |

*d20 asp64 GC result was at seq_len=512, bs=4. At seq_len=2048 it would use more.

- **d24 asp24** is very comfortable — even without GC it only uses 9.5 GB. With GC, just 5.1 GB leaves room to increase batch size or seq_len.
- **d24 asp48** is similar in parameter count to d20 asp64 (~900M) but uses 24 deeper, narrower layers. It fits with GC at 11.1 GB — tight but workable on 12GB.
- **GC savings scale with model width:** asp24 saves ~4.6 GB (47%), asp48 saves ~6.1 GB (35%). Wider models store larger activation tensors per layer, so GC's absolute savings grow with width.
- **Depth vs width tradeoff:** d24 asp48 (911M, 24 layers x 1152 dim) uses slightly less memory with GC than d20 asp64 (897M, 20 layers x 1280 dim) despite having more params — because GC cost is per-layer and narrower layers have smaller activations.
- **Batch size scaling with GC (d24 asp48, seq2048):** bs=4 → 11,406 MiB, bs=8 → 11,492 MiB (+86), bs=16 → 14,408 MiB (+2,916). The jump at bs=16 shows activation memory overtaking the param/optimizer floor. **bs=8 is max for 12GB.**
- **Non-linear memory cliff at bs=16:** The +2.9 GB jump (vs +86 MiB for bs=4→8) likely comes from `torch.compile` choosing different memory layouts or kernels at the larger batch size that don't fit in the previous allocation pattern. CUDA's `expandable_segments:True` allocator helps but can't eliminate these cliffs.
