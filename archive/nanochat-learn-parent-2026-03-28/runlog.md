# Run Log

## 2026-02-04 - Phase 0: Environment Setup

### System Info
| Component | Value |
|-----------|-------|
| OS | Windows 11 + WSL2 Ubuntu |
| Python | 3.12.3 |
| PyTorch | 2.6.0+cu124 |
| CUDA | Available |
| GPU | NVIDIA GeForce RTX 4070 |
| VRAM | 11 GB (usable) |
| Driver | 576.02 |

### Sanity Check
```
torch.cuda.is_available() = True
torch.cuda.get_device_name(0) = NVIDIA GeForce RTX 4070
```

### Status
Phase 0 complete. Ready for Phase 1 (tiny training).

---

## 2026-02-04 - Phase 1: Tiny Training Run

### Configuration
```bash
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=4 \
    --total-batch-size=16384 \
    --num-iterations=500
```

### Model Info
- Parameters: 73.5M
- Layers: 6
- Model dim: 384
- Heads: 6

### Results
| Metric | Value |
|--------|-------|
| Steps | 500 |
| Time | 0.86 min |
| Loss | 5.6 → 5.2 |
| Val bpb | 3.14 → 1.56 |
| Throughput | ~160k tok/sec |
| Peak VRAM | 1.1 GB |

### Checkpoint
`~/.cache/nanochat/base_checkpoints/d6/model_000500.pt`

### Notes
- Required patching `nanochat/optim.py` for PyTorch 2.6.0 compatibility (lerp_ dtype casting)
- Model is very small, outputs are repetitive but show basic learning
- Lots of headroom on VRAM - could increase batch size or model size

### SFT (Phase 2 preview)
- Ran 63 steps on identity data (~9 seconds)
- Loss: 0.72 → 0.006 (memorized the data)
- Model too small to chat meaningfully (outputs only BOS token)
- Would need larger model or more training for useful chat

---
