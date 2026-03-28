# Environment Setup Notes

## System
- OS: Windows 11 + WSL2 (Ubuntu)
- CPU: i7-13700K
- RAM: 64GB
- GPU: RTX 4070 12GB

## Software Versions
- Python: 3.12.3
- PyTorch: 2.6.0+cu124 (CUDA 12.4)
- NVIDIA Driver: 576.02
- Transformers: 5.0.0
- Datasets: 4.5.0

## Installation Steps
```bash
# 1. Install python3-venv (needed sudo)
sudo apt install -y python3.12-venv

# 2. Create project directory and venv
mkdir -p ~/nanochat-learn
cd ~/nanochat-learn
python3 -m venv .venv

# 3. Clone nanochat
git clone https://github.com/karpathy/nanochat.git

# 4. Install PyTorch with CUDA
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 5. Install dependencies
pip install datasets fastapi ipykernel matplotlib psutil python-dotenv \
    regex rustbpe scipy tabulate tiktoken tokenizers transformers \
    uvicorn wandb zstandard numpy
```

## Activation
```bash
cd ~/nanochat-learn
source .venv/bin/activate
```

## Notes
- nanochat repo wants torch 2.9.1, we have 2.6.0+cu124 - works fine for learning
- VRAM shows as 11GB in PyTorch (12GB physical, some reserved)
