"""
Supervised fine-tuning (SFT) the model.
Run as:

python -m scripts.chat_sft

Or torchrun for training:

torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16
"""

import argparse
import copy
import gc
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch
from contextlib import nullcontext
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.checkpoint_manager import load_model
from nanochat.lora import apply_lora_to_model, count_lora_params, count_total_params, get_lora_state_dict, merge_lora_weights
import torch.distributed as dist

from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mmlu import MMLU
from tasks.smoltalk import SmolTalk
from tasks.customjson import CustomJSON
from tasks.spellingbee import SimpleSpelling, SpellingBee

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Supervised fine-tuning (SFT) the model")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
parser.add_argument("--seed", type=int, default=42, help="global random seed")
parser.add_argument("--deterministic", action="store_true", help="request deterministic kernels where possible")
# Model loading
parser.add_argument("--model-source", type=str, default="base", choices=["base", "sft"], help="checkpoint source to load from")
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Output model tag (checkpoint directory name)
parser.add_argument("--output-tag", type=str, default=None, help="model tag to save checkpoints under (default: same as --model-tag or d{depth})")
# Training horizon
parser.add_argument("--num-iterations", type=int, default=-1, help="number of optimization steps (-1 = full epoch)")
# Batch sizes
parser.add_argument("--max-seq-len", type=int, default=2048, help="max context length")
parser.add_argument("--device-batch-size", type=int, default=32, help="per-device batch size")
parser.add_argument("--total-batch-size", type=int, default=524288, help="total batch size in tokens")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.3, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=1.0, help="initial LR as fraction of base LR")
parser.add_argument("--warmup-ratio", type=float, default=0.1, help="fraction of training for LR warmup (default 0.1)")
parser.add_argument("--warmdown-ratio", type=float, default=0.2, help="fraction of training for LR decay (default 0.2)")
parser.add_argument("--adamw-only", action="store_true", help="Use AdamW for all params instead of Muon+AdamW (recommended for SFT)")
parser.add_argument("--freeze-layers", type=int, default=0,
                    help="Freeze first N transformer layers (0 = no freezing)")
parser.add_argument("--freeze-embeddings", action="store_true", help="Freeze token/value embeddings during fine-tuning")
parser.add_argument("--freeze-scalars", action="store_true", help="Freeze residual scaling parameters during fine-tuning")
parser.add_argument("--optimizer", type=str, default="default", choices=["default", "adam8bit", "paged_adamw8bit"], help="optimizer backend for Adam-family fine-tuning")
# LoRA (Low-Rank Adaptation)
parser.add_argument("--lora", action="store_true", help="Enable LoRA fine-tuning (freezes base model)")
parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank (default 8)")
parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha scaling (default 16)")
parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout (default 0)")
parser.add_argument("--lora-lr", type=float, default=1e-4, help="Learning rate for LoRA params (default 1e-4)")
# Evaluation
parser.add_argument("--eval-every", type=int, default=150, help="evaluate val bpb every N steps (-1 = disable)")
parser.add_argument("--eval-tokens", type=int, default=20*524288, help="number of tokens to evaluate val loss on")
# Output
parser.add_argument("--dry-run", action="store_true", help="log to wandb but skip checkpoints/report")
# Checkpoint management
parser.add_argument("--keep-best-k", type=int, default=1, help="Keep only the K most recent best checkpoints (min 1)")
parser.add_argument("--no-save-optimizer", action="store_true", help="Skip saving optimizer state in final checkpoint")
# Gradient clipping
parser.add_argument("--max-grad-norm", type=float, default=0.0, help="Max gradient norm for clipping (0 = disable)")
parser.add_argument("--gradient-checkpoint", action="store_true", help="Enable gradient checkpointing (recompute activations in backward to save VRAM)")
parser.add_argument("--dataset-preset", type=str, default="default", choices=["default", "general_chat_reasoning", "reasoning_focus_v1", "reasoning_focus_v2", "reasoning_focus_v3", "reasoning_focus_v4", "reasoning_curated_v1", "reasoning_manual_v1", "reasoning_manual_v2", "teacher_reasoning_v1b", "teacher_reasoning_v2", "teacher_reasoning_v3", "teacher_distilled_v1", "teacher_distilled_v2", "curriculum_boost_v1", "curriculum_boost_v2"], help="training dataset mixture preset")
parser.add_argument("--manual-reasoning-jsonl", type=str, default=os.environ.get("MANUAL_REASONING_JSONL", ""), help="path to a curated or teacher-generated reasoning/chat JSONL file used by custom reasoning presets")
args = parser.parse_args()
assert args.keep_best_k >= 1, f"--keep-best-k must be >= 1, got {args.keep_best_k}"
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type, seed=args.seed, deterministic=args.deterministic)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

def maybe_compile(model_module):
    if args.lora:
        print0("Skipping torch.compile in LoRA mode for stability")
        return model_module
    if os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1":
        print0("TORCH_COMPILE_DISABLE=1, skipping torch.compile")
        return model_module
    return torch.compile(model_module, dynamic=False)

def build_bnb_optimizer(param_groups, paged=False):
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise ImportError(
            "bitsandbytes is not installed. Install it in the nanochat virtualenv to use --optimizer adam8bit/paged_adamw8bit."
        ) from exc
    factory = bnb.optim.PagedAdamW8bit if paged else bnb.optim.Adam8bit
    return factory(param_groups)

def build_train_dataset(base_dir, preset):
    identity_conversations_filepath = os.path.join(base_dir, "identity_conversations.jsonl")
    if args.manual_reasoning_jsonl:
        manual_reasoning_jsonl = args.manual_reasoning_jsonl
    elif preset == "reasoning_manual_v2":
        manual_reasoning_jsonl = os.path.join(base_dir, "data", "manual_reasoning_chat_v2.jsonl")
    elif preset == "teacher_reasoning_v1b":
        manual_reasoning_jsonl = os.path.join(base_dir, "data", "teacher_reasoning_v1b.jsonl")
    elif preset == "teacher_reasoning_v2":
        manual_reasoning_jsonl = os.path.join(base_dir, "data", "teacher_reasoning_v2.jsonl")
    elif preset == "teacher_reasoning_v3":
        manual_reasoning_jsonl = os.path.join(base_dir, "data", "teacher_reasoning_v3.jsonl")
    elif preset == "teacher_distilled_v1":
        manual_reasoning_jsonl = os.path.join(base_dir, "data", "teacher_distilled_v1.jsonl")
    elif preset == "teacher_distilled_v2":
        manual_reasoning_jsonl = os.path.join(base_dir, "data", "teacher_distilled_v2.jsonl")
    else:
        manual_reasoning_jsonl = os.path.join(base_dir, "data", "manual_reasoning_chat_v1.jsonl")
    if preset == "default":
        return TaskMixture([
            SmolTalk(split="train"), # 460K rows of general conversations
            MMLU(subset="auxiliary_train", split="train"), # 100K rows of multiple choice problems drawn from ARC, MC_TEST, OBQA, RACE
            GSM8K(subset="main", split="train"), # 8K rows teaching simple math and (calculator) tool use
            GSM8K(subset="main", split="train"), # 2x GSM8K
            GSM8K(subset="main", split="train"), # 3x GSM8K
            GSM8K(subset="main", split="train"), # 4x GSM8K
            GSM8K(subset="main", split="train"), # 5x GSM8K
            GSM8K(subset="main", split="train"), # 6x GSM8K
            CustomJSON(filepath=identity_conversations_filepath), # 1000 rows of synthetic identity conversations
            CustomJSON(filepath=identity_conversations_filepath), # let's do 2 epochs of these
            SimpleSpelling(size=200000, split="train"), # 200K rows of Simple Spelling (e.g. spell the word 'apple')
            SpellingBee(size=80000, split="train"), # 80K rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
        ])
    if preset == "general_chat_reasoning":
        return TaskMixture([
            SmolTalk(split="train", stop=200000),
            MMLU(subset="auxiliary_train", split="train", stop=50000),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
        ])
    if preset == "reasoning_focus_v1":
        return TaskMixture([
            SmolTalk(split="train", stop=120000),
            MMLU(subset="auxiliary_train", split="train", stop=50000),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
        ])
    if preset == "reasoning_focus_v2":
        return TaskMixture([
            SmolTalk(split="train", stop=120000),
            # Use the full auxiliary split; hard-coding 100000 exceeded the current dataset size.
            MMLU(subset="auxiliary_train", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
        ])
    if preset == "reasoning_focus_v3":
        return TaskMixture([
            SmolTalk(split="train", stop=160000),
            MMLU(subset="auxiliary_train", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
        ])
    if preset == "reasoning_focus_v4":
        return TaskMixture([
            SmolTalk(split="train", stop=200000),
            MMLU(subset="auxiliary_train", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
        ])
    if preset == "reasoning_curated_v1":
        return TaskMixture([
            # Smaller, cleaner general-chat anchor while keeping strong reasoning pressure.
            SmolTalk(split="train", stop=60000),
            MMLU(subset="auxiliary_train", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
        ])
    if preset == "reasoning_manual_v1":
        return TaskMixture([
            # High-signal manually curated conversations should dominate this branch.
            CustomJSON(filepath=manual_reasoning_jsonl),
            CustomJSON(filepath=manual_reasoning_jsonl),
            CustomJSON(filepath=manual_reasoning_jsonl),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
            MMLU(subset="auxiliary_train", split="train"),
            # Keep only a small amount of generic chat glue.
            SmolTalk(split="train", stop=20000),
        ])
    if preset == "reasoning_manual_v2":
        return TaskMixture([
            # Broader curated data, lower oversampling pressure, and the same stable reasoning backbone.
            CustomJSON(filepath=manual_reasoning_jsonl),
            CustomJSON(filepath=manual_reasoning_jsonl),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
            MMLU(subset="auxiliary_train", split="train"),
            SmolTalk(split="train", stop=10000),
        ])
    if preset == "teacher_reasoning_v1b":
        return TaskMixture([
            # Teacher-generated reasoning data dominates this branch, with one GSM8K and one MMLU anchor.
            CustomJSON(filepath=manual_reasoning_jsonl),
            CustomJSON(filepath=manual_reasoning_jsonl),
            GSM8K(subset="main", split="train"),
            MMLU(subset="auxiliary_train", split="train"),
            SmolTalk(split="train", stop=5000),
        ])
    if preset == "teacher_reasoning_v2":
        return TaskMixture([
            # Better-selected teacher data from OpenThoughts/OpenR1/Magpie with the same stable backbone recipe.
            CustomJSON(filepath=manual_reasoning_jsonl),
            CustomJSON(filepath=manual_reasoning_jsonl),
            GSM8K(subset="main", split="train"),
            MMLU(subset="auxiliary_train", split="train"),
            SmolTalk(split="train", stop=5000),
        ])
    if preset == "teacher_reasoning_v3":
        return TaskMixture([
            # Stricter teacher-selected data with shorter normalized answers and lower math dominance.
            CustomJSON(filepath=manual_reasoning_jsonl),
            CustomJSON(filepath=manual_reasoning_jsonl),
            GSM8K(subset="main", split="train"),
            MMLU(subset="auxiliary_train", split="train"),
            SmolTalk(split="train", stop=3000),
        ])
    if preset == "teacher_distilled_v1":
        return TaskMixture([
            # Teacher-distilled short-answer data should dominate this branch.
            CustomJSON(filepath=manual_reasoning_jsonl),
            CustomJSON(filepath=manual_reasoning_jsonl),
            GSM8K(subset="main", split="train"),
            MMLU(subset="auxiliary_train", split="train"),
            SmolTalk(split="train", stop=2000),
        ])
    if preset == "teacher_distilled_v2":
        return TaskMixture([
            # Better-balanced teacher-distilled short-answer data with more broad reasoning glue.
            CustomJSON(filepath=manual_reasoning_jsonl),
            CustomJSON(filepath=manual_reasoning_jsonl),
            GSM8K(subset="main", split="train"),
            MMLU(subset="auxiliary_train", split="train"),
            SmolTalk(split="train", stop=3000),
        ])
    if preset == "curriculum_boost_v1":
        return TaskMixture([
            SmolTalk(split="train", stop=80000),
            MMLU(subset="auxiliary_train", split="train", stop=50000),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
        ])
    if preset == "curriculum_boost_v2":
        return TaskMixture([
            SmolTalk(split="train", stop=120000),
            MMLU(subset="auxiliary_train", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
            GSM8K(subset="main", split="train"),
        ])
    raise ValueError(f"Unknown dataset preset: {preset}")

def build_val_dataset():
    return TaskMixture([
        SmolTalk(split="test"), # 24K rows in test set
        MMLU(subset="all", split="test", stop=5200), # 14K rows in test set, use only 5.2K to match the train ratios
        GSM8K(subset="main", split="test", stop=420), # 1.32K rows in test set, use only 420 to match the train ratios
    ])

# wandb logging init
wandb_project = os.environ.get("WANDB_PROJECT", "nanochat-sft")
wandb_entity = os.environ.get("WANDB_ENTITY") or None
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(
    project=wandb_project,
    entity=wandb_entity,
    name=args.run,
    config=user_config,
)

# Load the model and tokenizer
model, tokenizer, meta = load_model(args.model_source, device, phase="train", model_tag=args.model_tag, step=args.model_step)
pretrain_batch_size = meta.get("device_batch_size", None)
if args.model_source == "base" and pretrain_batch_size is not None and args.device_batch_size > pretrain_batch_size:
    print0(f"FOOTGUN WARNING: base model training used device_batch_size {pretrain_batch_size}, did you pass in a good --device-batch-size to this script?")
orig_model = model
if args.gradient_checkpoint:
    orig_model.gradient_checkpointing = True
    print0("Gradient checkpointing enabled (activation recomputation in backward)")
model = maybe_compile(model)
depth = model.config.n_layer
output_dirname = args.output_tag if args.output_tag else (args.model_tag if args.model_tag else f"d{depth}") # e.g. d12

# Freeze early transformer layers if requested (not compatible with LoRA)
if args.freeze_layers > 0 and not args.lora:
    freeze_count = min(args.freeze_layers, depth - 1)  # Keep at least 1 layer trainable
    for i in range(freeze_count):
        for param in orig_model.transformer.h[i].parameters():
            param.requires_grad = False
    print0(f"Froze first {freeze_count} of {depth} transformer layers")
if args.freeze_embeddings and not args.lora:
    for param in orig_model.transformer.wte.parameters():
        param.requires_grad = False
    for param in orig_model.value_embeds.parameters():
        param.requires_grad = False
    print0("Froze token and value embeddings")
if args.freeze_scalars and not args.lora:
    orig_model.resid_lambdas.requires_grad = False
    orig_model.x0_lambdas.requires_grad = False
    print0("Froze residual scaling parameters")

# Apply LoRA if requested (freezes all base params, only trains LoRA adapters)
lora_params = None
if args.lora:
    orig_model, lora_params, lora_count = apply_lora_to_model(
        orig_model,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=['c_q', 'c_k', 'c_v', 'c_proj']  # attention layers only
    )
    total_params = count_total_params(orig_model)
    trainable_params = count_lora_params(lora_params)
    print0(f"LoRA enabled: rank={args.lora_rank}, alpha={args.lora_alpha}")
    print0(f"Applied LoRA to {lora_count} layers")
    print0(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    model = maybe_compile(orig_model)

trainable_params = [p for p in orig_model.parameters() if p.requires_grad]
trainable_param_count = sum(p.numel() for p in trainable_params)
total_param_count = sum(p.numel() for p in orig_model.parameters())
print0(f"Trainable params after freezing/adaptation: {trainable_param_count:,} / {total_param_count:,} ({100*trainable_param_count/total_param_count:.2f}%)")

num_flops_per_token = model.estimate_flops()
tokens_per_fwdbwd = args.device_batch_size * args.max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert args.total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = args.total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {args.device_batch_size} x {args.max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {args.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
token_bytes = get_token_bytes(device=device)

# Initialize the Optimizer
if args.lora:
    # LoRA mode: Adam-family optimizer for LoRA params only (all base params are frozen)
    lora_param_groups = [dict(kind='adamw', params=lora_params, lr=args.lora_lr * args.init_lr_frac, betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)]
    if args.optimizer == "default":
        print0(f"Using AdamW optimizer for LoRA params with lr={args.lora_lr}")
        optimizer = torch.optim.AdamW(lora_param_groups)
    else:
        paged = args.optimizer == "paged_adamw8bit"
        print0(f"Using {args.optimizer} optimizer for LoRA params with lr={args.lora_lr}")
        optimizer = build_bnb_optimizer(lora_param_groups, paged=paged)
    for group in optimizer.param_groups:
        group["kind"] = "adamw"  # for compatibility with LR scheduler
        group["initial_lr"] = group["lr"]
else:
    # Standard mode: Muon+AdamW or AdamW-only
    if args.optimizer == "default":
        optimizer = orig_model.setup_optimizer(unembedding_lr=args.unembedding_lr, embedding_lr=args.embedding_lr, matrix_lr=args.matrix_lr, weight_decay=args.weight_decay, adamw_only=args.adamw_only)
    else:
        if not args.adamw_only:
            raise ValueError(f"--optimizer {args.optimizer} currently requires --adamw-only in non-LoRA mode")
        param_groups = orig_model.build_optimizer_param_groups(
            unembedding_lr=args.unembedding_lr,
            embedding_lr=args.embedding_lr,
            matrix_lr=args.matrix_lr,
            weight_decay=args.weight_decay,
            adamw_only=True,
        )
        paged = args.optimizer == "paged_adamw8bit"
        print0(f"Using {args.optimizer} for trainable params")
        optimizer = build_bnb_optimizer(param_groups, paged=paged)
    # Override the initial learning rate as a fraction of the base learning rate
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]
        group.setdefault("kind", "adamw")

# SFT data mixture and DataLoader
base_dir = get_base_dir()
train_dataset = build_train_dataset(base_dir, args.dataset_preset)
val_dataset = build_val_dataset()
print0(f"Using dataset preset: {args.dataset_preset}")
print0(f"Train dataset conversations: {len(train_dataset):,}")
print0(f"Val dataset conversations: {len(val_dataset):,}")
# DataLoader is defined here, it emits inputs, targets : 2D tensors of shape (device_batch_size, max_seq_len)
# A big problem is that we don't know the final num_iterations in advance. So we create
# these two global variables and update them from within the data generator.
last_step = False # we will toggle this to True when we reach the end of the training dataset
approx_progress = 0.0 # will go from 0 to 1 over the course of the epoch
current_epoch = 1 # track epoch for logging
def sft_data_generator_bos_bestfit(split, buffer_size=100):
    """
    BOS-aligned dataloader for SFT with bestfit-pad packing.

    Each row in the batch starts with BOS (beginning of a conversation).
    Conversations are packed using best-fit algorithm. When no conversation fits,
    the row is padded (instead of cropping) to ensure no tokens are ever discarded.
    Padding positions have targets masked with -1 (ignore_index for cross-entropy).
    """
    global last_step, approx_progress, current_epoch
    assert split in {"train", "val"}, "split must be 'train' or 'val'"
    dataset = train_dataset if split == "train" else val_dataset
    dataset_size = len(dataset)
    assert dataset_size > 0
    row_capacity = args.max_seq_len + 1  # +1 for target at last position
    bos_token = tokenizer.get_bos_token_id()

    # Conversation buffer: list of (ids, mask) tuples
    conv_buffer = []
    cursor = ddp_rank  # Each rank processes different conversations (for fetching)
    consumed = ddp_rank  # Track actual consumption separately from buffering
    epoch = 1

    def refill_buffer():
        nonlocal cursor, epoch
        while len(conv_buffer) < buffer_size:
            conversation = dataset[cursor]
            # Pass max_tokens=row_capacity to match our actual sequence length
            # Keep the mask for assistant-only loss masking
            ids, mask = tokenizer.render_conversation(conversation, max_tokens=row_capacity)
            conv_buffer.append((ids, mask))  # Store (ids, mask) tuple
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor = cursor % dataset_size
                epoch += 1
                # Note: last_step is now triggered based on consumption, not fetching

    while True:
        rows = []  # List of (row_ids, row_mask) tuples
        for _ in range(args.device_batch_size):
            row_ids = []
            row_mask = []
            while len(row_ids) < row_capacity:
                # Ensure buffer has conversations
                while len(conv_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - len(row_ids)

                # Find largest conversation that fits entirely
                # conv_buffer contains (ids, mask) tuples, so check len(conv[0])
                best_idx = -1
                best_len = 0
                for i, conv in enumerate(conv_buffer):
                    conv_len = len(conv[0])  # conv is (ids, mask), check len of ids
                    if conv_len <= remaining and conv_len > best_len:
                        best_idx = i
                        best_len = conv_len

                if best_idx >= 0:
                    # Found a conversation that fits - use it entirely
                    conv_ids, conv_mask = conv_buffer.pop(best_idx)
                    row_ids.extend(conv_ids)
                    row_mask.extend(conv_mask)
                    consumed += ddp_world_size  # Track actual consumption
                else:
                    # No conversation fits - pad the remainder instead of cropping
                    # This ensures we never discard any tokens
                    # Padding tokens have mask=0 (not supervised)
                    row_ids.extend([bos_token] * remaining)
                    row_mask.extend([0] * remaining)
                    break  # Row is now full (with padding)

            rows.append((row_ids[:row_capacity], row_mask[:row_capacity]))

        # Update progress tracking (based on consumed, not cursor, to account for buffering)
        # Note: step-based stopping (num_iterations > 0) is handled in the training loop, not here
        if split == "train":
            current_epoch = epoch
            if args.num_iterations <= 0:  # epoch mode only
                approx_progress = consumed / dataset_size
                # Trigger last_step when we've consumed enough (instead of when cursor wraps)
                if consumed >= dataset_size:
                    last_step = True

        # Build tensors from (row_ids, row_mask) tuples
        use_cuda = device_type == "cuda"
        batch_ids = torch.tensor([r[0] for r in rows], dtype=torch.long, pin_memory=use_cuda)
        batch_mask = torch.tensor([r[1] for r in rows], dtype=torch.long, pin_memory=use_cuda)
        inputs = batch_ids[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda)
        targets = batch_ids[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda)

        # Apply assistant-only loss masking using shifted mask for next-token prediction
        # mask=1 means supervised (assistant tokens), mask=0 means not supervised (user/padding)
        # The shift by 1 aligns the mask with the targets (we predict token[i+1] from token[i])
        shifted_mask = batch_mask[:, 1:].to(device=device, non_blocking=use_cuda)
        targets[shifted_mask == 0] = -1  # cross-entropy ignores targets with value -1

        yield inputs, targets

train_loader = sft_data_generator_bos_bestfit("train")
build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
progress = 0 # will go from 0 to 1 over the course of the epoch

# Learning rate scheduler with warmup
def get_lr_multiplier(progress, warmup_ratio, warmdown_ratio):
    """
    Three-phase LR schedule:
    1. Warmup: linear ramp from 0 to 1 over [0, warmup_ratio]
    2. Stable: hold at 1.0 over [warmup_ratio, 1 - warmdown_ratio]
    3. Decay: linear ramp from 1 to 0 over [1 - warmdown_ratio, 1.0]
    """
    if progress < warmup_ratio:
        # Linear warmup from 0 to 1
        return progress / warmup_ratio if warmup_ratio > 0 else 1.0
    elif progress < 1 - warmdown_ratio:
        # Stable phase
        return 1.0
    else:
        # Linear decay to 0 (clamped to prevent negative LR when progress > 1.0)
        return max(0.0, (1 - progress) / warmdown_ratio) if warmdown_ratio > 0 else 1.0

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Training loop
x, y = next(train_loader) # prefetch the very first batch of data
min_val_bpb = float("inf")
smooth_train_loss = 0 # EMA of training loss
ema_beta = 0.9 # EMA decay factor
total_training_time = 0 # total wall-clock time of training
step = 0
while True:
    flops_so_far = num_flops_per_token * args.total_batch_size * step

    # Synchronize last_step across all ranks to avoid hangs in the distributed setting
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or (args.eval_every > 0 and step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = args.eval_tokens // (args.device_batch_size * args.max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        # Don't treat the initial (step=0) evaluation as a "best" SFT checkpoint.
        # It's just the base model and can confuse downstream evals if it stays in best/.
        if step > 0 and val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
            # Save best checkpoint (lightweight - no optimizer state)
            if master_process and not args.dry_run:
                best_checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname, "best")
                os.makedirs(best_checkpoint_dir, exist_ok=True)

                if args.lora:
                    # LoRA: merge into a COPY (don't break ongoing training)
                    model_copy = copy.deepcopy(orig_model)
                    model_copy = merge_lora_weights(model_copy)
                    best_state_dict = model_copy.state_dict()
                    del model_copy  # free memory
                    gc.collect()
                    if device_type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    best_state_dict = orig_model.state_dict()

                save_checkpoint(
                    best_checkpoint_dir,
                    step,
                    best_state_dict,
                    None,  # No optimizer state (saves disk space)
                    {
                        "step": step,
                        "val_bpb": val_bpb,
                        "model_config": {
                            "sequence_len": args.max_seq_len,
                            "vocab_size": tokenizer.get_vocab_size(),
                            "n_layer": depth,
                            "n_head": model.config.n_head,
                            "n_kv_head": model.config.n_kv_head,
                            "n_embd": model.config.n_embd,
                            "window_pattern": model.config.window_pattern,
                        },
                        "user_config": user_config,
                    }
                )
                print0(f"Saved best checkpoint at step {step} (val_bpb={val_bpb:.4f})")
                # Rotate old best checkpoints, keep only --keep-best-k
                import glob
                best_models = glob.glob(os.path.join(best_checkpoint_dir, "model_*.pt"))
                best_models.sort(key=os.path.getmtime)
                if len(best_models) > args.keep_best_k:
                    for old_file in best_models[:-args.keep_best_k]:
                        old_step_str = old_file.split("_")[-1].split(".")[0]
                        os.remove(old_file)
                        meta_file = old_file.replace("model_", "meta_").replace(".pt", ".json")
                        if os.path.exists(meta_file):
                            os.remove(meta_file)
                        print0(f"Rotated old best checkpoint: step {old_step_str}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        })
        if args.lora and device_type == "cuda":
            torch.cuda.empty_cache()
        model.train()

    # save checkpoint at the end of the run (only on master process)
    if master_process and last_step and not args.dry_run:
        checkpoint_dir = os.path.join(base_dir, "chatsft_checkpoints", output_dirname)
        os.makedirs(checkpoint_dir, exist_ok=True)  # ensure dir exists FIRST

        if args.lora:
            # Save adapter-only weights FIRST (before merge removes LoRALinear modules)
            # These are much smaller (~4MB vs ~2GB) and useful for loading on different base models
            adapter_path = os.path.join(checkpoint_dir, f"lora_adapter_{step:06d}.pt")
            torch.save(get_lora_state_dict(orig_model), adapter_path)
            print0(f"Saved LoRA adapter weights to {adapter_path}")

            # Then merge for full model checkpoint (for inference compatibility)
            print0("Merging LoRA weights into base model for checkpoint...")
            orig_model = merge_lora_weights(orig_model)
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(),
            None if args.no_save_optimizer else optimizer.state_dict(),
            {
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": {
                    "sequence_len": args.max_seq_len,
                    "vocab_size": tokenizer.get_vocab_size(),
                    "n_layer": depth,
                    "n_head": model.config.n_head,
                    "n_kv_head": model.config.n_kv_head,
                    "n_embd": model.config.n_embd,
                    "window_pattern": model.config.window_pattern,
                },
                "user_config": user_config, # inputs to the training script
            }
        )

    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    did_backward = False
    for micro_step in range(grad_accum_steps):
        if (y != -1).sum() == 0:
            x, y = next(train_loader)
            if args.num_iterations <= 0:  # epoch mode only
                progress = max(progress, approx_progress) # only increase progress monotonically
            continue
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach() # for logging
        if not loss.requires_grad:
            x, y = next(train_loader)
            if args.num_iterations <= 0:  # epoch mode only
                progress = max(progress, approx_progress) # only increase progress monotonically
            continue
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        did_backward = True
        x, y = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
        if args.num_iterations <= 0:  # epoch mode only
            progress = max(progress, approx_progress) # only increase progress monotonically
    # For step-based mode, calculate progress based on actual optimizer steps
    if args.num_iterations > 0:
        progress = min(1.0, (step + 1) / args.num_iterations)  # +1 to avoid LR=0 on step 0
    # step the optimizer
    lrm = get_lr_multiplier(progress, args.warmup_ratio, args.warmdown_ratio)
    muon_momentum = get_muon_momentum(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
    if did_backward:
        # gradient clipping (on orig_model, not the compiled wrapper)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
        optimizer.step()
    else:
        print0(f"Warning: step {step + 1} had no valid targets; skipping optimizer.step()")
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # State
    step += 1

    # Step-based stopping condition (after step increment so we complete num_iterations steps)
    if args.num_iterations > 0 and step >= args.num_iterations:
        last_step = True

    # logging
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item() # EMA the training loss
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1)) # debias the EMA
    pct_done = 100 * progress
    tok_per_sec = int(args.total_batch_size / dt)
    flops_per_sec = num_flops_per_token * args.total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | total time: {total_training_time/60:.2f}m")
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_smooth_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
            "train/epoch": current_epoch,
        })

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
if not args.dry_run:
    from nanochat.report import get_report
    get_report().log(section="SFT", data=[
        user_config, # CLI args
        { # stats about the training setup
            "Number of iterations": step,
            "DDP world size": ddp_world_size,
        },
        { # stats about training outcomes
            "Minimum validation bpb": min_val_bpb,
        }
    ])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
