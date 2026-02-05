"""
LoRA (Low-Rank Adaptation) implementation for nanochat GPT model.

LoRA freezes all pretrained weights and injects trainable low-rank matrices
into attention layers. This prevents catastrophic forgetting during fine-tuning.

Reference: https://arxiv.org/abs/2106.09685
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    A Linear layer with LoRA (Low-Rank Adaptation) applied.

    The output is: original_output + (x @ A @ B) * scaling
    where A and B are low-rank matrices and scaling = alpha / rank.

    B is initialized to zeros so LoRA contribution starts at zero.
    """

    def __init__(self, original_linear: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = original_linear.in_features
        out_features = original_linear.out_features
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype

        # LoRA matrices: A projects down, B projects up
        # A: (in_features, rank) - Kaiming uniform init
        # B: (rank, out_features) - zeros init (so LoRA starts as identity)
        self.lora_A = nn.Parameter(torch.empty(in_features, rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device, dtype=dtype))

        # Initialize A with Kaiming uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Freeze the original weights
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

    def forward(self, x):
        # Original forward pass (frozen)
        result = self.original_linear(x)

        # LoRA forward pass (trainable)
        # x @ A @ B gives shape (batch, seq, out_features)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        result = result + lora_out * self.scaling

        return result

    def merge_weights(self):
        """Merge LoRA weights into original linear for efficient inference."""
        with torch.no_grad():
            # W_merged = W_original + (A @ B) * scaling
            delta = (self.lora_A @ self.lora_B) * self.scaling
            self.original_linear.weight.add_(delta.T)  # Linear stores weight as (out, in)
        return self.original_linear


def apply_lora_to_model(model, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0,
                        target_modules: list = None):
    """
    Apply LoRA to specified modules in the model.

    Args:
        model: The GPT model to modify
        rank: LoRA rank (higher = more capacity, more params)
        alpha: LoRA scaling factor (typically 2x rank)
        dropout: Dropout probability for LoRA layers
        target_modules: List of module name patterns to apply LoRA to.
                       Default: ['c_q', 'c_k', 'c_v', 'c_proj'] (attention layers)

    Returns:
        model: Modified model with LoRA applied
        lora_params: List of LoRA parameters for the optimizer
    """
    if target_modules is None:
        # Default: apply to attention projections only (most effective for fine-tuning)
        target_modules = ['c_q', 'c_k', 'c_v', 'c_proj']

    lora_params = []
    replaced_count = 0

    # First, freeze ALL parameters
    for param in model.parameters():
        param.requires_grad = False

    # Then, apply LoRA to target modules
    for name, module in model.named_modules():
        # Check if this module should have LoRA applied
        module_name = name.split('.')[-1]
        if module_name in target_modules and isinstance(module, nn.Linear):
            # Get the parent module to replace the child
            parent_name = '.'.join(name.split('.')[:-1])
            parent = model.get_submodule(parent_name) if parent_name else model

            # Create LoRA-wrapped linear
            lora_linear = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)

            # Replace in parent
            setattr(parent, module_name, lora_linear)

            # Collect LoRA params
            lora_params.extend([lora_linear.lora_A, lora_linear.lora_B])
            replaced_count += 1

    return model, lora_params, replaced_count


def count_lora_params(lora_params):
    """Count total trainable LoRA parameters."""
    return sum(p.numel() for p in lora_params)


def count_total_params(model):
    """Count total model parameters."""
    return sum(p.numel() for p in model.parameters())


def get_lora_state_dict(model):
    """Extract only LoRA weights from model state dict."""
    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state_dict[f"{name}.lora_A"] = module.lora_A.data
            state_dict[f"{name}.lora_B"] = module.lora_B.data
    return state_dict


def load_lora_state_dict(model, state_dict):
    """Load LoRA weights into model."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            if f"{name}.lora_A" in state_dict:
                module.lora_A.data = state_dict[f"{name}.lora_A"]
            if f"{name}.lora_B" in state_dict:
                module.lora_B.data = state_dict[f"{name}.lora_B"]


def merge_lora_weights(model):
    """
    Merge LoRA weights into the base model and replace LoRALinear with nn.Linear.

    This converts a LoRA-adapted model back to a standard model for inference,
    with LoRA adaptations baked into the weights.

    Args:
        model: Model with LoRALinear layers

    Returns:
        model: Model with merged weights (LoRALinear replaced with nn.Linear)
    """
    # Find all LoRALinear modules and their parent paths
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Get parent module path and child name
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                parent_name, child_name = parts
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                child_name = name

            # Merge weights and get the merged linear layer
            merged_linear = module.merge_weights()
            replacements.append((parent, child_name, merged_linear))

    # Apply replacements
    for parent, child_name, merged_linear in replacements:
        setattr(parent, child_name, merged_linear)

    return model
