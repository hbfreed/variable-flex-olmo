"""Prune FlexOlmo expert 1 and split experts for memory-efficient training.

Creates a model where:
- Expert 0: frozen at full size (11008), no gradients/optimizer states
- Expert 1: pruned to target width, trainable
"""

import argparse

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/Flex-math-2x7B-1T"
TOKENIZER_NAME = "allenai/dolma2-tokenizer"
WIDTHS = [8192, 5504, 2048]
IMPORTANCE_PATHS = {
    "math": "importance_scores_tensors/importance_scores_math.safetensors",
    "general": "importance_scores_tensors/importance_scores_general.safetensors",
}
DEVICE = "cuda:1"
EXPERT_0_SIZE = 11008  # full size, frozen
HIDDEN_SIZE = 4096


class PrunedFlexOlmoMLP(nn.Module):
    """Pruned MLP with same interface as FlexOlmoMLP."""

    def __init__(self, gate_proj_weight, up_proj_weight, down_proj_weight, act_fn):
        super().__init__()
        self.gate_proj = nn.Linear(gate_proj_weight.shape[1], gate_proj_weight.shape[0], bias=False)
        self.up_proj = nn.Linear(up_proj_weight.shape[1], up_proj_weight.shape[0], bias=False)
        self.down_proj = nn.Linear(down_proj_weight.shape[1], down_proj_weight.shape[0], bias=False)

        self.gate_proj.weight = nn.Parameter(gate_proj_weight)
        self.up_proj.weight = nn.Parameter(up_proj_weight)
        self.down_proj.weight = nn.Parameter(down_proj_weight)
        self.act_fn = act_fn

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def get_mlp_keep_indices(importance_scores, mlp_width):
    """Get indices of top-k important neurons per layer."""
    mlp_importance = importance_scores["mlp"]
    keep_indices = {}
    for layer_idx in range(mlp_importance.shape[0]):
        layer_scores = mlp_importance[layer_idx]
        top_k = layer_scores.argsort(descending=True)[:mlp_width]
        keep_indices[layer_idx] = top_k.sort().values
    return keep_indices


def split_and_prune_experts(model, importance_scores, expert_1_width):
    """Freeze everything except expert 1, which is pruned to target width."""
    keep_indices = get_mlp_keep_indices(importance_scores, expert_1_width)

    # Freeze entire model first
    for param in model.parameters():
        param.requires_grad = False

    for layer_idx, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts  # ModuleList of FlexOlmoMLP
        expert_1 = experts[1]

        idx = keep_indices[layer_idx].to(expert_1.gate_proj.weight.device)

        # Expert 1: prune to target width
        # gate_proj.weight is (intermediate_size, hidden_size)
        # up_proj.weight is (intermediate_size, hidden_size)
        # down_proj.weight is (hidden_size, intermediate_size)
        pruned_gate = expert_1.gate_proj.weight[idx, :].clone()
        pruned_up = expert_1.up_proj.weight[idx, :].clone()
        pruned_down = expert_1.down_proj.weight[:, idx].clone()

        # Replace expert 1 with pruned version (trainable by default)
        pruned_expert_1 = PrunedFlexOlmoMLP(
            gate_proj_weight=pruned_gate,
            up_proj_weight=pruned_up,
            down_proj_weight=pruned_down,
            act_fn=expert_1.act_fn,
        )
        experts[1] = pruned_expert_1

    # Update config to reflect expert 1's new size (for reference)
    model.config.expert_1_intermediate_size = expert_1_width
    return model


def count_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    return sum(p.numel() for p in model.parameters()) / 1e9


def save_pruned_model(model, save_path, importance_type, expert_1_width):
    """Save pruned model in a loadable format.

    Saves:
    - Full model state dict (for complete restore)
    - Metadata (importance type, width) for reconstruction
    """
    import json
    from pathlib import Path
    from safetensors.torch import save_file

    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Save full state dict
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, f"{save_path}/model.safetensors")

    # Save metadata
    metadata = {
        "base_model": MODEL,
        "importance_type": importance_type,
        "expert_1_width": expert_1_width,
    }
    with open(f"{save_path}/pruning_config.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved pruned model to {save_path}")


def load_pruned_model(save_path, device="cuda"):
    """Load a pruned model by creating matching architecture then loading weights.

    If local path doesn't exist, attempts to download from HuggingFace Hub.
    """
    import json
    import os
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download

    # If local path doesn't exist, try downloading from hub
    if not os.path.exists(save_path):
        # Derive hub repo from path: pruned_models/flex_math_2048 -> hbfreed/flex-math-2048
        model_name = os.path.basename(save_path)  # flex_math_2048
        repo_id = f"hbfreed/{model_name.replace('_', '-')}"  # hbfreed/flex-math-2048
        print(f"Local path {save_path} not found, downloading from {repo_id}...")
        save_path = snapshot_download(repo_id, local_dir=save_path)

    # Load metadata
    with open(f"{save_path}/pruning_config.json") as f:
        metadata = json.load(f)

    expert_1_width = metadata["expert_1_width"]
    print(f"Loading pruned model: width={expert_1_width}")

    # Load state dict first to get shapes
    state_dict = load_file(f"{save_path}/model.safetensors")

    # Load base model
    print(f"Loading base model {MODEL}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)

    # Freeze all params
    for param in model.parameters():
        param.requires_grad = False

    # Replace expert 1 in each layer with correctly-sized PrunedFlexOlmoMLP
    for layer_idx, layer in enumerate(model.model.layers):
        expert_1 = layer.mlp.experts[1]

        # Create pruned expert with correct shapes (will be overwritten by state_dict)
        pruned_expert_1 = PrunedFlexOlmoMLP(
            gate_proj_weight=torch.zeros(expert_1_width, HIDDEN_SIZE, dtype=torch.bfloat16),
            up_proj_weight=torch.zeros(expert_1_width, HIDDEN_SIZE, dtype=torch.bfloat16),
            down_proj_weight=torch.zeros(HIDDEN_SIZE, expert_1_width, dtype=torch.bfloat16),
            act_fn=expert_1.act_fn,
        )
        layer.mlp.experts[1] = pruned_expert_1

    # Load saved weights
    model.load_state_dict(state_dict)

    return model.to(device)


def prune_model(expert_1_width, importance_path):
    """Load model, split experts, and prune expert 1."""
    print(f"Loading importance scores from {importance_path}")
    importance_scores = load_file(importance_path)

    print(f"Loading model {MODEL}")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)

    print(f"Splitting experts and pruning expert 1 to width {expert_1_width}")
    model = split_and_prune_experts(model, importance_scores, expert_1_width)

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Prune FlexOlmo expert 1")
    parser.add_argument(
        "--importance", "-i",
        choices=["math", "general"],
        required=True,
        help="Which importance scores to use for pruning"
    )
    parser.add_argument(
        "--width", "-w",
        type=int,
        choices=WIDTHS,
        required=True,
        help=f"Target width for expert 1. Choices: {WIDTHS}"
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip the generation sanity check"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    importance_path = IMPORTANCE_PATHS[args.importance]
    expert_1_width = args.width

    model = prune_model(expert_1_width, importance_path)
    model = model.to(DEVICE)

    print(f"Total parameters: {count_params(model):.2f}B")
    print(f"Trainable parameters: {count_params(model, trainable_only=True):.2f}B")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    if not args.skip_sanity_check:
        # Sanity check
        inputs = tokenizer("Hello, world!", return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            outputs = model(**inputs)
        print(f"Forward pass works! Logits shape: {outputs.logits.shape}")

        message = [{"role": "user", "content": "What is 15 + 27?"}]
        inputs = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.inference_mode():
            response = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        print(f"Generation check:\n{tokenizer.decode(response[0], skip_special_tokens=True)}")

    # Save
    save_path = f"pruned_models/flex_{args.importance}_{expert_1_width}"
    save_pruned_model(model, save_path, args.importance, expert_1_width)
