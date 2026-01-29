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
    """Save pruned model in HuggingFace-compatible format.

    Saves:
    - model.safetensors: Full model state dict
    - config.json: Model config with auto_map for trust_remote_code loading
    - modeling/configuration files: Custom classes for HF loading
    """
    import json
    import shutil
    from pathlib import Path
    from safetensors.torch import save_file

    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Save full state dict
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    save_file(state_dict, f"{save_path}/model.safetensors")

    # Build config with auto_map for trust_remote_code
    config_dict = model.config.to_dict()
    config_dict["model_type"] = "pruned_flex_olmo"
    config_dict["expert_1_intermediate_size"] = expert_1_width
    config_dict["auto_map"] = {
        "AutoConfig": "configuration_pruned_flex_olmo.PrunedFlexOlmoConfig",
        "AutoModelForCausalLM": "modeling_pruned_flex_olmo.PrunedFlexOlmoForCausalLM",
    }
    # Add metadata for reference
    config_dict["_pruning_metadata"] = {
        "base_model": MODEL,
        "importance_type": importance_type,
    }

    with open(f"{save_path}/config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Copy the modeling and configuration files
    repo_root = Path(__file__).parent
    for filename in ["modeling_pruned_flex_olmo.py", "configuration_pruned_flex_olmo.py"]:
        src = repo_root / filename
        if src.exists():
            shutil.copy(src, f"{save_path}/{filename}")

    print(f"Saved pruned model to {save_path}")


def load_pruned_model(save_path, device="cuda"):
    """Load a pruned model using HuggingFace's AutoModelForCausalLM.

    Works with both local paths and HuggingFace Hub repo IDs.
    """
    import os

    # If local path doesn't exist, derive hub repo ID
    if not os.path.exists(save_path):
        model_name = os.path.basename(save_path)  # flex_math_2048
        save_path = f"hbfreed/{model_name.replace('_', '-')}"  # hbfreed/flex-math-2048
        print(f"Loading from HuggingFace Hub: {save_path}")

    model = AutoModelForCausalLM.from_pretrained(
        save_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Freeze all params (expert 1 will be unfrozen by training script if needed)
    for param in model.parameters():
        param.requires_grad = False

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
