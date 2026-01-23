"""Prune FlexOlmo expert 1 and split experts for memory-efficient training.

Creates a model where:
- Expert 0: frozen at full size (11008), no gradients/optimizer states
- Expert 1: pruned to target width, trainable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.activations import ACT2FN

MODEL = "allenai/Flex-math-2x7B-1T"
DEVICE = "cuda:1"
EXPERT_0_SIZE = 11008  # full size, frozen
HIDDEN_SIZE = 4096


class SplitFlexOlmoExperts(nn.Module):
    """FlexOlmoExperts with split parameters - expert 0 frozen, expert 1 trainable.

    This replaces the fused (num_experts, ...) tensors with separate parameters,
    allowing expert 0 to be frozen (requires_grad=False) while expert 1 is trainable.
    """

    def __init__(self, expert_0_gate_up, expert_0_down,
                 expert_1_gate_up, expert_1_down, act_fn):
        super().__init__()
        self.num_experts = 2
        self.act_fn = act_fn

        # Expert 0 - frozen (full size)
        self.expert_0_gate_up = nn.Parameter(expert_0_gate_up, requires_grad=False)
        self.expert_0_down = nn.Parameter(expert_0_down, requires_grad=False)

        # Expert 1 - trainable (possibly pruned)
        self.expert_1_gate_up = nn.Parameter(expert_1_gate_up, requires_grad=True)
        self.expert_1_down = nn.Parameter(expert_1_down, requires_grad=True)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx_val = expert_idx[0].item()
            if expert_idx_val == self.num_experts:
                continue

            top_k_pos, token_idx = torch.where(expert_mask[expert_idx_val])
            current_state = hidden_states[token_idx]

            # Select appropriate expert weights
            if expert_idx_val == 0:
                gate_up_weight = self.expert_0_gate_up
                down_weight = self.expert_0_down
            else:
                gate_up_weight = self.expert_1_gate_up
                down_weight = self.expert_1_down

            gate, up = F.linear(current_state, gate_up_weight).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, down_weight)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


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
    """Replace fused FlexOlmoExperts with SplitFlexOlmoExperts, pruning expert 1."""
    keep_indices = get_mlp_keep_indices(importance_scores, expert_1_width)

    for layer_idx, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        idx = keep_indices[layer_idx].to(experts.gate_up_proj.device)

        # Expert 0: keep full size (frozen)
        expert_0_gate_up = experts.gate_up_proj[0].clone()  # (2*11008, 4096)
        expert_0_down = experts.down_proj[0].clone()  # (4096, 11008)

        # Expert 1: prune to target width
        # gate_up_proj[1] is (2*intermediate_size, hidden_size)
        # We need to select the same neurons from both gate and up halves
        full_gate_up = experts.gate_up_proj[1]  # (22016, 4096)
        intermediate_size = full_gate_up.shape[0] // 2
        gate_part = full_gate_up[:intermediate_size, :]  # (11008, 4096)
        up_part = full_gate_up[intermediate_size:, :]  # (11008, 4096)

        pruned_gate = gate_part[idx, :]  # (expert_1_width, 4096)
        pruned_up = up_part[idx, :]  # (expert_1_width, 4096)
        expert_1_gate_up = torch.cat([pruned_gate, pruned_up], dim=0)  # (2*expert_1_width, 4096)

        # down_proj[1] is (hidden_size, intermediate_size)
        expert_1_down = experts.down_proj[1, :, idx].clone()  # (4096, expert_1_width)

        # Create split module
        split_experts = SplitFlexOlmoExperts(
            expert_0_gate_up=expert_0_gate_up,
            expert_0_down=expert_0_down,
            expert_1_gate_up=expert_1_gate_up,
            expert_1_down=expert_1_down,
            act_fn=experts.act_fn,
        )

        # Replace the module
        layer.mlp.experts = split_experts

    # Update config to reflect expert 1's new size (for reference)
    model.config.expert_1_intermediate_size = expert_1_width
    return model


def count_params(model, trainable_only=False):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
    return sum(p.numel() for p in model.parameters()) / 1e9


def prune_model(expert_1_width, importance_path):
    """Load model, split experts, and prune expert 1."""
    print(f"Loading importance scores from {importance_path}")
    importance_scores = load_file(importance_path)

    print(f"Loading model {MODEL}")
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

    print(f"Splitting experts and pruning expert 1 to width {expert_1_width}")
    model = split_and_prune_experts(model, importance_scores, expert_1_width)

    return model


if __name__ == "__main__":
    expert_1_width = 2048
    importance_path = "importance_scores_tensors/importance_scores_math.safetensors"

    model = prune_model(expert_1_width, importance_path)
    model = model.to(DEVICE)

    print(f"Total parameters: {count_params(model):.2f}B")
    print(f"Trainable parameters: {count_params(model, trainable_only=True):.2f}B")

    # Sanity check
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
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
    save_path = f"pruned_models/split_flex_math_{expert_1_width}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved to {save_path}")
