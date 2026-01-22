"""Adapt FlexOlmo to use megablocks instead of for loops over experts (https://github.com/huggingface/transformers/blob/a7f29523361b2cc12e51c1f5133d95f122f6f45c/src/transformers/models/flex_olmo/modeling_flex_olmo.py#L296)"""

import math

import bitsandbytes as bnb
import torch
from megablocks.layers.silu import silu

from megablocks_core import moe_forward


def dequantize_if_needed(linear_layer) -> torch.Tensor:
    """Extract weight from a Linear layer, dequantizing if 4-bit."""
    if hasattr(linear_layer, "weight") and hasattr(linear_layer.weight, "quant_state"):
        # 4-bit quantized weight
        return bnb.functional.dequantize_4bit(
            linear_layer.weight.data, linear_layer.weight.quant_state
        )
    else:
        return linear_layer.weight.data


class FlexOlmoAdapter:
    def __init__(self, config):
        self.activation = silu  # silu + gated = swiglu
        self.num_experts = config.num_experts
        self.num_active_experts = config.num_experts_per_tok
        self.sort_end_bit = max(int(math.ceil(math.log2(self.num_experts))), 1)
        self.transpose_sort_end_bit = self.sort_end_bit

        # for uniform experts (FlexOlmo), all same size
        ffn = config.intermediate_size
        self.expert_widths = [ffn] * self.num_experts
        self.total_expert_width = ffn * self.num_experts

        # precompute block info
        block_size = 128
        self.block_size = block_size
        self.expert_size_blocks = torch.tensor(
            [ffn // block_size] * self.num_experts, dtype=torch.int32
        )
        self.expert_block_offsets = torch.tensor(
            [i * ffn // block_size for i in range(self.num_experts + 1)],
            dtype=torch.int32,
        )

        # Precompute offsets for 2x2 case (avoids .tolist() every forward)
        if self.num_experts == 2 and self.num_active_experts == 2:
            e0_end = ffn
            e1_start = ffn
            e1_end = 2 * ffn
            self.precomputed_offsets_2x2 = (e0_end, e1_start, e1_end)
        else:
            self.precomputed_offsets_2x2 = None

    def transform_weights(self, experts_module) -> dict[str, torch.Tensor]:
        # experts_module is a ModuleList of FlexOlmoMLP modules
        # Each has gate_proj, up_proj, down_proj with shapes:
        #   gate_proj.weight: (ffn, hidden)
        #   up_proj.weight: (ffn, hidden)
        #   down_proj.weight: (hidden, ffn)
        # Handles both regular and 4-bit quantized weights
        #
        # Memory-efficient: use cat + transpose instead of stack + einops
        # to avoid creating intermediate tensors

        # gate/up: (ffn, hidden) per expert -> cat to (e*ffn, hidden) -> transpose to (hidden, e*ffn)
        w1_gate = (
            torch.cat(
                [dequantize_if_needed(e.gate_proj) for e in experts_module], dim=0
            )
            .t()
            .contiguous()
        )
        w1_up = (
            torch.cat([dequantize_if_needed(e.up_proj) for e in experts_module], dim=0)
            .t()
            .contiguous()
        )
        # down: (hidden, ffn) per expert -> cat to (e*hidden, ffn)... wait, need (e*ffn, hidden)
        # down_proj.weight is (hidden, ffn), we need (e*ffn, hidden)
        # So transpose each first, then cat
        w2 = torch.cat(
            [dequantize_if_needed(e.down_proj).t() for e in experts_module], dim=0
        ).contiguous()

        return {"w1_gate": w1_gate, "w1_up": w1_up, "w2": w2}

    def forward(self, hidden_states, top_k_index, top_k_weights, transformed_weights):
        return moe_forward(
            hidden_states=hidden_states,
            top_k_index=top_k_index,
            top_k_weights=top_k_weights,
            w1_up=transformed_weights["w1_up"],
            w1=transformed_weights["w1_gate"],
            w2=transformed_weights["w2"],
            activation_fn=self.activation,
            num_experts=self.num_experts,
            num_active_experts=self.num_active_experts,
            sort_end_bit=self.sort_end_bit,
            transpose_sort_end_bit=self.transpose_sort_end_bit,
            block_size=self.block_size,
            expert_size_blocks=self.expert_size_blocks,
            expert_block_offsets=self.expert_block_offsets,
            total_expert_width=self.total_expert_width,
            precomputed_offsets_2x2=self.precomputed_offsets_2x2,
        )
