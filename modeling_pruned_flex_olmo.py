"""Pruned FlexOlmo model with variable-width expert 1.

This module provides a HuggingFace-compatible model that can be loaded with:
    AutoModelForCausalLM.from_pretrained("hbfreed/flex-math-8192", trust_remote_code=True)
"""

import torch
import torch.nn as nn
from transformers import FlexOlmoForCausalLM
from transformers.models.flex_olmo.modeling_flex_olmo import FlexOlmoMLP

from .configuration_pruned_flex_olmo import PrunedFlexOlmoConfig


class PrunedFlexOlmoMLP(nn.Module):
    """Pruned MLP with same interface as FlexOlmoMLP but variable width."""

    def __init__(self, intermediate_size: int, hidden_size: int, act_fn, dtype=torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)
        self.act_fn = act_fn

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class PrunedFlexOlmoForCausalLM(FlexOlmoForCausalLM):
    """FlexOlmo with pruned expert 1 for variable-width MoE.

    Expert 0 remains at full intermediate_size, while expert 1 is pruned
    to expert_1_intermediate_size specified in the config.
    """

    config_class = PrunedFlexOlmoConfig

    def __init__(self, config: PrunedFlexOlmoConfig):
        # Initialize parent with full architecture
        super().__init__(config)

        # Replace expert 1 in each layer with pruned version
        expert_1_width = config.expert_1_intermediate_size
        hidden_size = config.hidden_size

        for layer in self.model.layers:
            # Get activation function from existing expert
            act_fn = layer.mlp.experts[1].act_fn

            # Replace expert 1 with pruned version
            layer.mlp.experts[1] = PrunedFlexOlmoMLP(
                intermediate_size=expert_1_width,
                hidden_size=hidden_size,
                act_fn=act_fn,
                dtype=self.dtype,
            )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Load pruned model, handling both local and hub paths."""
        # Let parent handle the loading - it will use our config_class
        # and __init__ which sets up the pruned architecture
        return super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
