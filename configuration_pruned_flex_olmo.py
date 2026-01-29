"""Configuration for pruned FlexOlmo models with variable-width expert 1."""

from transformers import FlexOlmoConfig


class PrunedFlexOlmoConfig(FlexOlmoConfig):
    """Config for FlexOlmo with a pruned expert 1.

    Extends FlexOlmoConfig with expert_1_intermediate_size to specify
    the width of the pruned expert.
    """

    model_type = "pruned_flex_olmo"

    def __init__(self, expert_1_intermediate_size: int = None, **kwargs):
        super().__init__(**kwargs)
        # expert_1_intermediate_size: width of pruned expert 1
        # If None, falls back to intermediate_size (no pruning)
        self.expert_1_intermediate_size = (
            expert_1_intermediate_size or self.intermediate_size
        )
