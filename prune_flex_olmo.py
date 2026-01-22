import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/Flex-math-2x7B-1T"
DEVICE = "cuda:1"
importance_scores = load_file(
    "importance_scores_tensors/flex_math_importance_scores.safetensors"
)
# ORIGINAL MODEL ATTRIBUTES
HIDDEN_SIZE = 4096


def get_mlp_keep_indices(mlp_width):
    mlp_importance = importance_scores["mlp"]
    keep_indices = {}
    for layer_idx in range(mlp_importance.shape[0]):
        layer_scores = mlp_importance[layer_idx]
        top_k = layer_scores.argsort(descending=True)[:mlp_width]
        keep_indices[layer_idx] = top_k.sort().values
    return keep_indices


def apply_mlp_pruning(model, mlp_width):
    keep_indices = get_mlp_keep_indices(mlp_width)
    for new_idx, layer in enumerate(model.model.layers):
        idx = keep_indices[new_idx].to(layer.mlp.gate_proj.weight.device)
        layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data[idx, :]
        layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[idx, :]
        layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data[:, idx]

    model.config.intermediate_size = mlp_width
    return model


def prune_model(mlp_width):
    print(f"Pruning to: MLP={mlp_width}")

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)

    # Width pruning with correct layer mapping
    model = apply_mlp_pruning(model, mlp_width)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e9


if __name__ == "__main__":
    expert_mlp_width = 8192
    model = prune_model(expert_mlp_width)  # mlp_width, num_heads, num_layers
    model = model.to(DEVICE)

    print(f"Parameters: {count_params(model):.2f}B")

    # Sanity check
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model(**inputs)
    print(f"Forward pass works! Logits shape: {outputs.logits.shape}")

    message = [{"role": "user", "content": "What are the main ingredients in dashi?"}]
    inputs = tokenizer.apply_chat_template(
        message, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        response = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    print(
        f"Generation Check:\n {tokenizer.decode(response[0], skip_special_tokens=True)}"
    )

    # Save
    save_path = f"pruned_models/pruned_flex_math_{expert_mlp_width}"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved to {save_path}")
