import types
from itertools import chain
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

"""importance analysis for flexolmo math expert 1. we try one with math-specific data, and one with general purpose data."""

# old api, inductor hasn't migrated yet
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL = "allenai/Flex-math-2x7B-1T"
EXPERT_IDX = 1  # expert 0 is base/public, expert 1 is math-specialized


def get_device_map():
    """Hardcoded device map for even distribution across 3 GPUs."""
    device_map = {"model.embed_tokens": 0}
    # GPU 0: layers 0-5 (6 layers) + embed
    for i in range(6):
        device_map[f"model.layers.{i}"] = 0
    # GPU 1: layers 6-20 (15 layers)
    for i in range(6, 21):
        device_map[f"model.layers.{i}"] = 1
    # GPU 2: layers 21-31 (11 layers) + norm + lm_head
    for i in range(21, 32):
        device_map[f"model.layers.{i}"] = 2
    device_map["model.norm"] = 2
    device_map["lm_head"] = 2
    return device_map


DATASETS = {
    "math": {
        "data_files": [
            "data/math/gsm8k/**/*.jsonl",
            "data/math/metamath-owmfilter/**/*.jsonl",
            "data/math/tulu_math/**/*.jsonl",
        ],
        "cache_path": "cached_analysis_tokens_math.safetensors",
        "output_path": "importance_scores_tensors/importance_scores_math.safetensors",
    },
    "general": {
        "data_files": [
            "data/flan/**/*.jsonl",
            "data/wiki/**/*.jsonl",
        ],
        "cache_path": "cached_analysis_tokens_general.safetensors",
        "output_path": "importance_scores_tensors/importance_scores_general.safetensors",
    },
}

sequence_length = 4096  # same seq_len as pretraining
num_samples = 1024  # specified in the nvidia paper
batch_size = 7  # adjust for hardware


def get_dataset(data_files):
    ds = load_dataset(
        "allenai/dolmino-mix-1124",
        data_files=data_files,
        split="train",
        streaming=True,
    ).shuffle(buffer_size=50_000, seed=1223)
    return list(ds.take(10_000))  # 10k docs should be ~10M tokens


def tokenize_dataset(dataset):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    texts = [sample["text"] for sample in dataset]
    tokenized = tokenizer(
        texts, padding=False, truncation=False, add_special_tokens=False
    )
    return list(chain.from_iterable(tokenized["input_ids"]))


def make_capturing_forward(layer_idx, importance_dict, target_expert=EXPERT_IDX):
    """Create a patched FlexOlmoExperts.forward that captures target expert's activations.

    FlexOlmoExperts uses fused weights (gate_up_proj, down_proj) with shape [num_experts, ...],
    so we can't use hooks on individual experts. Instead we patch the forward method.
    """

    def patched_forward(self, hidden_states, top_k_index, top_k_weights):
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
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx_val]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up

            # Capture target expert's pre-down_proj activations
            if expert_idx_val == target_expert:
                x = current_hidden_states.detach().abs().mean(dim=0).pow(2).cpu()
                name = f"layer_{layer_idx}.mlp"
                if name in importance_dict:
                    importance_dict[name] += x
                else:
                    importance_dict[name] = x

            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx_val])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    return patched_forward


def load_or_cache_tokens(config):
    """Load tokens from cache or tokenize dataset and cache."""
    cache_path = config["cache_path"]
    if Path(cache_path).exists():
        return load_file(cache_path)["tokens"]

    dataset = tokenize_dataset(get_dataset(config["data_files"]))
    total_tokens = sequence_length * num_samples
    all_tokens = torch.tensor(dataset[:total_tokens]).view(num_samples, sequence_length)
    save_file({"tokens": all_tokens}, cache_path)
    return all_tokens


def run_analysis(model, config, all_tokens, importance_dict):
    """Run importance analysis on a dataset, storing results in importance_dict."""
    num_hidden_layers = model.config.num_hidden_layers
    intermediate_size = model.config.intermediate_size

    # Patch experts forward to capture activations
    original_forwards = []
    for i, layer in enumerate(model.model.layers):
        experts = layer.mlp.experts
        original_forwards.append(experts.forward)
        experts.forward = types.MethodType(
            make_capturing_forward(i, importance_dict), experts
        )

    # Run forward passes (input goes to cuda:0 where embed_tokens lives)
    for batch_idx in tqdm(
        range(0, num_samples, batch_size), desc=config["output_path"]
    ):
        batch = all_tokens[batch_idx : batch_idx + batch_size].to("cuda:0")
        with torch.inference_mode():
            model(batch, use_cache=False)

    # Restore original forwards
    for i, layer in enumerate(model.model.layers):
        layer.mlp.experts.forward = original_forwards[i]

    # Finish L2 norm (sqrt) and assemble into tensor (all on CPU)
    mlp_importance = torch.zeros(num_hidden_layers, intermediate_size)
    for key, value in importance_dict.items():
        layer_idx = int(key.split("_")[1].split(".")[0])
        mlp_importance[layer_idx] = value.pow(0.5)

    # Save results
    scores = {"mlp": mlp_importance}
    Path(config["output_path"]).parent.mkdir(parents=True, exist_ok=True)
    save_file(scores, config["output_path"])
    print(f"Saved importance scores to {config['output_path']}")


def main():
    # Load model once, reuse for all datasets
    device_map = get_device_map()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
    )

    for dataset_name, config in DATASETS.items():
        print(f"\n{'=' * 60}")
        print(f"Running analysis for: {dataset_name}")
        print(f"{'=' * 60}")

        all_tokens = load_or_cache_tokens(config)
        importance_dict = {}  # fresh dict for each dataset
        run_analysis(model, config, all_tokens, importance_dict)

    # Cleanup
    del model
    torch.cuda.empty_cache()
    print("\nDone!")


if __name__ == "__main__":
    main()
