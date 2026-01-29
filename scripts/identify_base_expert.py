"""Identify which expert index matches the base public model across all FlexOlmo variants."""
import torch
from safetensors import safe_open
from pathlib import Path

BASE_MODEL = "allenai/Flex-public-7B-1T"

# All 2x7B variants (each has base + 1 specialized expert)
MOE_2X_MODELS = [
    ("allenai/Flex-math-2x7B-1T", "math"),
    ("allenai/Flex-code-2x7B-1T", "code"),
    ("allenai/Flex-creative-2x7B-1T", "creative"),
    ("allenai/Flex-news-2x7B-1T", "news"),
    ("allenai/Flex-pes2o-2x7B-1T", "pes2o"),
    ("allenai/Flex-reddit-2x7B-1T", "reddit"),
]

# 7x7B model (has all 7 experts)
MOE_7X_MODEL = "allenai/FlexOlmo-7x7B-1T"


def get_cache_path(model_name):
    """Get local HF cache path for model."""
    safe_name = model_name.replace("/", "--")
    return Path.home() / ".cache/huggingface/hub" / f"models--{safe_name}" / "snapshots"


def load_safetensors(model_path):
    """Load all safetensor shards into a single state dict."""
    state_dict = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def compare_to_2x_model(base_sd, moe_model):
    """Compare base model weights to a 2x7B MoE model (2 experts)."""
    try:
        moe_path = next(get_cache_path(moe_model).iterdir())
    except (StopIteration, FileNotFoundError):
        print(f"  SKIPPED: Model not found in cache")
        return None

    moe_sd = load_safetensors(moe_path)

    num_layers = 32
    expert_0_matches = 0
    expert_1_matches = 0

    for layer in range(num_layers):
        base_key = f"model.layers.{layer}.mlp.gate_proj.weight"
        expert_0_key = f"model.layers.{layer}.mlp.experts.0.gate_proj.weight"
        expert_1_key = f"model.layers.{layer}.mlp.experts.1.gate_proj.weight"

        base_w = base_sd[base_key]
        expert_0_w = moe_sd[expert_0_key]
        expert_1_w = moe_sd[expert_1_key]

        if torch.allclose(base_w, expert_0_w, atol=1e-6):
            expert_0_matches += 1
        if torch.allclose(base_w, expert_1_w, atol=1e-6):
            expert_1_matches += 1

    return expert_0_matches, expert_1_matches, moe_sd


def analyze_7x_model(base_sd, specialized_experts):
    """Analyze the 7x7B model to map all 7 expert indices.

    Args:
        base_sd: State dict from base model
        specialized_experts: Dict mapping domain name -> state dict of 2x model's expert 1
    """
    try:
        moe_path = next(get_cache_path(MOE_7X_MODEL).iterdir())
    except (StopIteration, FileNotFoundError):
        print(f"  SKIPPED: 7x7B model not found in cache")
        return None

    print(f"\nLoading 7x7B model from: {moe_path}")
    moe_7x_sd = load_safetensors(moe_path)
    print(f"7x7B model loaded ({len(moe_7x_sd)} tensors)")

    num_layers = 32
    num_experts = 7

    # Map each expert index to what it matches
    expert_mapping = {}

    for expert_idx in range(num_experts):
        print(f"\nChecking expert {expert_idx}...")

        # Check if this expert matches the base model
        base_matches = 0
        for layer in range(num_layers):
            base_key = f"model.layers.{layer}.mlp.gate_proj.weight"
            expert_key = f"model.layers.{layer}.mlp.experts.{expert_idx}.gate_proj.weight"

            if torch.allclose(base_sd[base_key], moe_7x_sd[expert_key], atol=1e-6):
                base_matches += 1

        print(f"  vs base: {base_matches}/32 layers match")

        if base_matches == 32:
            expert_mapping[expert_idx] = "public (base)"
            continue

        # Check if this expert matches any specialized expert
        best_match = None
        best_match_count = 0

        for domain, spec_sd in specialized_experts.items():
            spec_matches = 0
            for layer in range(num_layers):
                spec_key = f"model.layers.{layer}.mlp.experts.1.gate_proj.weight"
                expert_key = f"model.layers.{layer}.mlp.experts.{expert_idx}.gate_proj.weight"

                if torch.allclose(spec_sd[spec_key], moe_7x_sd[expert_key], atol=1e-6):
                    spec_matches += 1

            print(f"  vs {domain}: {spec_matches}/32 layers match")

            if spec_matches > best_match_count:
                best_match_count = spec_matches
                best_match = domain

            if spec_matches == 32:
                expert_mapping[expert_idx] = domain
                break

        if expert_idx not in expert_mapping:
            if best_match_count > 0:
                expert_mapping[expert_idx] = f"PARTIAL: {best_match} ({best_match_count}/32)"
            else:
                expert_mapping[expert_idx] = "UNKNOWN"

    return expert_mapping


def main():
    # Load base model once
    try:
        base_path = next(get_cache_path(BASE_MODEL).iterdir())
    except (StopIteration, FileNotFoundError):
        print(f"ERROR: Base model {BASE_MODEL} not found in cache")
        return

    print(f"Loading base model from: {base_path}")
    base_sd = load_safetensors(base_path)
    print(f"Base model loaded ({len(base_sd)} tensors)\n")

    # Compare against each 2x MoE variant and collect specialized expert weights
    print("=" * 60)
    print("ANALYZING 2x7B MODELS")
    print("=" * 60)

    results_2x = {}
    specialized_experts = {}  # domain -> state dict

    for moe_model, domain in MOE_2X_MODELS:
        print(f"\nComparing: {moe_model}")
        result = compare_to_2x_model(base_sd, moe_model)
        if result:
            expert_0_matches, expert_1_matches, moe_sd = result
            print(f"  expert_0: {expert_0_matches}/32 layers match base")
            print(f"  expert_1: {expert_1_matches}/32 layers match base")
            results_2x[domain] = (expert_0_matches, expert_1_matches)
            specialized_experts[domain] = moe_sd

    # Summary for 2x models
    print("\n" + "=" * 60)
    print("2x7B SUMMARY")
    print("=" * 60)

    for domain, (e0, e1) in results_2x.items():
        if e0 == 32:
            print(f"Flex-{domain}-2x7B-1T: expert 0 = base, expert 1 = {domain}")
        elif e1 == 32:
            print(f"Flex-{domain}-2x7B-1T: expert 0 = {domain}, expert 1 = base")

    # Analyze 7x7B model
    print("\n" + "=" * 60)
    print("ANALYZING 7x7B MODEL")
    print("=" * 60)

    expert_mapping = analyze_7x_model(base_sd, specialized_experts)

    if expert_mapping:
        print("\n" + "=" * 60)
        print("7x7B EXPERT MAPPING")
        print("=" * 60)
        for idx in range(7):
            domain = expert_mapping.get(idx, "UNKNOWN")
            print(f"  expert {idx}: {domain}")


if __name__ == "__main__":
    main()
