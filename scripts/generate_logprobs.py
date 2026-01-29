# would use vllm, but multi gpu with vllm is broken with this model.
import gc
import sys
from itertools import chain
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from datasets import Dataset, load_dataset
from einops import rearrange
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from flexolmo_adapter import FlexOlmoAdapter

model_name = "allenai/Flex-math-2x7B-1T"
dataset_path = "allenai/dolmino-mix-1124"
cache_file = "tokenized_chunks.pt"
compile = True
batch_size = 7
TOP_K = 128
chunk_size = 4096  # context size of flexolmo

# Enable TF32 for better performance on Ampere+ GPUs. Old API for torch.compile compatibility for now
torch.set_float32_matmul_precision("high")


@torch.compile
def extract_topk_logprobs(logits, k):
    topk_logits, topk_indices = torch.topk(logits, k=k, dim=-1)
    logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
    topk_logprobs = topk_logits - logsumexp
    return topk_logprobs.half(), topk_indices.int()


def patch_model_with_megablocks(model):
    """Patch FlexOlmo MoE layers to use megablocks for faster inference."""
    adapter = FlexOlmoAdapter(model.config)

    for layer in model.model.layers:
        mlp = layer.mlp
        if not (hasattr(mlp, "gate") and hasattr(mlp, "experts")):
            continue

        layer_device = mlp.gate.weight.device
        weights = adapter.transform_weights(mlp.experts)
        weights = {k: v.to(layer_device) for k, v in weights.items()}
        expert_size_blocks = adapter.expert_size_blocks.to(layer_device)
        expert_block_offsets = adapter.expert_block_offsets.to(layer_device)
        gate = mlp.gate

        def make_patched_forward(
            gate, weights, adapter, expert_size_blocks, expert_block_offsets
        ):
            def patched_forward(hidden_states):
                batch_size, seq_len, hidden_dim = hidden_states.shape
                hidden_flat = rearrange(hidden_states, "b s h -> (b s) h")
                router_logits = gate(hidden_flat)
                router_probs = torch.softmax(router_logits, dim=-1)
                top_k_weights, top_k_index = torch.topk(
                    router_probs, k=adapter.num_active_experts, dim=-1
                )

                from megablocks_core import moe_forward

                output_flat = moe_forward(
                    hidden_states=hidden_flat,
                    top_k_index=top_k_index,
                    top_k_weights=top_k_weights,
                    w1=weights["w1_gate"],
                    w2=weights["w2"],
                    activation_fn=adapter.activation,
                    num_experts=adapter.num_experts,
                    num_active_experts=adapter.num_active_experts,
                    sort_end_bit=adapter.sort_end_bit,
                    transpose_sort_end_bit=adapter.transpose_sort_end_bit,
                    block_size=adapter.block_size,
                    expert_size_blocks=expert_size_blocks,
                    expert_block_offsets=expert_block_offsets,
                    total_expert_width=adapter.total_expert_width,
                    w1_up=weights["w1_up"],
                    precomputed_offsets_2x2=adapter.precomputed_offsets_2x2,
                )
                return rearrange(
                    output_flat, "(b s) h -> b s h", b=batch_size, s=seq_len
                ), router_logits

            return patched_forward

        mlp.forward = make_patched_forward(
            gate, weights, adapter, expert_size_blocks, expert_block_offsets
        )

        # Free original weights to save memory
        for expert in mlp.experts:
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                if hasattr(expert, proj):
                    getattr(expert, proj)._parameters.clear()
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Patched {len(model.model.layers)} layers with megablocks")


def load_or_build_chunks():
    cache_path = Path(cache_file)
    if cache_path.exists():
        print(f"Loading cached chunks from {cache_file}...")
        chunks = torch.load(cache_file)
        print(f"Loaded {len(chunks)} chunks")
    else:
        print(f"Loading dataset from {dataset_path}...")
        ds = load_dataset(
            "allenai/dolmino-mix-1124",
            data_files=[
                "data/math/gsm8k/**/*.jsonl",
                "data/math/metamath-owmfilter/**/*.jsonl",
                "data/math/tulu_math/**/*.jsonl",
            ],
            split="train",
        )
        tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")

        print("Tokenizing dataset in memory...")
        all_texts = list(ds["text"])
        tokenized = tokenizer(all_texts)["input_ids"]
        all_tokens = list(
            chain.from_iterable(
                tokens + [tokenizer.eos_token_id] for tokens in tokenized
            )
        )
        print(f"Total tokens: {len(all_tokens):,}")

        num_chunks = len(all_tokens) // chunk_size
        chunks = torch.tensor(
            [
                all_tokens[i * chunk_size : (i + 1) * chunk_size]
                for i in range(num_chunks)
            ]
        )
        print(f"Saving {len(chunks)} chunks to {cache_file}...")
        torch.save(chunks, cache_file)

    return chunks


def get_device_map():
    """Hardcoded device map for actual even distribution. HuggingFace auto/balanced/whatever is not ideal"""
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


def main():
    chunks = load_or_build_chunks().pin_memory()

    print(f"Loading model {model_name}...")
    device_map = get_device_map()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=device_map,
    )
    model.eval()

    patch_model_with_megablocks(model)

    if compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode="default")

    rows = []
    shard_idx = 0

    num_batches = len(chunks) // batch_size
    # Arrow list columns use 32-bit offsets and ~2GB buffers; cap rows per shard to avoid overflow.
    per_row_topk = chunk_size * TOP_K
    max_rows_by_elements = (2**31 - 1) // per_row_topk
    max_rows_by_bytes = (2**31 - 1) // (
        per_row_topk * 4
    )  # int32 topk_indices is the tightest column
    rows_per_shard = max(1, min(max_rows_by_elements, max_rows_by_bytes))

    for batch_start in tqdm(
        range(0, num_batches * batch_size, batch_size), desc="Processing batches"
    ):
        batch = chunks[batch_start : batch_start + batch_size].to(
            model.device, non_blocking=True
        )
        with torch.inference_mode():
            logits = model(batch).logits  # [batch, seq, vocab]

        # Get top-k from raw logits (softmax is monotonic, so indices are the same)
        # Then normalize only the top-k values to avoid full-vocabulary log_softmax
        topk_logprobs, topk_indices = extract_topk_logprobs(logits, TOP_K)

        # Batch transfer to CPU, then convert to lists
        batch_ids = chunks[batch_start : batch_start + batch_size].numpy()
        topk_indices_list = topk_indices.int().cpu().numpy()
        topk_logprobs_list = topk_logprobs.half().cpu().numpy()

        for sample_idx in range(len(batch_ids)):
            rows.append(
                {
                    "input_ids": batch_ids[sample_idx],
                    "topk_indices": topk_indices_list[sample_idx],
                    "topk_logprobs": topk_logprobs_list[sample_idx],
                }
            )

        if len(rows) >= rows_per_shard:
            shard = Dataset.from_list(rows)
            shard.push_to_hub(
                "hbfreed/flexolmo-math-logprobs", split=f"train_{shard_idx}"
            )
            rows = []
            shard_idx += 1
            print(f"Pushed shard {shard_idx}")

    # Leftover rows
    if rows:
        shard = Dataset.from_list(rows)
        shard.push_to_hub("hbfreed/flexolmo-math-logprobs", split=f"train_{shard_idx}")


if __name__ == "__main__":
    main()
