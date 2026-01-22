from itertools import chain
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# old api, inductor hasn't migrated yet
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

MODEL = "allenai/Olmo-3-7B-Instruct"
DEVICE = "cuda:1"
DATASET_NAME = "allenai/dolma3_mix-6T"
CACHE_PATH = "cached_tokens.safetensors"

sequence_length = 8192  # trying the same seq_len as pretraining
num_samples = 1024  # specified in the nvidia paper
batch_size = 8  # adjust for hardware


def get_dataset():
    ds = load_dataset(DATASET_NAME, split="train", streaming=True).shuffle(
        buffer_size=50_000, seed=1223
    )
    return list(ds.take(10_000))  # 10k docs should be ~10M tokens


def tokenize_dataset(dataset):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    texts = [sample["text"] for sample in dataset]
    tokenized = tokenizer(
        texts, padding=False, truncation=False, add_special_tokens=False
    )
    return list(chain.from_iterable(tokenized["input_ids"]))


width_importance_dict = {}
layer_importance_dict = {}


def make_output_hook(name):
    """OUTPUT, computes the mean over the sequence and then the L2 norm over the batch dim. Use this for Layer Norms."""

    def hook(module, input, output):
        output = output.detach()
        output = output.abs()  # mean sequence dimension
        output = output.mean(dim=1).pow(2)
        output = output.sum(dim=0)  # less to drag around this way, sqrt later
        if name in width_importance_dict:
            width_importance_dict[name] += output
        else:
            width_importance_dict[name] = output

    return hook


def make_input_hook(name):
    """INPUT, computes the mean over the sequence and then the L2 norm over the batch dim. Use this for the MLP and Attention Layers."""

    def hook(module, input, output):
        input = input[0].detach()
        input = input.abs()  # mean sequence dimension
        input = input.mean(dim=1).pow(2)
        input = input.sum(dim=0)  # less to drag around this way
        if name in width_importance_dict:
            width_importance_dict[name] += input
        else:
            width_importance_dict[name] = input

    return hook


def make_layer_hook(name):
    """Hooks the layer as a whole, and computes the cosine similarity between the input and the output. For depth pruning."""

    def hook(module, input, output):
        score = 1 - F.cosine_similarity(input[0], output, dim=2).mean()
        if name in layer_importance_dict:
            layer_importance_dict[name] += score
        else:
            layer_importance_dict[name] = score

    return hook


def main():
    if Path(CACHE_PATH).exists():
        all_tokens = load_file(CACHE_PATH)["tokens"]
    else:
        dataset = tokenize_dataset(get_dataset())
        total_tokens = sequence_length * num_samples
        all_tokens = torch.tensor(dataset[:total_tokens]).view(
            num_samples, sequence_length
        )
        save_file({"tokens": all_tokens}, CACHE_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    config = model.config
    model = model.model.to(
        DEVICE
    )  # skips the lm_head, we don't need it for these purposes
    """We want to hook into self_attn, mlp, and layernorms (olmo has 2: post_attention_layernorm, and post_feedforward_layernorm).
    - Hooks should be before compiling (tbd if compiling works at all?)
    - For self attention, we take the L2 norm after the attention operation, registering the hook on o_proj. Still curious about the SWA layers.
    - For mlp, we take activations after W_1. In olmo, this is after mlp.up_proj.
    - For layernorms, we grab the activations after norm."""
    for i, layer in enumerate(model.layers):
        layer.self_attn.o_proj.register_forward_hook(
            make_input_hook(f"layer_{i}.attention")
        )
        layer.mlp.down_proj.register_forward_hook(make_input_hook(f"layer_{i}.mlp"))
        layer.post_attention_layernorm.register_forward_hook(
            make_output_hook(f"layer_{i}.attn_ln")
        )
        layer.post_feedforward_layernorm.register_forward_hook(
            make_output_hook(f"layer_{i}.ffn_ln")
        )
        layer.register_forward_hook(make_layer_hook(f"layer_{i}"))
    # model = torch.compile(model)  # tbd if this works

    for batch_idx in tqdm(range(0, num_samples, batch_size)):
        batch = all_tokens[batch_idx : batch_idx + batch_size].to(DEVICE)
        with torch.inference_mode():
            model(batch, use_cache=False)

    # free up the space on gpu that the model was using
    del model
    torch.cuda.empty_cache()
    for key, value in width_importance_dict.items():
        width_importance_dict[key] = value.pow(0.5)  # finish the l2 norming
    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size
    mlp_importance = torch.zeros(
        num_hidden_layers, intermediate_size, device=DEVICE
    )  # ranking of each neuron
    attention_importance = torch.zeros(
        num_hidden_layers, num_attention_heads, device=DEVICE
    )  # ranking of each head
    ffn_ln_importance = torch.zeros(
        hidden_size, device=DEVICE
    )  # ranking of each ln element
    attn_ln_importance = torch.zeros(hidden_size, device=DEVICE)

    for key, value in width_importance_dict.items():
        layer_idx = int(key.split("_")[1].split(".")[0])
        if "mlp" in key:
            mlp_importance[layer_idx] = value
        elif "attention" in key:
            head_dim = hidden_size // num_attention_heads
            per_head = value.view(num_attention_heads, head_dim)
            attention_importance[layer_idx] = torch.linalg.vector_norm(
                per_head, ord=2, dim=1
            )
        elif "attn_ln" in key:
            attn_ln_importance += value
        elif "ffn_ln" in key:
            ffn_ln_importance += value

    # Flatten for safetensors (only supports flat {str: tensor} dicts)
    scores = {
        "mlp": mlp_importance.cpu(),
        "attention": attention_importance.cpu(),
        "attn_ln": attn_ln_importance.cpu(),
        "ffn_ln": ffn_ln_importance.cpu(),
    }
    # Add layer scores with flattened keys
    for k, v in layer_importance_dict.items():
        scores[f"layer.{k}"] = v.cpu()

    save_file(scores, "importance_scores_tensors/importance_scores.safetensors")


if __name__ == "__main__":
    main()
