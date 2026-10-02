# Variable FlexOlmo

Pruning and distillation for [FlexOlmo](https://arxiv.org/abs/2507.07024) experts. Shrink the specialized expert MLPs, retrain with KLD distillation, and get competitive math performance with significantly smaller experts.

## Results

Pruned models trained with ~228M tokens of distillation from [Flex-math-2x7B-1T](https://huggingface.co/allenai/Flex-math-2x7B-1T):

| Model | Total Params | Expert Params | Expert Width | GSM8K | MATH | Math2 |
|-------|-------------|---------------|-------------|-------|------|-------|
| Public model, no expert | 7.3B | 0 | — | — | — | 8.1 |
| Flex-math-2x7B-1T (baseline) | 11.6B | 4.3B | 11008 (100%) | 69.7 | 35.4 | 52.5 |
| [flex-math-8192](https://huggingface.co/hbfreed/flex-math-8192) | 10.5B | 3.2B | 8192 (74%) | 70.1 | 31.3 | 50.7 |
| [flex-math-5504](https://huggingface.co/hbfreed/flex-math-5504) | 9.5B | 2.2B | 5504 (50%) | 66.6 | 26.8 | 46.7 |
| [flex-math-2048](https://huggingface.co/hbfreed/flex-math-2048) | 8.1B | 0.8B | 2048 (19%) | 44.3 | 13.9 | 29.1 |
| flex-math-2048 (pruned only, no distillation) | 8.1B | 0.8B | 2048 (19%) | 13.1 | 3.3 | 8.2 |

The 8192 model matches the full-sized expert. Even the 2048 model (0.8B expert) scores 3.6x the no-expert baseline. Without distillation, pruning alone basically breaks the model (13% GSM8K) — distillation is doing the heavy lifting.

## FlexOlmo Expert Mapping

Analysis of AllenAI's FlexOlmo model family to identify expert indices.

### 2x7B Models (2 experts each)

All 2x7B variants share the same structure:

| Expert Index | Role |
|-------------|------|
| 0 | Base "public" model (Flex-public-7B-1T weights) |
| 1 | Specialized domain expert |

**Available 2x7B variants:**
- `Flex-math-2x7B-1T` - expert 1 = math
- `Flex-code-2x7B-1T` - expert 1 = code
- `Flex-creative-2x7B-1T` - expert 1 = creative
- `Flex-news-2x7B-1T` - expert 1 = news
- `Flex-pes2o-2x7B-1T` - expert 1 = pes2o (scientific papers)
- `Flex-reddit-2x7B-1T` - expert 1 = reddit

### 7x7B Model (7 experts)

The `FlexOlmo-7x7B-1T` model has a different composition:

| Expert Index | Domain |
|-------------|--------|
| 0 | public (base) |
| 1 | unknown* |
| 2 | unknown* |
| 3 | unknown* |
| 4 | pes2o |
| 5 | creative |
| 6 | news |

*Experts 1-3 do not match any of the released 2x7B specialized experts (math, code, reddit). They appear to be different domain experts not released as standalone 2x7B models.

### Verification

Run `scripts/identify_base_expert.py` to verify these mappings:

```bash
uv run python scripts/identify_base_expert.py
```

This script compares weights using `torch.allclose` between:
- Base model: `allenai/Flex-public-7B-1T`
- 2x7B models: `allenai/Flex-{domain}-2x7B-1T`
- 7x7B model: `allenai/FlexOlmo-7x7B-1T`

## Optimized MoE Implementations

`megablocks_core.py` includes multiple MoE forward implementations that are automatically selected based on configuration:

### Hardcoded 2x2 Path (with torch.compile)

For 2 experts with top-k=2 (every token goes to both experts), a specialized implementation eliminates all dynamic indexing:
- No loop iteration
- No mask computation or `nonzero()` calls
- No `index_add_()` accumulation
- `torch.where` instead of `scatter_` for weight remapping
- `torch.compile` to fuse operations

**Speedup vs generic loop:**
| Token Count | Speedup |
|-------------|---------|
| 1           | 2.09x   |
| 16          | 1.95x   |
| 256         | 1.58x   |
| 4096        | 1.17x   |
| 65536       | 1.16x   |

## Training

```bash
# Single GPU
uv run python distill_flex_olmo.py -w WIDTH --push-to-hub

# Multi-GPU
uv run torchrun --nproc_per_node=N distill_flex_olmo.py -w WIDTH --push-to-hub
```

Widths: 2048, 5504, 8192. Add `--resume` to continue from checkpoint.

## Evaluation

Run evals with lm-eval-harness:

```bash
uv run lm_eval --model hf \
    --model_args "pretrained=hbfreed/flex-math-2048,tokenizer=allenai/dolma2-tokenizer,trust_remote_code=True,dtype=bfloat16,attn_implementation=flash_attention_2" \
    --tasks gsm8k_cot,minerva_math \
    --batch_size auto \
    --device cuda:0 \
    --output_path ./results/flex-math-2048
```

For multi-GPU, add `parallelize=True` to model_args and remove `--device`.

To eval a pre-distillation (pruned-only) checkpoint, use an earlier HuggingFace revision:

```bash
# Download pruned-only weights from first commit, pair with current config
python -c "
from huggingface_hub import hf_hub_download
for f in ['model.safetensors']:
    hf_hub_download('hbfreed/flex-math-2048', f, revision='d84212b90f49', local_dir='pruned_models/flex_math_2048_pruned_only')
for f in ['config.json', 'modeling_pruned_flex_olmo.py', 'configuration_pruned_flex_olmo.py']:
    hf_hub_download('hbfreed/flex-math-2048', f, local_dir='pruned_models/flex_math_2048_pruned_only')
"

# Then eval as usual, pointing at the local dir
uv run lm_eval --model hf \
    --model_args "pretrained=pruned_models/flex_math_2048_pruned_only,tokenizer=allenai/dolma2-tokenizer,trust_remote_code=True,dtype=bfloat16,attn_implementation=flash_attention_2" \
    --tasks gsm8k_cot,minerva_math \
    --batch_size 4 \
    --device cuda:0 \
    --output_path ./results/flex-math-2048-pruned-only
```
