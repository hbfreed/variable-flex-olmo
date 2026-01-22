# Variable FlexOlmo

Megablocks-based MoE implementation for FlexOlmo models with support for variable-sized experts.

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

Run `identify_base_expert.py` to verify these mappings:

```bash
uv run python identify_base_expert.py
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

### Hybrid Loop/Sparse Dispatch

For other configurations, automatically selects between loop and sparse:

| Expert Count | Token Count | Implementation |
|-------------|-------------|----------------|
| 2 (top-k=2) | any | hardcoded 2x2 |
| ≤2 | any | loop |
| 3-7 | <1024 | loop |
| 3-7 | ≥1024 | sparse |
| ≥8 | <16 | loop |
| ≥8 | ≥16 | sparse |

This was determined empirically via `benchmark_loop_vs_sparse.py`.
