#!/bin/bash
# Train all FlexOLMo variants on a fresh GPU instance.
#
# Usage:
#   ./train_on_gpu.sh              # Auto-tune batch sizes
#   ./train_on_gpu.sh --force-autotune  # Re-run auto-tuning even if cached
#
# Trains in size-pair order (smallest to largest):
#   1. flex_math_2048 → flex_general_2048
#   2. flex_math_5504 → flex_general_5504
#   3. flex_math_8192 → flex_general_8192
#
# Batch sizes are auto-tuned per GPU and cached in batch_size_cache.json

set -euo pipefail

REPO_URL="https://github.com/hbfreed/variable-flex-olmo.git"
EXTRA_ARGS="${*:-}"

# 1. Install uv (fast Python package manager)
if ! command -v uv &>/dev/null; then
    echo "=== Installing uv ==="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # shellcheck source=/dev/null
    source "$HOME/.local/bin/env"
fi

# 2. Install nvtop for GPU monitoring (optional but useful)
if ! command -v nvtop &>/dev/null; then
    echo "=== Installing nvtop ==="
    if command -v snap &>/dev/null; then
        sudo snap install nvtop
    else
        echo "Note: snap not available, skipping nvtop install"
        echo "You can monitor GPU with: watch -n1 nvidia-smi"
    fi
fi

# 3. Clone repo if not present
if [ ! -d "variable-flex-olmo" ]; then
    echo "=== Cloning repository ==="
    git clone "$REPO_URL"
fi
cd variable-flex-olmo

# 4. Install Python dependencies
echo "=== Installing dependencies ==="
uv sync

# 5. Check HuggingFace authentication
echo "=== Checking HuggingFace auth ==="
if ! uv run huggingface-cli whoami &>/dev/null; then
    echo "Please log in to HuggingFace:"
    uv run huggingface-cli login
fi

# 6. Check Weights & Biases authentication
echo "=== Checking wandb auth ==="
if ! uv run wandb verify &>/dev/null 2>&1; then
    echo "Please log in to Weights & Biases:"
    uv run wandb login
fi

# 7. Train all models in size-pair order
train_model() {
    local importance=$1
    local width=$2
    echo ""
    echo "========================================"
    echo "Training flex_${importance}_${width}"
    echo "========================================"
    # shellcheck disable=SC2086
    uv run python distill_flex_olmo.py -i "$importance" -w "$width" $EXTRA_ARGS
}

echo ""
echo "=== Starting training runs ==="
echo "Batch sizes will be auto-tuned and cached per GPU/width"
echo ""

# Train smallest to largest (2048 → 5504 → 8192)
train_model math 2048
train_model general 2048

train_model math 5504
train_model general 5504

train_model math 8192
train_model general 8192

echo ""
echo "========================================"
echo "All training complete!"
echo "========================================"
