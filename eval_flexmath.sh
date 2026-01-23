#!/bin/bash
# Evaluate FlexMath (allenai/Flex-math-2x7B-1T) using lm-eval-harness with vLLM backend

set -euo pipefail

# Configuration
MODEL_NAME="allenai/Flex-math-2x7B-1T"
TOKENIZER_NAME="allenai/dolma2-tokenizer"
PORT=8000
BASE_URL="http://localhost:${PORT}/v1/completions"
OUTPUT_DIR="./results/flexmath"
VLLM_LOG="${OUTPUT_DIR}/vllm_server.log"

# Parse command line arguments
RUN_SANITY=false
LIMIT=""
SKIP_SERVER=false

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --sanity        Also run sanity check benchmarks (hellaswag, arc_challenge)"
    echo "  --limit N       Limit number of examples per task (for testing)"
    echo "  --skip-server   Skip starting vLLM server (assume already running)"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --limit 10           # Quick test with 10 examples per task"
    echo "  $0 --sanity             # Run math + sanity checks"
    echo "  $0                      # Full math evaluation (gsm8k + minerva_math)"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --sanity)
            RUN_SANITY=true
            shift
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --skip-server)
            SKIP_SERVER=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Cleanup function
cleanup() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        echo "Shutting down vLLM server (PID: ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Start vLLM server
start_vllm_server() {
    echo "Starting vLLM server (4-bit bitsandbytes quantization)..."
    echo "Model: ${MODEL_NAME}"
    echo "Port: ${PORT}"
    echo "Log: ${VLLM_LOG}"

    vllm serve "${MODEL_NAME}" \
        --tokenizer "${TOKENIZER_NAME}" \
        --quantization bitsandbytes \
        --load-format bitsandbytes \
        --port "${PORT}" \
        --trust-remote-code \
        --api-key "" \
        > "${VLLM_LOG}" 2>&1 &

    VLLM_PID=$!
    echo "vLLM server started with PID: ${VLLM_PID}"
}

# Wait for server to be ready
wait_for_server() {
    echo "Waiting for vLLM server to be ready..."
    local max_attempts=120  # 10 minutes (5s * 120)
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "vLLM server is ready!"
            return 0
        fi

        # Check if process is still running
        if [[ -n "${VLLM_PID:-}" ]] && ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "ERROR: vLLM server process died unexpectedly"
            echo "Check log: ${VLLM_LOG}"
            tail -50 "${VLLM_LOG}" 2>/dev/null || true
            exit 1
        fi

        attempt=$((attempt + 1))
        echo "  Attempt ${attempt}/${max_attempts}... (waiting 5s)"
        sleep 5
    done

    echo "ERROR: vLLM server did not become ready in time"
    exit 1
}

# Test the endpoint
test_endpoint() {
    echo "Testing vLLM endpoint..."
    local response
    response=$(curl -s "http://localhost:${PORT}/v1/models")

    if echo "${response}" | grep -q "${MODEL_NAME}"; then
        echo "Endpoint test successful!"
        echo "Model available: ${MODEL_NAME}"
        return 0
    else
        echo "ERROR: Endpoint test failed"
        echo "Response: ${response}"
        exit 1
    fi
}

# Run lm-eval with specified tasks
run_lm_eval() {
    local tasks=$1
    local output_path=$2
    local description=$3

    echo ""
    echo "========================================"
    echo "Running ${description}"
    echo "Tasks: ${tasks}"
    echo "Output: ${output_path}"
    echo "========================================"

    uv run lm_eval --model local-completions \
        --model_args "model=${MODEL_NAME},base_url=${BASE_URL},tokenizer_backend=huggingface,tokenizer=${TOKENIZER_NAME},num_concurrent=1,max_retries=3" \
        --tasks "${tasks}" \
        --batch_size 1 \
        ${LIMIT} \
        --output_path "${output_path}"

    echo "Results saved to: ${output_path}"
}

# Main execution
main() {
    echo "============================================"
    echo "FlexMath Evaluation Script"
    echo "Model: ${MODEL_NAME}"
    echo "============================================"
    echo ""

    # Start server unless skipped
    if [[ "${SKIP_SERVER}" == "false" ]]; then
        start_vllm_server
        wait_for_server
    else
        echo "Skipping server start (--skip-server specified)"
        echo "Assuming vLLM server is already running on port ${PORT}"
    fi

    # Test endpoint
    test_endpoint

    # Run Math2 benchmarks (gsm8k + minerva_math)
    run_lm_eval "gsm8k,minerva_math" "${OUTPUT_DIR}/math2" "Math2 Benchmarks (gsm8k + minerva_math)"

    # Run sanity checks if requested
    if [[ "${RUN_SANITY}" == "true" ]]; then
        run_lm_eval "hellaswag,arc_challenge" "${OUTPUT_DIR}/sanity" "Sanity Check Benchmarks"
    fi

    echo ""
    echo "============================================"
    echo "Evaluation complete!"
    echo "Results directory: ${OUTPUT_DIR}"
    echo ""
    echo "Note: Results evaluated with 4-bit bitsandbytes quantization."
    echo "Scores may differ slightly from bf16/fp16 results."
    echo "============================================"
}

main
