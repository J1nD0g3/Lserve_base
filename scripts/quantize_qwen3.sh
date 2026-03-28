#!/bin/bash
# Quantize Qwen3 models to W8A8KV8 for LServe
# Prerequisites: pip install deepcompressor (in omniserve env)
#
# This script performs:
# 1. W8A8 quantization using DeepCompressor
# 2. Checkpoint conversion to OmniServe format
#
# Usage:
#   bash scripts/quantize_qwen3.sh /home/jheo/models/Qwen3-8B
#   bash scripts/quantize_qwen3.sh /home/jheo/models/Qwen3-8B-128k

set -e

MODEL_PATH="${1:?Usage: $0 <model_path>}"
MODEL_NAME=$(basename "$MODEL_PATH")
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OMNISERVE_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate omniserve
cd "${OMNISERVE_DIR}"

echo "=== Quantizing $MODEL_NAME ==="
echo "Model path: $MODEL_PATH"

# Step 1: Check DeepCompressor installation
python -c "import deepcompressor" 2>/dev/null || {
    echo "[ERROR] DeepCompressor not installed. Run: pip install deepcompressor"
    echo "Or install from source: https://github.com/mit-han-lab/deepcompressor"
    exit 1
}

# Step 2: Run quantization with DeepCompressor
# W8A8 per-channel quantization with per-tensor KV cache
QUANT_OUTPUT_DIR="${OMNISERVE_DIR}/quant_output/${MODEL_NAME}-w8a8"
echo "Quantization output: $QUANT_OUTPUT_DIR"

# NOTE: The exact DeepCompressor command may need adjustment based on version.
# This is the general pattern from the omniserve README (commented-out section).
python -m deepcompressor.app.llm.ptq \
    --model "$MODEL_PATH" \
    --w-bit 8 \
    --w-group-size -1 \
    --kv-bit 8 \
    --kv-group-size -1 \
    --output-dir "$QUANT_OUTPUT_DIR" \
    2>&1 | tee "logs/quantize_${MODEL_NAME}.log"

# Step 3: Convert checkpoint to OmniServe format
FINAL_CKPT="${OMNISERVE_DIR}/models/${MODEL_NAME}-w8a8-per-channel-kv-per-tensor"
echo "Converting checkpoint to: $FINAL_CKPT"

python scripts/ckpt_converter/checkpoint_converter.py \
    --model-path "$MODEL_PATH" \
    --quant-path "$QUANT_OUTPUT_DIR" \
    --output-path "${OMNISERVE_DIR}/models" \
    --w-bit 8 \
    --group-size -1 \
    --device cpu \
    --kv-per-tensor

echo "=== Done ==="
echo "Quantized model saved to: $FINAL_CKPT"
echo ""
echo "To use with LServe evaluation, update model_path in run scripts:"
echo "  MODEL_PATH=\"${FINAL_CKPT}\""
