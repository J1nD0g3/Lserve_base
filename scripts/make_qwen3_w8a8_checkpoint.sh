#!/bin/bash
# Build an OmniServe-loadable W8A8KV8 (per-channel weights, per-tensor KV) checkpoint
# for a Qwen3 model, using the KV-calibration-fixed simple_quantize_w8a8.py.
#
# IMPORTANT for Qwen3:
#   - LServe's int8-KV DECODE requires per_tensor granularity with REAL calibrated KV
#     scales. simple_quantize_w8a8.py captures K after k_norm and V at v_proj (Qwen3
#     QK-norm + outlier channels). Run with `--precision w8a8kv8 --kv-quant-granularity
#     per_tensor` (NOT fine_grained — its decode kernel is broken for Qwen3).
#   - deepcompressor does NOT support Qwen3 (only up to Qwen2); do not use it here.
#
# Usage: bash scripts/make_qwen3_w8a8_checkpoint.sh <HF_MODEL_DIR> [OUTPUT_ROOT] [CALIB_GPU]
#   e.g. bash scripts/make_qwen3_w8a8_checkpoint.sh /workspace/models/Qwen3-14B-128k /root/lserve_models 2
set -euo pipefail
MODEL_PATH="${1:?usage: $0 <hf_model_dir> [output_root] [gpu]}"
OUT_ROOT="${2:-/root/lserve_models}"          # local disk (avoid networked-FS large-write errors)
GPU="${3:-0}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"; OMNISERVE_DIR="$(dirname "$SCRIPT_DIR")"
NAME="$(basename "$MODEL_PATH")"
QUANT_OUT="/root/quant_output/${NAME}-w8a8"
cd "$OMNISERVE_DIR"

echo "### [1/2] simple_quantize_w8a8 (real LongBench calib + fixed K/V capture) ###"
CUDA_VISIBLE_DEVICES="$GPU" python scripts/simple_quantize_w8a8.py \
  --model-path "$MODEL_PATH" --output-dir "$QUANT_OUT" \
  --num-calib-samples 16 --calib-seq-len 4096

echo "### [2/2] checkpoint_converter (qwen3, w8 per-channel, kv per-tensor) ###"
CUDA_VISIBLE_DEVICES="$GPU" python scripts/ckpt_converter/checkpoint_converter.py \
  --model-type qwen3 --model-path "$MODEL_PATH" --quant-path "$QUANT_OUT" \
  --output-path "$OUT_ROOT" --w-bit 8 --group-size -1 --device cpu --kv-per-tensor

CKPT="$OUT_ROOT/${NAME}-w8a8-per-channel-kv-per-tensor"
echo "### done -> $CKPT"
echo "Run LServe with: --model_path $CKPT --quant-path $CKPT --precision w8a8kv8 --kv-quant-granularity per_tensor"
