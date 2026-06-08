#!/bin/bash
# LongBench-v2 evaluation with Qwen3-14B (W16A16, generation-based scoring)
#
# Usage:
#   bash eval/scripts/longbench_v2/run_longbench_v2_qwen3_14b.sh [MODE] [ENABLE_THINKING] [MAX_SAMPLES]
#     MODE: dense | lserve   (default: dense)
#     ENABLE_THINKING: 0 | 1 (default: 1)
#     MAX_SAMPLES: 0 = all 503 (default: 0)
#
# Run from the repo root (Lserve_base/).

set -e

MODE="${1:-dense}"
ENABLE_THINKING="${2:-1}"
MAX_SAMPLES="${3:-0}"

base_model="Qwen3-14B-128k"
model_path=/workspace/models/Qwen3-14B-128k   # YaRN x4 config (131072 max positions)
attn_path=./attn_patterns/Qwen3-14B

# NOTE: KV cache is always int8/int4 in these kernels. With raw HF weights
# (no calibrated per-tensor KV scales) fine_grained granularity is REQUIRED:
# per_tensor falls back to kv_scale=1.0 and visibly corrupts generation.
precision="w16a16kv8"
kv_quant_granularity=fine_grained
datalen=122880   # 131072 - 8192 (thinking budget)
device=0

# LServe sparse config (used when MODE=lserve)
static_sparsity=0.5
dynamic_attn_budget=4096
selector_update_interval=4
sub_chunk_per_block=4
ctx_sink_token=128
ctx_local_token=4096
dec_sink_token=128
dec_local_token=256

thinking_arg=""
think_tag="think_off"
if [ "$ENABLE_THINKING" == "1" ]; then
    thinking_arg="--enable-thinking"
    think_tag="think_on"
fi

max_samples_arg=""
if [ "$MAX_SAMPLES" -gt 0 ] 2>/dev/null; then
    max_samples_arg="--max-samples $MAX_SAMPLES"
fi

common_engine_args="--ifb-mode \
    --precision $precision \
    --kv-quant-granularity $kv_quant_granularity \
    --quant-path $model_path \
    --group-size -1 \
    --max-num-batched-tokens 4195000 \
    --max-num-seqs 1 \
    --omit-prompt \
    --chunk-prefill-size 32000 \
    --multiblock-switch 1024000 \
    --static-sparse-attn-load-dir $attn_path \
    --ctx-sink-token $ctx_sink_token \
    --ctx-local-token $ctx_local_token \
    --dec-sink-token $dec_sink_token \
    --dec-local-token $dec_local_token \
    --sub-chunk-per-block $sub_chunk_per_block \
    --dynamic-sparse-token-budget $dynamic_attn_budget \
    --selector-update-interval $selector_update_interval"

mkdir -p logs

if [ "$MODE" == "dense" ]; then
    run_tag="dense"
    NUM_RETRIEVAL_GPU_PAGE_BLOCKS=2500 \
    NUM_STREAMING_GPU_PAGE_BLOCKS=2500 \
    CUDA_VISIBLE_DEVICES=${device} python -u eval/longbench_v2/eval_longbench_v2.py \
        --model_path $model_path \
        --datalen $datalen \
        --run_tag $run_tag \
        $thinking_arg $max_samples_arg \
        $common_engine_args \
        --static-sparsity 0.0 \
        --sparse-decode-mode 0 \
        2>&1 | tee logs/longbenchv2_${base_model}_${run_tag}_${think_tag}.log
elif [ "$MODE" == "lserve" ]; then
    run_tag="lserve_sp${static_sparsity}_budget${dynamic_attn_budget}"
    NUM_RETRIEVAL_GPU_PAGE_BLOCKS=2500 \
    NUM_STREAMING_GPU_PAGE_BLOCKS=2500 \
    CUDA_VISIBLE_DEVICES=${device} python -u eval/longbench_v2/eval_longbench_v2.py \
        --model_path $model_path \
        --datalen $datalen \
        --run_tag $run_tag \
        $thinking_arg $max_samples_arg \
        $common_engine_args \
        --static-sparsity $static_sparsity \
        --sparse-decode-mode 1 \
        --sparse-context-mode \
        2>&1 | tee logs/longbenchv2_${base_model}_${run_tag}_${think_tag}.log
else
    echo "[Error] Invalid MODE '$MODE'. Choose from ['dense', 'lserve']."
    exit 1
fi
