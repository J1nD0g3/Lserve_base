#!/bin/bash
# RULER evaluation with Qwen3-8B-128k using LServe (sparse attention)
# 102k context, 10 tasks (matching ShadowKV RULER benchmark)
# Logging format follows ShadowKV convention:
#   logs/{Model}_{bench}_{algo}_{timestamp}/
#     progress.log   — full stdout/stderr
#     engine.log     — LServe engine stderr
#     summary.txt    — result table
#     samples/       — per-task result JSONs

set -e

# ===================== Config =====================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OMNISERVE_DIR="$(dirname "$SCRIPT_DIR")"
BASE_MODEL="Qwen3-8B-128k"
MODEL_PATH="${OMNISERVE_DIR}/models/Qwen3-8B-128k-w8a8-per-channel-kv-per-tensor"

# LServe params (same structure as LongBench)
ATTN_PATH="${OMNISERVE_DIR}/attn_patterns/Qwen3-8B-128k"
PRECISION="w8a8kv8"
KV_QUANT_GRANULARITY="per_tensor"
STATIC_SPARSITY=0.50
SPARSE_PREFILL_MODE=1
SPARSE_DECODE_MODE=0
DYNAMIC_ATN_BUDGET=4096
DYNAMIC_SELECT_INTERVAL=4
SUB_CHUNK_PER_BLOCK=4
CTX_SINK_TOKEN=128
CTX_LOCAL_TOKEN=4096
DEC_SINK_TOKEN=128
DEC_LOCAL_TOKEN=256

# RULER config
RULER_DATA_DIR="${OMNISERVE_DIR}/data/ruler/data/qwen3/102400"
DATALEN=102400
ENABLE_THINKING=0  # Set to 1 to enable Qwen3 thinking mode
DEVICE=0
# ==================================================

# Environment setup
if [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
fi
conda activate omniserve
cd "${OMNISERVE_DIR}"
export PYTHONPATH=""
export LD_LIBRARY_PATH="$(python -c 'import torch; print(torch.__path__[0])')/lib:${LD_LIBRARY_PATH}"

# RULER tasks (10 tasks, matching ShadowKV)
RULER_TASKS="niah_single_1,niah_single_2,niah_multikey_1,niah_multikey_2,niah_multivalue,niah_multiquery,vt,fwe,qa_1,qa_2"

# Log directory (ShadowKV naming)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OMNISERVE_DIR}/logs/${BASE_MODEL}_ruler_lserve_${TIMESTAMP}"
mkdir -p "$LOG_DIR/samples"

# Optional args
THINKING_ARG=""
[ "$ENABLE_THINKING" == "1" ] && THINKING_ARG="--enable-thinking"

LSERVE_ARGS="--ifb-mode \
  --precision $PRECISION \
  --quant-path $MODEL_PATH \
  --group-size -1 \
  --max-num-batched-tokens 4195000 \
  --max-num-seqs 1 \
  --omit-prompt \
  --kv-quant-granularity $KV_QUANT_GRANULARITY \
  --chunk-prefill-size 32000 \
  --multiblock-switch 1024000 \
  --static-sparse-attn-load-dir $ATTN_PATH \
  --static-sparsity $STATIC_SPARSITY \
  --sparse-decode-mode $SPARSE_DECODE_MODE \
  --ctx-sink-token $CTX_SINK_TOKEN \
  --ctx-local-token $CTX_LOCAL_TOKEN \
  --dec-sink-token $DEC_SINK_TOKEN \
  --dec-local-token $DEC_LOCAL_TOKEN \
  --sub-chunk-per-block $SUB_CHUNK_PER_BLOCK \
  --dynamic-sparse-token-budget $DYNAMIC_ATN_BUDGET \
  --selector-update-interval $DYNAMIC_SELECT_INTERVAL"

[ "$SPARSE_PREFILL_MODE" == "1" ] && LSERVE_ARGS="$LSERVE_ARGS --sparse-context-mode"

# ==================== Run eval_ruler.py ====================
echo "=== RULER LServe ($BASE_MODEL) ===" | tee "$LOG_DIR/progress.log"
echo "Start: $(date)" | tee -a "$LOG_DIR/progress.log"
echo "Model: $MODEL_PATH" | tee -a "$LOG_DIR/progress.log"
echo "Dataset: RULER (10 tasks, ${DATALEN} context)" | tee -a "$LOG_DIR/progress.log"
echo "Static sparsity: $STATIC_SPARSITY" | tee -a "$LOG_DIR/progress.log"
echo "Log dir: $LOG_DIR" | tee -a "$LOG_DIR/progress.log"
echo "" | tee -a "$LOG_DIR/progress.log"

CUDA_VISIBLE_DEVICES=${DEVICE} \
NUM_RETRIEVAL_GPU_PAGE_BLOCKS=5000 \
NUM_STREAMING_GPU_PAGE_BLOCKS=500 \
python -u eval/ruler/eval_ruler.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$RULER_DATA_DIR" \
    --tasks "$RULER_TASKS" \
    --output_dir "$LOG_DIR" \
    --datalen $DATALEN \
    --precision $PRECISION \
    --static_sparsity $STATIC_SPARSITY \
    --sparse_prefill_mode $SPARSE_PREFILL_MODE \
    --sparse_decode_mode $SPARSE_DECODE_MODE \
    --dynamic_attn_budget $DYNAMIC_ATN_BUDGET \
    --sub_chunk_per_block $SUB_CHUNK_PER_BLOCK \
    $THINKING_ARG \
    $LSERVE_ARGS \
    2>"$LOG_DIR/engine.log" | tee -a "$LOG_DIR/progress.log"

echo "" | tee -a "$LOG_DIR/progress.log"
echo "End: $(date)" | tee -a "$LOG_DIR/progress.log"
echo "Results saved to: $LOG_DIR" | tee -a "$LOG_DIR/progress.log"
