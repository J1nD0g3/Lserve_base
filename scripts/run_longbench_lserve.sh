#!/bin/bash
# LongBench evaluation with Qwen3-8B using LServe (sparse attention)
# Logging format follows ShadowKV convention:
#   logs/{Model}_{bench}_{algo}[_3pct]_{timestamp}/
#     progress.log   — full stdout/stderr
#     summary.txt    — result table
#     samples/       — per-task prediction JSONLs

set -e

# ===================== Config =====================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OMNISERVE_DIR="$(dirname "$SCRIPT_DIR")"
BASE_MODEL="Qwen3-8B"
MODEL_PATH="${OMNISERVE_DIR}/models/Qwen3-8B-w8a8-per-channel-kv-per-tensor"

# LServe params
ATTN_PATH="${OMNISERVE_DIR}/attn_patterns/Qwen3-8B"
PRECISION="w8a8kv8"
KV_QUANT_GRANULARITY="per_tensor"
STATIC_SPARSITY=0.75
SPARSE_PREFILL_MODE=1
SPARSE_DECODE_MODE=1
DYNAMIC_ATN_BUDGET=4096
DYNAMIC_SELECT_INTERVAL=4
SUB_CHUNK_PER_BLOCK=4
CTX_SINK_TOKEN=128
CTX_LOCAL_TOKEN=4096
DEC_SINK_TOKEN=128
DEC_LOCAL_TOKEN=256

ENABLE_THINKING=0  # Set to 1 to enable Qwen3 thinking mode
MAX_SAMPLES=0      # Set to >0 to limit samples per task (e.g., 6 for ~3% test). 0 = all.
DEVICE=0
# ==================================================

# Environment setup
source ~/anaconda3/etc/profile.d/conda.sh
conda activate omniserve
cd "${OMNISERVE_DIR}"
export PYTHONPATH=""
export LD_LIBRARY_PATH="$(python -c 'import torch; print(torch.__path__[0])')/lib:${LD_LIBRARY_PATH}"

# Symlinks
cd eval/LongBench
[ ! -d "models" ] && [ -d "../../models" ] && ln -sf ../../models .
[ ! -d "attn_patterns" ] && [ -d "../../attn_patterns" ] && ln -sf ../../attn_patterns .
cd ../..

# Log directory (ShadowKV naming)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAMPLE_TAG=""
[ "$MAX_SAMPLES" -gt 0 ] 2>/dev/null && SAMPLE_TAG="_3pct"
LOG_DIR="${OMNISERVE_DIR}/logs/${BASE_MODEL}_longbench_lserve${SAMPLE_TAG}_${TIMESTAMP}"
mkdir -p "$LOG_DIR/samples"

# LongBench tasks (19 datasets, matching ShadowKV), joined with + for single pred.py invocation
TASK_LIST=("qasper" "multifieldqa_en" "multifieldqa_zh" "hotpotqa" "2wikimqa" "musique" "narrativeqa" "gov_report" "qmsum" "multi_news" "trec" "triviaqa" "samsum" "passage_count" "passage_retrieval_en" "lcc" "repobench-p" "lsht" "vcsum")
TASKS_JOINED=$(IFS=+; echo "${TASK_LIST[*]}")

CKPT_NAME=$(basename "$MODEL_PATH")
SUFFIX="sparse_prefill_${SPARSE_PREFILL_MODE}_${PRECISION}_sparsity${STATIC_SPARSITY}_decMode${SPARSE_DECODE_MODE}_tokenBudget${DYNAMIC_ATN_BUDGET}_interval${DYNAMIC_SELECT_INTERVAL}"

# Optional args
THINKING_ARG=""
[ "$ENABLE_THINKING" == "1" ] && THINKING_ARG="--enable-thinking"
MAX_SAMPLES_ARG=""
[ "$MAX_SAMPLES" -gt 0 ] 2>/dev/null && MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"

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

# ==================== Run pred.py (single process, all tasks) ====================
START_TIME=$(date +%s)
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

cd eval/LongBench

CUDA_VISIBLE_DEVICES=${DEVICE} \
NUM_RETRIEVAL_GPU_PAGE_BLOCKS=5000 \
NUM_STREAMING_GPU_PAGE_BLOCKS=500 \
python -u pred.py \
    --base_model $BASE_MODEL \
    --quant_model $CKPT_NAME \
    --model_path ./models/$CKPT_NAME \
    --task $TASKS_JOINED \
    --sparse_prefill_mode $SPARSE_PREFILL_MODE \
    --model_name_suffix $SUFFIX \
    $THINKING_ARG $MAX_SAMPLES_ARG \
    $LSERVE_ARGS \
    2>"$LOG_DIR/engine.log" | tee "$LOG_DIR/progress.log"

# ==================== Eval ====================
echo "" | tee -a "$LOG_DIR/progress.log"
echo "--- Computing metrics ---" | tee -a "$LOG_DIR/progress.log"
python -u eval.py --model "$CKPT_NAME" 2>&1 | tee -a "$LOG_DIR/progress.log"

# Copy predictions & results to samples/
for task in "${TASK_LIST[@]}"; do
    PRED_FILE="pred/${CKPT_NAME}/${task}-${SUFFIX}.jsonl"
    [ -f "$PRED_FILE" ] && cp "$PRED_FILE" "$LOG_DIR/samples/${task}.jsonl"
done
[ -f "pred/${CKPT_NAME}/result.json" ] && cp "pred/${CKPT_NAME}/result.json" "$LOG_DIR/samples/"

cd ../..

END_TIME=$(date +%s)
TOTAL_ELAPSED=$((END_TIME - START_TIME))
TOTAL_MIN=$(echo "scale=1; $TOTAL_ELAPSED / 60" | bc)
PEAK_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $DEVICE | head -1)
PEAK_MEM_GB=$(echo "scale=2; $PEAK_MEM / 1024" | bc)

# ==================== Summary (ShadowKV format) ====================
cat > "$LOG_DIR/summary.txt" <<SUMMARY_EOF
======================================================================
LServe Evaluation Log
======================================================================

Timestamp:           $(date '+%Y-%m-%d %H:%M:%S')
Model:               $BASE_MODEL
Method:              lserve
Thinking mode:       $([ "$ENABLE_THINKING" == "1" ] && echo "ON" || echo "OFF")
Max context len:     32768
Samples/dataset:     $([ "$MAX_SAMPLES" -gt 0 ] 2>/dev/null && echo "$MAX_SAMPLES" || echo "-1")
Precision:           $PRECISION

--- Hardware ---
GPU:                 $GPU_NAME
Peak GPU memory:     ${PEAK_MEM_GB} GB
Total elapsed time:  ${TOTAL_ELAPSED}s (${TOTAL_MIN}min)

--- LServe Config ---
Static sparsity:     $STATIC_SPARSITY
Sparse prefill:      $([ "$SPARSE_PREFILL_MODE" == "1" ] && echo "ON" || echo "OFF")
Sparse decode mode:  $SPARSE_DECODE_MODE
Dynamic attn budget: $DYNAMIC_ATN_BUDGET
Sub chunk per block: $SUB_CHUNK_PER_BLOCK

--- Results ---

SUMMARY_EOF

# Append results table from result.json (ShadowKV category format)
if [ -f "$LOG_DIR/samples/result.json" ]; then
    python3 -c "
import json, collections

with open('$LOG_DIR/samples/result.json') as f:
    raw = json.load(f)

# Normalize keys: 'hotpotqa-sparse_prefill_...jsonl' -> 'hotpotqa'
import re
results = {}
for k, v in raw.items():
    task_name = re.split(r'-sparse_prefill_', k)[0]
    task_name = task_name.replace('.jsonl', '')
    results[task_name] = v

# Task -> category mapping & metric names (matching ShadowKV)
CATEGORIES = collections.OrderedDict([
    ('Single-Document QA', [
        ('narrativeqa', 'qa_f1_score'),
        ('qasper', 'qa_f1_score'),
        ('multifieldqa_en', 'qa_f1_score'),
        ('multifieldqa_zh', 'qa_f1_zh_score'),
    ]),
    ('Multi-Document QA', [
        ('hotpotqa', 'qa_f1_score'),
        ('2wikimqa', 'qa_f1_score'),
        ('musique', 'qa_f1_score'),
    ]),
    ('Summarization', [
        ('gov_report', 'rouge_score'),
        ('qmsum', 'rouge_score'),
        ('multi_news', 'rouge_score'),
        ('vcsum', 'rouge_zh_score'),
    ]),
    ('Few-shot Learning', [
        ('trec', 'classification_score'),
        ('triviaqa', 'qa_f1_score'),
        ('samsum', 'rouge_score'),
        ('lsht', 'classification_score'),
    ]),
    ('Synthetic Tasks', [
        ('passage_count', 'count_score'),
        ('passage_retrieval_en', 'retrieval_score'),
    ]),
    ('Code Completion', [
        ('lcc', 'code_sim_score'),
        ('repobench-p', 'code_sim_score'),
    ]),
])

# Task-level table
hdr = '|{:^40s}|{:>6s}|{:>20s}|   |{:>7s}|'.format('Tasks', 'n-shot', 'Metric', 'Value')
sep = '|' + '-'*40 + '|' + '-'*6 + ':|' + '-'*20 + '|---|' + '-'*7 + ':|'
print(hdr)
print(sep)

all_scores = []
group_scores = {}
for cat, tasks in CATEGORIES.items():
    cat_vals = []
    for t, metric in tasks:
        if t in results:
            cat_vals.append(results[t])
    if not cat_vals:
        continue
    cat_avg = sum(cat_vals) / len(cat_vals)
    group_scores[cat] = cat_avg
    print('|- {:<38s}|      |{:<20s}| \u2191 |{:6.4f}|'.format(cat, 'score', cat_avg / 100))
    for t, metric in tasks:
        if t in results:
            v = results[t]
            all_scores.append(v)
            print('| - longbench_{:<27s}|{:>5d}|{:>20s}| \u2191 |{:6.4f}|'.format(t, 0, metric, v / 100))

# Group summary table
print()
print('|{:^26s}|{:>8s}|   |{:>7s}|'.format('Groups', 'Metric', 'Value'))
print('|' + '-'*26 + '|' + '-'*8 + '|---|' + '-'*7 + ':|')
for cat, avg in group_scores.items():
    print('|- {:<24s}|{:<8s}| \u2191 |{:6.4f}|'.format(cat, 'score', avg / 100))

print()
avg = sum(all_scores) / len(all_scores) if all_scores else 0
print('Average score (all datasets): {:.1f}'.format(avg))
" >> "$LOG_DIR/summary.txt"
fi

echo "" >> "$LOG_DIR/summary.txt"
echo "======================================================================" >> "$LOG_DIR/summary.txt"

cat "$LOG_DIR/summary.txt" | tee -a "$LOG_DIR/progress.log"
echo "" | tee -a "$LOG_DIR/progress.log"
echo "Results saved to: $LOG_DIR" | tee -a "$LOG_DIR/progress.log"
