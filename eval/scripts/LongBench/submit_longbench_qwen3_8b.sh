base_model="Qwen3-8B"
attn_path=./attn_patterns/Qwen3-8B
model_path=./models/Qwen3-8B-w8a8-per-channel-kv-per-tensor

cd eval/LongBench
if [ ! -d "models" ]; then
    ln -s ../../models .
fi
if [ ! -d "attn_patterns" ]; then
    ln -s ../../attn_patterns .
fi
cd ../..


# For pure dense baseline, please set static_sparsity=0.0, sparse_prefill_mode=0, and sparse_decode_mode=0

task_list=("2wikimqa" "dureader" "hotpotqa" "multi_news" "qasper" "qmsum" "samsum" "triviaqa")
static_sparsity=0.5

sparse_prefill_mode=1
precision="w8a8kv8"
kv_quant_granularity=per_tensor

sparse_decode_mode=1
dynamic_attn_budget=4096
dynamic_select_interval=4
sub_chunk_per_block=4

device=0
enable_thinking=0  # Set to 1 to enable Qwen3 thinking mode

ckpt_name=$(basename "$model_path")

# Single GPU: run tasks sequentially
for task in ${task_list[@]}; do
    NUM_RETRIEVAL_GPU_PAGE_BLOCKS=5000 \
    NUM_STREAMING_GPU_PAGE_BLOCKS=500 \
    bash eval/scripts/LongBench/longbench.sh \
    $base_model $model_path $attn_path \
    $task \
    $static_sparsity $sparse_prefill_mode \
    $precision $kv_quant_granularity \
    $sparse_decode_mode $dynamic_attn_budget $dynamic_select_interval $sub_chunk_per_block \
    $device $enable_thinking
done

cd eval/LongBench
python -u eval.py --model $ckpt_name
cd ../..
