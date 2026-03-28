cd eval/LongBench
mkdir -p logs

base_model=$1
model_path=$2
attn_pattern_path=$3
task=$4
static_sparsity=$5
sparse_prefill_mode=$6
precision=$7
kv_quant_granularity=$8

sparse_decode_mode=$9
dynamic_sparse_token_budget=${10}
selector_update_interval=${11}
sub_chunk_per_block=${12}

device=${13}
enable_thinking=${14}
max_samples=${15}

ctx_sink_token=128
ctx_local_token=4096
dec_sink_token=128
dec_local_token=256

ckpt_name=$(basename "$model_path")

suffix=sparse_prefill_${sparse_prefill_mode}_${precision}_sparsity${static_sparsity}_decMode${sparse_decode_mode}_tokenBudget${dynamic_sparse_token_budget}_interval${selector_update_interval}

thinking_arg=""
if [ "$enable_thinking" == "1" ]; then
    thinking_arg="--enable-thinking"
fi

max_samples_arg=""
if [ -n "$max_samples" ] && [ "$max_samples" -gt 0 ] 2>/dev/null; then
    max_samples_arg="--max-samples $max_samples"
fi

longbench_args="--base_model $base_model \
                --quant_model $ckpt_name \
                --model_path $model_path \
                --task $task \
                --sparse_prefill_mode $sparse_prefill_mode \
                --model_name_suffix $suffix \
                $thinking_arg \
                $max_samples_arg"


lserve_args="--ifb-mode \
             --precision $precision \
             --quant-path $model_path \
             --group-size -1 \
             --max-num-batched-tokens 4195000 \
             --max-num-seqs 1 \
             --omit-prompt \
             --kv-quant-granularity $kv_quant_granularity \
             --chunk-prefill-size 32000 \
             --multiblock-switch 1024000 \
             --static-sparse-attn-load-dir $attn_pattern_path \
             --static-sparsity $static_sparsity \
             --sparse-decode-mode $sparse_decode_mode \
             --ctx-sink-token $ctx_sink_token \
             --ctx-local-token $ctx_local_token \
             --dec-sink-token $dec_sink_token \
             --dec-local-token $dec_local_token \
             --sub-chunk-per-block $sub_chunk_per_block \
             --dynamic-sparse-token-budget $dynamic_sparse_token_budget \
             --selector-update-interval $selector_update_interval"


if [ "$sparse_prefill_mode" == "1" ]; then
    CUDA_VISIBLE_DEVICES=${device} python -u pred.py $longbench_args $lserve_args --sparse-context-mode
elif [ "$sparse_prefill_mode" = "0" ]; then
    CUDA_VISIBLE_DEVICES=${device} python -u pred.py $longbench_args $lserve_args
else
    echo "[Error] Invalid sparse_prefill_mode. Choose from ['0', '1']."
fi 2>&1 | tee logs/eval_${model_name}_${task}_${suffix}.log