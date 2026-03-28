"""
RULER benchmark evaluation for LServe.

Evaluates long-context capabilities using the RULER benchmark (10 tasks)
with LServe's sparse attention engine.

Usage:
    python eval/ruler/eval_ruler.py \
        --model_path ./models/Qwen3-8B-128k-w8a8kv8 \
        --data_dir ./eval/ruler/data/qwen3/102400 \
        --tasks niah_single_1,niah_single_2,... \
        --output_dir ./logs/Qwen3-8B-128k_ruler_lserve_YYYYMMDD_HHMMSS \
        [LServe engine args passed via CLI]
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig

# Add omniserve to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from omniserve import EngineArgs, LLMEngine, SamplingParams
from eval.ruler.metrics import (
    compute_ruler_score, postprocess_pred,
    RULER_TASK_METRICS, RULER_GROUPS,
)


def wrap_qwen3_prompt(prompt, enable_thinking=False):
    """Wrap raw prompt with Qwen3 chat template."""
    if enable_thinking:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    else:
        return f"<|im_start|>user\n{prompt}\n/no_think<|im_end|>\n<|im_start|>assistant\n"


def strip_think_tags(response):
    """Remove <think>...</think> block and special tokens from response."""
    # Remove complete <think>...</think> blocks
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    # Remove orphaned </think> (closing tag without opening)
    response = response.replace("</think>", "").strip()
    # Remove orphaned <think> and everything after (unclosed thinking)
    response = re.sub(r"<think>.*", "", response, flags=re.DOTALL).strip()
    response = response.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
    return response


def load_ruler_data(data_dir, task_name):
    """Load RULER task data from jsonl file."""
    jsonl_path = os.path.join(data_dir, task_name, "validation.jsonl")
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"RULER data not found: {jsonl_path}")

    samples = []
    with open(jsonl_path, "r") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def get_gen_len(task_name):
    """Get max generation length for each RULER task."""
    gen_lens = {
        "niah_single_1": 64,
        "niah_single_2": 64,
        "niah_multikey_1": 64,
        "niah_multikey_2": 64,
        "niah_multivalue": 128,
        "niah_multiquery": 128,
        "vt": 30,
        "fwe": 50,
        "qa_1": 32,
        "qa_2": 32,
    }
    return gen_lens.get(task_name, 128)


def initialize_lserve_engine(model_path):
    """Initialize LServe engine from CLI args."""
    parser = argparse.ArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    args, _ = parser.parse_known_args()
    args.model = model_path
    engine_args = EngineArgs.from_cli_args(args)
    engine = LLMEngine.from_engine_args(engine_args)
    return engine, args


_global_request_id = 0


def generate_with_lserve(engine, prompts, stop_token_ids, max_gen_len):
    """Generate responses using LServe engine (matches LongBench process_requests pattern)."""
    global _global_request_id
    request_id_start = _global_request_id

    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0,
        stop_token_ids=stop_token_ids,
        max_tokens=max_gen_len,
    )

    # Add all requests
    for prompt in prompts:
        engine.add_request(str(_global_request_id), prompt, sampling_params)
        _global_request_id += 1

    # Run inference
    outputs = {}
    while engine.has_unfinished_requests():
        request_outputs = engine.step()
        if len(request_outputs) == 0:
            break
        for out in request_outputs:
            if out["finished"]:
                outputs[out["id"]] = out["text"]

    # Return in order, handling both int and str keys
    result = []
    for i in range(len(prompts)):
        key = request_id_start + i
        if key in outputs:
            result.append(outputs[key])
        elif str(key) in outputs:
            result.append(outputs[str(key)])
        else:
            result.append("")
    return result


def evaluate_task(engine, tokenizer, stop_token_ids, data_dir, task_name, output_dir, enable_thinking=False):
    """Evaluate a single RULER task."""
    samples = load_ruler_data(data_dir, task_name)
    gen_len = get_gen_len(task_name)

    print(f"\n{'='*60}")
    print(f"Task: {task_name} ({len(samples)} samples, gen_len={gen_len})")
    print(f"{'='*60}")

    results = []
    scores = []
    input_lens = []
    output_lens = []

    start_time = time.time()
    peak_mem = 0.0

    pbar = tqdm(samples, desc=task_name)
    for i, sample in enumerate(pbar):
        raw_input = sample["input"]
        # Skip wrapping if prompt already has chat template
        if "<|im_start|>" in raw_input:
            prompt = raw_input
            # Inject /no_think before the last <|im_end|> preceding <|im_start|>assistant
            # to prevent Qwen3 from entering thinking mode
            if not enable_thinking and "/no_think" not in prompt:
                assistant_marker = "<|im_start|>assistant"
                idx = prompt.rfind("<|im_end|>", 0, prompt.rfind(assistant_marker))
                if idx != -1:
                    prompt = prompt[:idx] + "\n/no_think" + prompt[idx:]
        else:
            prompt = wrap_qwen3_prompt(raw_input, enable_thinking)
        ground_truth = sample["outputs"]

        # Generate
        predictions = generate_with_lserve(engine, [prompt], stop_token_ids, gen_len)
        raw_prediction = predictions[0] if predictions else ""
        prediction = strip_think_tags(raw_prediction)
        prediction = postprocess_pred(prediction)


        # Compute score
        score = compute_ruler_score(prediction, ground_truth, task_name)

        # Track token lengths
        input_tokens = tokenizer(prompt, return_tensors="pt").input_ids[0]
        input_len = len(input_tokens)
        output_len = len(tokenizer(prediction, return_tensors="pt").input_ids[0]) if prediction else 0

        scores.append(score)
        input_lens.append(input_len)
        output_lens.append(output_len)
        pbar.set_postfix(avg=f"{np.mean(scores):.2f}")
        results.append({
            "index": sample["index"],
            "score": score,
            "input_len": input_len,
            "output_len": output_len,
            "prediction": prediction,
            "ground_truth": ground_truth,
        })

        # Track GPU memory
        if torch.cuda.is_available():
            mem = torch.cuda.max_memory_allocated() / (1024**3)
            peak_mem = max(peak_mem, mem)

    elapsed = time.time() - start_time
    avg_score = np.mean(scores) if scores else 0.0
    stderr = np.std(scores) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0

    # Save per-sample results
    samples_dir = os.path.join(output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    with open(os.path.join(samples_dir, f"{task_name}.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    stats = {
        "task": task_name,
        "num_samples": len(samples),
        "avg_score": avg_score,
        "stderr": stderr,
        "elapsed": elapsed,
        "peak_mem_gb": peak_mem,
        "input_len_mean": int(np.mean(input_lens)),
        "input_len_min": int(np.min(input_lens)),
        "input_len_max": int(np.max(input_lens)),
        "output_len_mean": int(np.mean(output_lens)),
        "output_len_min": int(np.min(output_lens)),
        "output_len_max": int(np.max(output_lens)),
        "metric_name": RULER_TASK_METRICS[task_name][0],
    }

    print(f"  Score: {avg_score:.4f} +/- {stderr:.4f}")
    print(f"  Time: {elapsed:.1f}s, Peak GPU: {peak_mem:.2f}GB")

    return stats


def write_summary(all_stats, args, output_dir, total_elapsed, global_peak_mem):
    """Write summary.txt in ShadowKV-compatible format."""
    summary_path = os.path.join(output_dir, "summary.txt")

    lines = []
    lines.append("=" * 70)
    lines.append("LServe Evaluation Log")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Timestamp:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Model:               {args.model_path}")
    lines.append(f"Method:              lserve")
    lines.append(f"Precision:           {args.precision}")
    lines.append(f"Static sparsity:     {args.static_sparsity}")
    lines.append(f"Max context len:     {args.datalen}")
    lines.append("")
    lines.append("--- Hardware ---")

    gpu_name = "N/A"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    lines.append(f"GPU:                 {gpu_name}")
    lines.append(f"Peak GPU memory:     {global_peak_mem:.2f} GB")
    lines.append(f"Total elapsed time:  {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    lines.append("")

    lines.append("--- LServe Config ---")
    lines.append(f"Sparse prefill:      {'ON' if args.sparse_prefill_mode else 'OFF'}")
    lines.append(f"Sparse decode mode:  {args.sparse_decode_mode}")
    lines.append(f"Dynamic attn budget: {args.dynamic_attn_budget}")
    lines.append(f"Sub chunk per block: {args.sub_chunk_per_block}")
    lines.append("")

    # Per-dataset timing
    lines.append("--- Per-Dataset Timing ---")
    for s in all_stats:
        lines.append(f"  {s['task']:<35s} {s['num_samples']:>3d} samples  {s['elapsed']:>8.1f}s  peak {s['peak_mem_gb']:.2f}GB")
    lines.append("")

    # Results table (ShadowKV format)
    lines.append("--- Results ---")
    lines.append("")
    lines.append(f"|{'Tasks':^42s}|{'n-shot':>6s}|{'Metric':^20s}|   | {'Value':>5s} |   |{'Stderr':>7s}|")
    lines.append(f"|{'-'*42}|{'-'*6}:|{'-'*20}|---|{'-'*7}:|---|{'-'*7}:|")

    # Group results
    task_scores = {s["task"]: s for s in all_stats}
    for group_name, group_tasks in RULER_GROUPS.items():
        group_scores = [task_scores[t]["avg_score"] for t in group_tasks if t in task_scores]
        group_stderrs = [task_scores[t]["stderr"] for t in group_tasks if t in task_scores]
        if group_scores:
            group_avg = np.mean(group_scores)
            group_stderr = np.mean(group_stderrs)
            lines.append(f"|- {group_name:<39s}|      |{'score':^20s}| \u2191 |{group_avg:.4f}| \u00b1 |{group_stderr:.4f}|")
            for t in group_tasks:
                if t in task_scores:
                    s = task_scores[t]
                    lines.append(f"| - {'ruler_' + t:<37s}|     0|{s['metric_name']:^20s}| \u2191 |{s['avg_score']:.4f}| \u00b1 |{s['stderr']:.4f}|")

    lines.append("")

    # Group summary
    lines.append(f"|{'Groups':^26s}| {'Metric':^6s} |   | {'Value':>5s} |   |{'Stderr':>7s}|")
    lines.append(f"|{'-'*26}|{'-'*8}|---|{'-'*7}:|---|{'-'*7}:|")
    for group_name, group_tasks in RULER_GROUPS.items():
        group_scores = [task_scores[t]["avg_score"] for t in group_tasks if t in task_scores]
        group_stderrs = [task_scores[t]["stderr"] for t in group_tasks if t in task_scores]
        if group_scores:
            lines.append(f"|- {group_name:<23s}|{'score':^8s}| \u2191 |{np.mean(group_scores):.4f}| \u00b1 |{np.mean(group_stderrs):.4f}|")

    lines.append("")
    all_scores = [s["avg_score"] for s in all_stats]
    lines.append(f"Average score (all datasets): {np.mean(all_scores)*100:.1f}")
    lines.append("")

    # Inference stats
    lines.append("--- Inference Stats ---")
    for s in all_stats:
        lines.append(
            f"  {s['task']}:  "
            f"input_len={s['input_len_mean']} [{s['input_len_min']}-{s['input_len_max']}]  "
            f"output_len={s['output_len_mean']} [{s['output_len_min']}-{s['output_len_max']}]"
        )
    lines.append("")
    lines.append("=" * 70)

    summary_text = "\n".join(lines)
    with open(summary_path, "w") as f:
        f.write(summary_text)

    print(f"\n{summary_text}")
    print(f"\nSummary saved to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="RULER evaluation with LServe")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated task names")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--datalen", type=int, default=102400)
    # LServe params (for summary logging)
    parser.add_argument("--precision", type=str, default="w8a8kv8")
    parser.add_argument("--static_sparsity", type=float, default=0.5)
    parser.add_argument("--sparse_prefill_mode", type=int, default=1)
    parser.add_argument("--sparse_decode_mode", type=int, default=1)
    parser.add_argument("--dynamic_attn_budget", type=int, default=4096)
    parser.add_argument("--sub_chunk_per_block", type=int, default=4)
    parser.add_argument("--enable-thinking", action="store_true", default=False,
                        help="Enable Qwen3 thinking mode (default: off)")

    args, _ = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize LServe engine
    print("Initializing LServe engine...")
    engine, engine_args = initialize_lserve_engine(args.model_path)

    # Load tokenizer and get stop tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    try:
        gen_config = GenerationConfig.from_pretrained(args.model_path)
        stop_token_ids = gen_config.eos_token_id
        if not isinstance(stop_token_ids, list):
            stop_token_ids = [stop_token_ids]
    except Exception:
        stop_token_ids = [tokenizer.eos_token_id]

    # Evaluate each task
    task_list = [t.strip() for t in args.tasks.split(",")]
    all_stats = []
    total_start = time.time()

    for task_name in task_list:
        stats = evaluate_task(
            engine, tokenizer, stop_token_ids,
            args.data_dir, task_name, args.output_dir,
            enable_thinking=args.enable_thinking,
        )
        all_stats.append(stats)

    total_elapsed = time.time() - total_start
    global_peak_mem = max(s["peak_mem_gb"] for s in all_stats) if all_stats else 0.0

    # Write summary
    write_summary(all_stats, args, args.output_dir, total_elapsed, global_peak_mem)


if __name__ == "__main__":
    main()
