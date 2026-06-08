"""
LongBench-v2 evaluation for LServe (generation-based scoring).

Prompt template, context middle-truncation and answer extraction are ported
verbatim from ShadowKV (data/dataset.py, data/metrics.py) so that results are
directly comparable across repos.

Scoring is purely generation-based: the model generates a CoT answer and the
choice letter is regex-extracted (no log-prob comparison).

Usage:
    python eval/longbench_v2/eval_longbench_v2.py \
        --model_path /workspace/models/Qwen3-14B \
        --datalen 32768 --enable-thinking \
        [LServe engine args: --precision w16a16kv8 --kv-quant-granularity fine_grained
         --static-sparse-attn-load-dir ... etc.]
    NOTE: with raw HF weights use fine_grained KV granularity (per_tensor needs
    calibrated KV scales from a quantized checkpoint).
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import random
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, GenerationConfig

# Add omniserve to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from omniserve import EngineArgs, LLMEngine, SamplingParams

# ============================================================
# Ported verbatim from ShadowKV data/dataset.py
# ============================================================

# LongBench-v2: 0-shot CoT prompt (ABCD multiple choice)
LONGBENCHV2_PROMPT = (
    "Please read the following text and answer the questions below.\n\n"
    "<text>\n{context}\n</text>\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_A}\n"
    "(B) {choice_B}\n"
    "(C) {choice_C}\n"
    "(D) {choice_D}\n\n"
    "Let's think step by step:"
)

LONGBENCHV2_GEN_LEN = 1024  # CoT reasoning needs more tokens


# ============================================================
# Ported verbatim from ShadowKV data/metrics.py
# ============================================================

def postprocess_pred(predict_str: str):

    predict_str = predict_str.strip().replace('<|eot_id|>', '').replace('</s>', '').replace('</s', '').replace('</', '')

    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()

    return predict_str


def longbenchv2_extract_answer(result):
    """Extract A/B/C/D answer from model output."""
    result = result.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', result)
    if match:
        return match.group(1)
    match = re.search(r'The correct answer is ([A-D])', result)
    if match:
        return match.group(1)
    # Fallback: find last standalone A/B/C/D
    match = re.findall(r'\b([A-D])\b', result)
    if match:
        return match[-1]
    return None


def longbenchv2_metric(prediction, ground_truth):
    """LongBench-v2 accuracy metric (exact match on A/B/C/D)."""
    prediction = postprocess_pred(prediction).strip()
    pred_answer = longbenchv2_extract_answer(prediction)
    return 1.0 if pred_answer == ground_truth else 0.0


# ============================================================
# Qwen3 chat wrapping (manual ChatML; byte-identical to
# tokenizer.apply_chat_template(..., enable_thinking=...) for Qwen3)
# ============================================================

def wrap_qwen3_chat(prompt, enable_thinking):
    if enable_thinking:
        # clean prompt; the model generates <think>...</think> then the answer
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    # empty think block skips thinking (official template rendering for enable_thinking=False)
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def build_prompt(tokenizer, sample, datalen, enable_thinking):
    """Fill LONGBENCHV2_PROMPT, middle-truncating the context only (ShadowKV logic).

    Returns (chat_text, input_len)."""
    fields = dict(
        question=sample['question'],
        choice_A=sample['choice_A'],
        choice_B=sample['choice_B'],
        choice_C=sample['choice_C'],
        choice_D=sample['choice_D'],
    )
    prompt_text = LONGBENCHV2_PROMPT.format(context=sample['context'], **fields)
    chat_text = wrap_qwen3_chat(prompt_text, enable_thinking)
    input_ids = tokenizer.encode(chat_text, add_special_tokens=False)
    if len(input_ids) > datalen:
        # Middle truncation on context only
        empty_prompt = LONGBENCHV2_PROMPT.format(context='', **fields)
        overhead_ids = tokenizer.encode(
            wrap_qwen3_chat(empty_prompt, enable_thinking), add_special_tokens=False
        )
        max_ctx_tokens = datalen - len(overhead_ids)
        ctx_ids = tokenizer.encode(sample['context'], add_special_tokens=False)
        if len(ctx_ids) > max_ctx_tokens:
            half = max_ctx_tokens // 2
            ctx_ids = ctx_ids[:half] + ctx_ids[-half:]
        truncated_ctx = tokenizer.decode(ctx_ids, skip_special_tokens=False)
        prompt_text = LONGBENCHV2_PROMPT.format(context=truncated_ctx, **fields)
        chat_text = wrap_qwen3_chat(prompt_text, enable_thinking)
        input_ids = tokenizer.encode(chat_text, add_special_tokens=False)
    return chat_text, len(input_ids)


def strip_thinking(text):
    """Keep only the content after </think> (ShadowKV qwen3.py behavior)."""
    if '</think>' in text:
        return text.split('</think>')[-1].strip()
    if '<think>' in text:
        return ''
    return text


def clean_engine_output(text):
    return text.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()


# ============================================================
# LServe engine (same pattern as eval/ruler/eval_ruler.py)
# ============================================================

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
    """Generate responses using LServe engine."""
    global _global_request_id
    request_id_start = _global_request_id

    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0,
        stop_token_ids=stop_token_ids,
        max_tokens=max_gen_len,
    )

    for prompt in prompts:
        engine.add_request(str(_global_request_id), prompt, sampling_params)
        _global_request_id += 1

    outputs = {}
    while engine.has_unfinished_requests():
        request_outputs = engine.step()
        if len(request_outputs) == 0:
            break
        for out in request_outputs:
            if out["finished"]:
                outputs[out["id"]] = out["text"]

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


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--enable-thinking", action="store_true", default=False,
                        help="Enable Qwen3 thinking mode")
    parser.add_argument("--keep_ratio", type=float, default=None,
                        help="overall decode KV keep ratio to match across methods (e.g. 0.27)")
    parser.add_argument("--datalen", type=int, default=32768,
                        help="max prompt length in tokens (context is middle-truncated)")
    parser.add_argument("--max-gen", type=int, default=0,
                        help="max new tokens (0 = 1024, or 8192 with thinking)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="max samples (0 = all 503)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--run_tag", type=str, default="",
                        help="suffix to distinguish runs (e.g. dense / sparse config)")
    args, _ = parser.parse_known_args()
    return args




def set_lserve_keep_ratio(engine, input_len, keep_ratio):
    """Set per-sample dynamic sparse decode budget = keep_ratio * input_len on all
    layers, so LServe's overall decode KV kept ratio matches across samples/methods.
    With static_sparsity=0 every head is a retrieval head, so all layers keep this budget."""
    if keep_ratio is None:
        return None
    budget = max(64, int(round(keep_ratio * input_len)))
    model = engine.workers[0].model_runner.model
    n = 0
    for layer in model.model.layers:
        waw = getattr(layer.self_attn, "decoding_attention_wrapper", None)
        if waw is not None:
            waw.dynamic_sparse_token_budget = budget
            n += 1
    return budget


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()

    # context window from model config (40960 native, 131072 with YaRN config)
    with open(os.path.join(args.model_path, "config.json")) as f:
        max_position = json.load(f).get("max_position_embeddings", 40960)

    max_gen = args.max_gen
    if max_gen <= 0:
        max_gen = 8192 if args.enable_thinking else LONGBENCHV2_GEN_LEN
    # keep prompt + generation within the model's context window
    max_gen = min(max_gen, max_position - args.datalen)
    assert max_gen > 0, f"datalen {args.datalen} leaves no room for generation (max_position={max_position})"

    model_short = os.path.basename(args.model_path.rstrip("/"))
    think_tag = "think_on" if args.enable_thinking else "think_off"
    run_name = f"longbenchv2-{think_tag}"
    if args.run_tag:
        run_name += f"-{args.run_tag}"
    out_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "pred", model_short)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_name}.jsonl")

    # resume: skip already-processed sample indices
    done_idx = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    done_idx.add(json.loads(line)["idx"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resuming: {len(done_idx)} samples already done in {out_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(args.model_path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    engine, engine_args = initialize_lserve_engine(args.model_path)

    data = load_dataset('THUDM/LongBench-v2', split='train')
    num_samples = len(data) if args.max_samples <= 0 else min(args.max_samples, len(data))

    print(f"LongBench-v2: {num_samples} samples | datalen={args.datalen} "
          f"max_gen={max_gen} thinking={args.enable_thinking} run={run_name}")
    print(f"Output: {out_path}")

    for i in tqdm(range(num_samples), desc="longbenchv2"):
        if i in done_idx:
            continue
        sample = data[i]
        chat_text, input_len = build_prompt(tokenizer, sample, args.datalen, args.enable_thinking)
        applied_budget = set_lserve_keep_ratio(engine, input_len, args.keep_ratio)

        t0 = time.time()
        raw_text = generate_with_lserve(engine, [chat_text], eos_token_ids, max_gen)[0]
        elapsed = time.time() - t0

        raw_text = clean_engine_output(raw_text)
        pred_text = strip_thinking(raw_text)
        correct = longbenchv2_metric(pred_text, sample['answer'])

        record = {
            "idx": i,
            "prediction": pred_text,
            "ground_truth": sample['answer'],
            "correct": correct,
            "extracted": longbenchv2_extract_answer(postprocess_pred(pred_text).strip()),
            "input_len": input_len,
            "difficulty": sample.get('difficulty', ''),
            "length": sample.get('length', ''),
            "domain": sample.get('domain', ''),
            "elapsed_sec": round(elapsed, 2),
        }
        if args.enable_thinking:
            record["raw_output_with_think"] = raw_text
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # final summary over the full output file (including resumed samples)
    records = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    if records:
        acc = sum(r["correct"] for r in records) / len(records)
        summary = {
            "run": run_name,
            "model": model_short,
            "samples": len(records),
            "accuracy": round(acc, 4),
        }
        for key in ("difficulty", "length"):
            groups = {}
            for r in records:
                groups.setdefault(r.get(key, ""), []).append(r["correct"])
            summary[f"by_{key}"] = {
                k: {"n": len(v), "acc": round(sum(v) / len(v), 4)}
                for k, v in sorted(groups.items())
            }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        with open(os.path.join(out_dir, f"{run_name}.summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
