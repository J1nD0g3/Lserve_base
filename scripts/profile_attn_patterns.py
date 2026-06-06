"""
Attention pattern profiling for LServe sparse attention.

Profiles a model to identify streaming vs retrieval attention heads
by measuring attention distribution on a passkey retrieval task.

Uses forward hooks to process and discard attention weights layer-by-layer,
so only one layer's attention matrix is in memory at a time.

Output: full_attention_heads.tsv (num_layers x num_kv_heads) + config.json
Compatible with LServe's attn_config.py loading format.

Usage:
    python scripts/profile_attn_patterns.py \
        --model_path /home/jheo/models/Qwen3-8B \
        --output_dir ./attn_patterns/Qwen3-8B \
        --context_length 8192 \
        --num_samples 5
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def generate_passkey_prompt(tokenizer, context_length, essay_text,
                            passkey_value="eat a sandwich and sit in Dolores Park on a sunny day"):
    """Generate a passkey retrieval prompt of a given context length."""
    needle = f"The best thing to do in San Francisco is to {passkey_value}."
    question = "What is the best thing to do in San Francisco?"
    prompt_prefix = "Answer the question based on the given context.\n\nContext:\n"
    prompt_suffix = f"\n\nQuestion: {question}\nAnswer: The best thing to do in San Francisco is to"

    prefix_tokens = tokenizer(prompt_prefix, return_tensors="pt").input_ids[0]
    suffix_tokens = tokenizer(prompt_suffix, return_tensors="pt").input_ids[0]
    needle_tokens = tokenizer(needle, return_tensors="pt").input_ids[0]

    available_tokens = context_length - len(prefix_tokens) - len(suffix_tokens) - len(needle_tokens)
    if available_tokens <= 0:
        raise ValueError(f"Context length {context_length} too short for passkey prompt")

    essay_tokens = tokenizer(essay_text, return_tensors="pt").input_ids[0]
    while len(essay_tokens) < available_tokens:
        essay_tokens = torch.cat([essay_tokens, essay_tokens])
    filler_tokens = essay_tokens[:available_tokens]

    needle_pos = available_tokens // 2
    filler_before = filler_tokens[:needle_pos]
    filler_after = filler_tokens[needle_pos:]

    all_tokens = torch.cat([prefix_tokens, filler_before, needle_tokens, filler_after, suffix_tokens])
    return all_tokens[:context_length]


def compute_retrieval_score(attn_weights, num_kv_heads, heads_per_kv, seq_len, sink_size, local_size):
    """Compute per-KV-head retrieval scores from attention weights.

    attn_weights: (num_q_heads, seq_len, seq_len) float tensor
    Returns: dict mapping kv_idx -> retrieval_mass
    """
    scores = {}
    for kv_idx in range(num_kv_heads):
        start_h = kv_idx * heads_per_kv
        end_h = start_h + heads_per_kv
        kv_attn = attn_weights[start_h:end_h].mean(dim=0)  # (seq_len, seq_len)

        # For the last 32 query positions, measure attention to retrieval region
        n_query = min(32, seq_len)
        query_positions = kv_attn[-n_query:]  # (n_query, seq_len)

        retrieval_mass = 0.0
        count = 0
        for q_offset in range(n_query):
            q_pos = seq_len - n_query + q_offset
            local_start = max(0, q_pos - local_size)
            if local_start > sink_size:
                retrieval_mass += query_positions[q_offset, sink_size:local_start].sum().item()
                count += 1

        scores[kv_idx] = retrieval_mass / count if count > 0 else 0.0
    return scores


def profile_attention_heads(model, tokenizer, args):
    """Profile attention heads using hooks to process one layer at a time."""
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    num_heads = model.config.num_attention_heads
    heads_per_kv = num_heads // num_kv_heads

    # Load essay text for filler
    essay_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval", "needle", "PaulGrahamEssays")
    essay_text = ""
    if os.path.exists(essay_dir):
        for fname in sorted(os.listdir(essay_dir)):
            if fname.endswith(".txt"):
                with open(os.path.join(essay_dir, fname), "r") as f:
                    essay_text += f.read() + "\n"
    if not essay_text:
        essay_text = "This is a filler sentence for attention profiling. " * 10000

    print(f"Model: {args.model_path}")
    print(f"Layers: {num_layers}, KV heads: {num_kv_heads}, Q heads: {num_heads}")
    print(f"Context length: {args.context_length}, Samples: {args.num_samples}")

    head_scores = np.zeros((num_layers, num_kv_heads))

    for sample_idx in range(args.num_samples):
        print(f"\n--- Sample {sample_idx + 1}/{args.num_samples} ---")
        input_ids = generate_passkey_prompt(
            tokenizer, args.context_length, essay_text
        ).unsqueeze(0).to(model.device)

        seq_len = input_ids.shape[1]
        print(f"Input length: {seq_len}")

        # Storage for this sample's scores, collected by hooks
        sample_scores = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output is (attn_output, attn_weights, past_kv) or (attn_output, attn_weights)
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    attn_w = output[1][0].float()  # (num_q_heads, seq_len, seq_len)
                    scores = compute_retrieval_score(
                        attn_w, num_kv_heads, heads_per_kv,
                        seq_len, args.sink_size, args.local_size,
                    )
                    sample_scores[layer_idx] = scores
                    # Drop attention weights to free memory before next layer
                    return (output[0],) + (None,) + output[2:]
                return output
            return hook_fn

        # Register hooks on all attention modules
        hooks = []
        for lidx in range(num_layers):
            h = model.model.layers[lidx].self_attn.register_forward_hook(make_hook(lidx))
            hooks.append(h)

        # Single forward pass with output_attentions=True
        # Hooks capture each layer's attention and replace with None to free memory
        with torch.no_grad():
            model(input_ids, output_attentions=True, use_cache=False)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Accumulate scores
        for lidx, scores in sample_scores.items():
            for kv_idx, score in scores.items():
                head_scores[lidx, kv_idx] += score

        del sample_scores
        import gc; gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Done (peak GPU: {mem_gb:.1f}GB)")

    # Normalize to [0, 1]
    head_scores /= args.num_samples
    if head_scores.max() > 0:
        head_scores = head_scores / head_scores.max()

    return head_scores


def save_patterns(head_scores, args):
    """Save attention patterns in LServe format."""
    os.makedirs(args.output_dir, exist_ok=True)

    tsv_path = os.path.join(args.output_dir, "full_attention_heads.tsv")
    np.savetxt(tsv_path, head_scores, delimiter="\t", fmt="%.18e")
    print(f"\nSaved attention scores to {tsv_path}")
    print(f"Shape: {head_scores.shape}")
    print(f"Score range: [{head_scores.min():.4f}, {head_scores.max():.4f}]")
    print(f"Mean score: {head_scores.mean():.4f}")

    config = {
        "model_name": args.model_path,
        "context_length_min": 1000,
        "context_length_max": args.context_length,
        "num_samples": args.num_samples,
        "sink_size": args.sink_size,
        "recent_size": args.local_size,
        "profiled_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {args.output_dir}/config.json")


def main():
    parser = argparse.ArgumentParser(description="Profile attention patterns for LServe")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--context_length", type=int, default=8192)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--sink_size", type=int, default=128)
    parser.add_argument("--local_size", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--device_map", type=str, default="cuda:0",
                        help="'cuda:0'=single GPU, 'auto'=multi-GPU layer split")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,   # "cuda:0"=single, "auto"=multi-GPU (NO max_memory: it caused NaN)
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    print("Profiling attention heads...")
    start_time = time.time()
    head_scores = profile_attention_heads(model, tokenizer, args)
    elapsed = time.time() - start_time
    print(f"\nProfiling completed in {elapsed:.1f}s")

    save_patterns(head_scores, args)
    print("\nDone!")


if __name__ == "__main__":
    main()
