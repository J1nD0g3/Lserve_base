"""
Simple W8A8KV8 quantization for LServe checkpoint converter.

Performs per-channel weight quantization (W8) and computes KV cache
scaling factors using calibration data.

Generates model.pt, scale.pt, acts.pt compatible with:
  scripts/ckpt_converter/checkpoint_converter.py

Usage:
    python scripts/simple_quantize_w8a8.py \
        --model-path /home/jheo/models/Qwen3-8B \
        --output-dir ./quant_output/Qwen3-8B-w8a8 \
        --num-calib-samples 32
"""

import argparse
import os
import sys

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Patch typing.Self for Python < 3.11
if sys.version_info < (3, 11):
    import typing
    try:
        from typing_extensions import Self
        typing.Self = Self
    except ImportError:
        pass


def skip(*args, **kwargs):
    pass

# Skip weight init for speed
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.kaiming_normal_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip


def get_calib_data(tokenizer, num_samples=32, seq_len=2048):
    """Generate calibration data from random wiki-like text."""
    # Use the tokenizer's vocab to create pseudo-random calibration sequences
    np.random.seed(42)
    calib_data = []
    vocab_size = tokenizer.vocab_size
    for _ in range(num_samples):
        ids = np.random.randint(100, vocab_size, size=seq_len)
        calib_data.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0))
    return calib_data


def quantize_weight_per_channel(weight):
    """Per-channel symmetric INT8 quantization of a 2D weight matrix.

    Returns: (fake_quant_weight, scale, zero_point)
    """
    # Per output-channel (dim=0) symmetric quantization
    max_val = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scale = max_val / 127.0
    quantized = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)
    fake_quant = quantized.float() * scale
    return fake_quant, scale.squeeze(1), torch.zeros(weight.shape[0], dtype=torch.int8)


def collect_kv_scales(model, tokenizer, num_samples=32, seq_len=512):
    """Run calibration to collect KV cache dynamic ranges."""
    device = next(model.parameters()).device
    calib_data = get_calib_data(tokenizer, num_samples, seq_len)

    num_layers = model.config.num_hidden_layers
    k_maxes = [0.0] * num_layers
    v_maxes = [0.0] * num_layers

    # Hook to capture KV values
    hooks = []
    kv_stats = {}

    def make_kv_hook(layer_idx):
        def hook_fn(module, input, output):
            # For Qwen3/Llama attention, we need to capture K and V after projection
            # output = (attn_output, attn_weights, past_key_value)
            # But with use_cache=True, past_key_value has K and V
            if isinstance(output, tuple) and len(output) >= 3 and output[2] is not None:
                past_kv = output[2]
                if hasattr(past_kv, 'key_cache') and hasattr(past_kv, 'value_cache'):
                    # DynamicCache
                    if len(past_kv.key_cache) > layer_idx:
                        k = past_kv.key_cache[layer_idx]
                        v = past_kv.value_cache[layer_idx]
                        k_maxes[layer_idx] = max(k_maxes[layer_idx], k.abs().max().item())
                        v_maxes[layer_idx] = max(v_maxes[layer_idx], v.abs().max().item())
        return hook_fn

    for lidx in range(num_layers):
        h = model.model.layers[lidx].self_attn.register_forward_hook(make_kv_hook(lidx))
        hooks.append(h)

    print("Collecting KV cache statistics...")
    for i, input_ids in enumerate(tqdm(calib_data[:num_samples])):
        input_ids = input_ids.to(device)
        with torch.no_grad():
            model(input_ids, use_cache=True)

    for h in hooks:
        h.remove()

    return k_maxes, v_maxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-calib-samples", type=int, default=32)
    parser.add_argument("--calib-seq-len", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    config = model.config
    num_layers = config.num_hidden_layers

    # Step 1: Quantize weights per-channel W8
    print("Quantizing weights (W8 per-channel)...")
    model_state = {}
    scale_state = {}

    for name, param in tqdm(list(model.named_parameters())):
        model_state[name] = param.data.clone().cpu()
        if "_proj.weight" in name and param.ndim == 2:
            fq_weight, scale, zeros = quantize_weight_per_channel(param.data.float().cpu())
            model_state[name] = fq_weight.half()
            scale_state[f"{name}.scale.0"] = scale.half()
            # zero_point
            scale_state[f"{name.replace('.weight', '')}.weight.zero"] = zeros

    # Fix scale key names: checkpoint_converter expects "model.layers.{i}.{subname}.weight.scale.0"
    # Our keys are "model.layers.{i}.{subname}.weight.scale.0" which matches
    # But we need to reformat zeros key
    fixed_scale_state = {}
    for key, val in scale_state.items():
        if ".weight.scale.0" in key:
            fixed_scale_state[key] = val
        elif ".weight.zero" in key:
            fixed_scale_state[key] = val

    # Step 2: Collect KV cache scaling factors
    print("Collecting KV cache dynamic ranges...")
    k_maxes, v_maxes = collect_kv_scales(
        model, tokenizer,
        num_samples=min(args.num_calib_samples, 8),
        seq_len=args.calib_seq_len,
    )

    # Build acts.pt in deepcompressor format
    acts_state = {}
    for i in range(num_layers):
        k_key = f"model.layers.{i}.self_attn.k_rotary_emb.output"
        v_key = f"model.layers.{i}.self_attn.v_proj.output"
        acts_state[k_key] = {
            "dynamic_range": [{"max": torch.tensor(k_maxes[i])}]
        }
        acts_state[v_key] = {
            "dynamic_range": [{"max": torch.tensor(v_maxes[i])}]
        }

    # Save
    print(f"Saving to {args.output_dir}...")
    torch.save(model_state, os.path.join(args.output_dir, "model.pt"))
    torch.save(fixed_scale_state, os.path.join(args.output_dir, "scale.pt"))
    torch.save(acts_state, os.path.join(args.output_dir, "acts.pt"))

    print(f"Done! Output files:")
    for f in ["model.pt", "scale.pt", "acts.pt"]:
        fpath = os.path.join(args.output_dir, f)
        size_gb = os.path.getsize(fpath) / 1024**3
        print(f"  {f}: {size_gb:.2f} GB")

    print(f"\nNext: run checkpoint_converter.py:")
    print(f"  cd scripts/ckpt_converter")
    print(f"  python checkpoint_converter.py \\")
    print(f"    --model-path {args.model_path} \\")
    print(f"    --quant-path {args.output_dir} \\")
    print(f"    --w-bit 8 --group-size -1 --device cpu --kv-per-tensor")


if __name__ == "__main__":
    main()
