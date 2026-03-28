import os
import re
import sys
import warnings
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from tqdm import tqdm
import numpy as np
import random
import argparse

# Suppress engine init noise from stdout — redirect to stderr during import & init
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

from utils import add_lbench_args, initialize_engine, process_requests


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name, enable_thinking=False):
    if "llama-2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "qwen3" in model_name.lower() or "Qwen3" in model_name:
        if enable_thinking:
            prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"
        else:
            prompt = f"<|im_start|>user\n{prompt}\n/no_think<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def post_process(response, model_name, enable_thinking=False):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    elif "llama-3" in model_name.lower():
        response = (
            response.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .strip()
        )
    elif "Llama-2-7B-32K-Instruct" in model_name:
        response = (
            response.split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("(Passage")[0]
            .strip()
        )
    elif "qwen3" in model_name.lower() or "Qwen3" in model_name:
        # Strip <think>...</think> block (always, in case model still produces it)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        response = (
            response.split("<|im_end|>")[0]
            .split("<|endoftext|>")[0]
            .split("\n\nQuestion")[0]
            .strip()
        )
    return response


def get_pred(
    lserve_engine,
    tokenizer,
    eos_token_ids,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    base_model_name,
    enable_thinking=False,
    pbar=None,
):
    preds = []
    for idx, json_obj in enumerate(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, base_model_name, enable_thinking)

        # NOTE (Shang): Fix by adding eos_token_ids
        outputs = process_requests(lserve_engine, [prompt], eos_token_ids, max_gen)
        pred = outputs[0]
        pred = pred.replace("<|eot_id|>", "")
        pred = post_process(pred, base_model_name, enable_thinking)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
        if pbar is not None:
            pbar.update(1)
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    generation_config = GenerationConfig.from_pretrained(model_path)
    lserve_engine, lserve_args = initialize_engine(args.model_path)
    
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    return lserve_engine, tokenizer, eos_token_ids


if __name__ == "__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser()
    add_lbench_args(parser)
    args, _ = parser.parse_known_args()
    # model2path = json.load(open("eval/LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("./config/model2maxlen.json", "r"))
    base_model_name = args.base_model
    quant_model_name = args.quant_model
    # define your model
    lserve_engine, tokenizer, eos_token_ids = load_model_and_tokenizer(args.model_path)

    max_length = model2maxlen[base_model_name]
    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = args.task.split("+")
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("./config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("./config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("./pred"):
        os.makedirs("./pred")
    if not os.path.exists("./pred_e"):
        os.makedirs("./pred_e")
    import time
    # Suppress datasets library cache messages
    logging.getLogger("datasets").setLevel(logging.ERROR)
    for dataset in datasets:
        data = load_dataset("THUDM/LongBench", dataset, split="test")
        if args.max_samples > 0:
            data = data.select(range(min(args.max_samples, len(data))))
        os.makedirs(f"./pred/{quant_model_name}", exist_ok=True)
        out_path = f"./pred/{quant_model_name}/{dataset}-{args.model_name_suffix}.jsonl"
        if os.path.exists(out_path):
            continue
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]

        t0 = time.time()
        pbar = tqdm(total=len(data), desc=f"Running longbench/{dataset}", dynamic_ncols=True, file=sys.stdout)
        preds = get_pred(
            lserve_engine,
            tokenizer,
            eos_token_ids,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            base_model_name,
            enable_thinking=args.enable_thinking,
            pbar=pbar,
        )
        pbar.close()
        elapsed = time.time() - t0
        print(f"  {dataset}: {len(data)} samples, {elapsed:.1f}s")

        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")