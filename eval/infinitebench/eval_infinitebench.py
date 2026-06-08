"""InfiniteBench evaluation for LServe (generation-based scoring).

Prompts, truncation and metrics ported verbatim from ShadowKV
(data/dataset.py, data/metrics.py) for cross-repo comparability.

Usage:
  INFINITEBENCH_DIR=/workspace/data/InfiniteBench \
  python eval/infinitebench/eval_infinitebench.py \
      --model_path /workspace/models/Qwen3-14B-128k --datalen 122880 --enable-thinking \
      [LServe engine args: --precision w16a16kv8 --kv-quant-granularity fine_grained ...]
"""
import os
import re
import sys
import json
import time
import string
import argparse
from collections import Counter

import torch
import numpy as np
import random
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from omniserve import EngineArgs, LLMEngine, SamplingParams
from transformers import AutoTokenizer, GenerationConfig

# ============================================================
# Ported verbatim from ShadowKV data/dataset.py
# ============================================================

INFINITEBENCH_PROMPTS = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is",
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is",
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below.\n\n{context}\n\n{input}",
    "longbook_sum_eng": "Summarize the book below.\n\n{context}\n\nSummary:",
    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",
    "longbook_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",
    "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n{context}\n\n问题：{question}\n答案：",
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:",
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\n{context}\n\nThe name that has been replaced with $$MASK$$ is likely",
}

INFINITEBENCH_GEN_LEN = {
    "passkey": 30,
    "number_string": 50,
    "kv_retrieval": 50,
    "longbook_sum_eng": 1200,
    "longbook_qa_eng": 40,
    "longbook_qa_chn": 40,
    "longdialogue_qa_eng": 40,
    "math_find": 30,
    "code_debug": 30,
    "longbook_choice_eng": 30,
}


# ============================================================
# Ported verbatim from ShadowKV data/metrics.py
# ============================================================

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def postprocess_pred(predict_str: str):

    predict_str = predict_str.strip().replace('<|eot_id|>', '').replace('</s>', '').replace('</s', '').replace('</', '')

    # Remove all non-printable characters
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()

    return predict_str


def _f1_score(prediction, ground_truth):
    """Token-level F1 score."""
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def rouge_score(prediction, ground_truth):
    """ROUGE-L F1 score for summarization tasks."""
    try:
        from rouge import Rouge
    except ImportError:
        raise ImportError("Please install rouge: pip install rouge")
    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def infinitebench_metric(prediction, ground_truth, task_name):
    """Unified InfiniteBench metric dispatcher."""
    prediction = postprocess_pred(prediction).strip()

    if task_name in ('passkey', 'number_string', 'kv_retrieval'):
        # Exact substring match
        answer = str(ground_truth).strip()
        return 1.0 if answer in prediction else 0.0

    elif task_name == 'longbook_qa_eng':
        # F1 score
        gts = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        best = 0.0
        for gt in gts:
            pred_tokens = normalize_answer(prediction).split()
            gt_tokens = normalize_answer(gt).split()
            if pred_tokens and gt_tokens:
                best = max(best, _f1_score(pred_tokens, gt_tokens))
        return best

    elif task_name == 'longbook_qa_chn':
        # Chinese F1 score with jieba
        try:
            import jieba
        except ImportError:
            return 0.0
        gts = ground_truth if isinstance(ground_truth, list) else [ground_truth]
        cn_punc = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‟…‧﹏."
        all_punc = set(string.punctuation + cn_punc)

        def _norm_zh(s):
            return "".join(ch for ch in s.lower() if ch not in all_punc).replace(" ", "")

        best = 0.0
        for gt in gts:
            pred_tokens = [_norm_zh(t) for t in jieba.cut(prediction, cut_all=False)]
            gt_tokens = [_norm_zh(t) for t in jieba.cut(gt, cut_all=False)]
            pred_tokens = [t for t in pred_tokens if t]
            gt_tokens = [t for t in gt_tokens if t]
            if pred_tokens and gt_tokens:
                best = max(best, _f1_score(pred_tokens, gt_tokens))
        return best

    elif task_name in ('longbook_choice_eng', 'code_debug'):
        # Multiple choice: check A/B/C/D
        answer = ground_truth
        if isinstance(answer, list):
            correct_letter = answer[1] if len(answer) > 1 else answer[0]
            correct_text = answer[0]
        else:
            correct_letter = answer
            correct_text = answer
        pred_upper = prediction.upper().strip()
        if correct_letter.upper() in pred_upper[:5]:
            return 1.0
        if correct_text.lower() in prediction.lower():
            return 1.0
        return 0.0

    elif task_name == 'longbook_sum_eng':
        # ROUGE-L
        if not prediction or not str(ground_truth):
            return 0.0
        return rouge_score(prediction, str(ground_truth))

    elif task_name == 'longdialogue_qa_eng':
        # Character name match
        answer = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
        return 1.0 if answer.lower() in prediction.lower() else 0.0

    elif task_name == 'math_find':
        # First integer match
        answer = str(ground_truth).strip()
        pred_nums = re.split(r"[^0-9]", prediction)
        for item in pred_nums:
            if item:
                return 1.0 if item == answer else 0.0
        return 0.0

    else:
        # Fallback: substring match
        return 1.0 if str(ground_truth).lower() in prediction.lower() else 0.0


# ============================================================
# Prompt building with ShadowKV-style middle truncation
# ============================================================

def wrap_qwen3_chat(prompt, enable_thinking):
    if enable_thinking:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def fill_template(task, template, sample, context):
    if task == 'code_debug':
        return template.format(
            context=context,
            OPTION_A=sample['options'][0], OPTION_B=sample['options'][1],
            OPTION_C=sample['options'][2], OPTION_D=sample['options'][3],
        )
    elif task == 'longbook_choice_eng':
        return template.format(
            context=context, question=sample['input'],
            OPTION_A=sample['options'][0], OPTION_B=sample['options'][1],
            OPTION_C=sample['options'][2], OPTION_D=sample['options'][3],
        )
    elif task in ('longbook_qa_eng', 'longbook_qa_chn'):
        return template.format(context=context, question=sample['input'])
    elif task in ('longbook_sum_eng', 'longdialogue_qa_eng'):
        return template.format(context=context)
    elif task == 'math_find':
        prompt = sample['input']
        find_result = re.findall(r"The .+ of", prompt)
        assert find_result, f"Cannot find target number in: {prompt}"
        target_number = find_result[0].lower()[:-3]
        prefix = f"What is {target_number} in the following list?"
        return template.format(prefix=prefix, context=context, input=prompt)
    else:
        # passkey, number_string, kv_retrieval
        return template.format(context=context, input=sample['input'])


def build_prompt(tokenizer, task, sample, datalen, enable_thinking):
    """Returns (chat_text, input_len)."""
    template = INFINITEBENCH_PROMPTS[task]
    context = sample['context']
    prompt_text = fill_template(task, template, sample, context)
    chat_text = wrap_qwen3_chat(prompt_text, enable_thinking)
    input_ids = tokenizer.encode(chat_text, add_special_tokens=False)
    if len(input_ids) > datalen:
        empty_prompt = fill_template(task, template, sample, '')
        overhead_ids = tokenizer.encode(
            wrap_qwen3_chat(empty_prompt, enable_thinking), add_special_tokens=False
        )
        max_ctx_tokens = datalen - len(overhead_ids)
        ctx_ids = tokenizer.encode(context, add_special_tokens=False)
        if len(ctx_ids) > max_ctx_tokens:
            half = max_ctx_tokens // 2
            ctx_ids = ctx_ids[:half] + ctx_ids[-half:]
        truncated_ctx = tokenizer.decode(ctx_ids, skip_special_tokens=False)
        prompt_text = fill_template(task, template, sample, truncated_ctx)
        chat_text = wrap_qwen3_chat(prompt_text, enable_thinking)
        input_ids = tokenizer.encode(chat_text, add_special_tokens=False)
    return chat_text, len(input_ids)


def get_ground_truth(task, sample):
    if task in ('code_debug', 'longbook_choice_eng'):
        OPTIONS = "ABCD"
        answer = sample['answer']
        if isinstance(answer, str):
            return [answer, OPTIONS[sample['options'].index(answer)]]
        elif isinstance(answer, list):
            if len(answer) == 1:
                return [answer[0], OPTIONS[sample['options'].index(answer[0])]]
            return answer
        return answer
    return sample['answer']


# ============================================================
# Generation
# ============================================================

def initialize_lserve_engine(model_path):
    parser = argparse.ArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    ns, _ = parser.parse_known_args()
    ns.model = model_path
    engine_args = EngineArgs.from_cli_args(ns)
    return LLMEngine.from_engine_args(engine_args)


_global_request_id = 0


def generate_with_lserve(engine, prompt, stop_token_ids, max_gen_len):
    global _global_request_id
    rid = _global_request_id
    _global_request_id += 1
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, stop_token_ids=stop_token_ids, max_tokens=max_gen_len
    )
    engine.add_request(str(rid), prompt, sampling_params)
    outputs = {}
    while engine.has_unfinished_requests():
        for out in engine.step():
            if out["finished"]:
                outputs[out["id"]] = out["text"]
    text = outputs.get(rid, outputs.get(str(rid), ""))
    return text.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()


def strip_thinking(text):
    if '</think>' in text:
        return text.split('</think>')[-1].strip()
    if '<think>' in text:
        return ''
    return text


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)


ALL_TASKS = list(INFINITEBENCH_PROMPTS.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--enable-thinking", action="store_true", default=False)
    parser.add_argument("--keep_ratio", type=float, default=None,
                        help="overall decode KV keep ratio to match across methods (e.g. 0.27)")
    parser.add_argument("--datalen", type=int, default=122880)
    parser.add_argument("--max-gen", type=int, default=0,
                        help="max new tokens (0 = per-task default, or context-capped with thinking)")
    parser.add_argument("--tasks", type=str, default=",".join(ALL_TASKS))
    parser.add_argument("--max-samples", type=int, default=100,
                        help="max samples per task (0 = all)")
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

    with open(os.path.join(args.model_path, "config.json")) as f:
        max_position = json.load(f).get("max_position_embeddings", 40960)
    assert args.datalen < max_position, f"datalen {args.datalen} must be < {max_position}"

    infinitebench_dir = os.environ.get('INFINITEBENCH_DIR', '/workspace/data/InfiniteBench')
    model_short = os.path.basename(args.model_path.rstrip("/"))
    think_tag = "think_on" if args.enable_thinking else "think_off"
    run_tag = think_tag if not args.run_tag else f"{args.run_tag}-{think_tag}"
    out_dir = args.output_dir or os.path.join(os.path.dirname(__file__), "pred", model_short)
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained(args.model_path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    engine = initialize_lserve_engine(args.model_path)

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    all_summaries = {}
    for task in task_list:
        assert task in INFINITEBENCH_PROMPTS, f"Unknown task {task}"
        out_path = os.path.join(out_dir, f"infinitebench_{task}-{run_tag}.jsonl")

        done_idx = set()
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        done_idx.add(json.loads(line)["idx"])
                    except (json.JSONDecodeError, KeyError):
                        pass

        data = load_dataset("json", data_files=f'{infinitebench_dir}/{task}.jsonl', split='train')
        num_samples = len(data) if args.max_samples <= 0 else min(args.max_samples, len(data))
        print(f"[{task}] {num_samples} samples ({len(done_idx)} done) -> {out_path}", flush=True)

        for i in tqdm(range(num_samples), desc=task):
            if i in done_idx:
                continue
            sample = data[i]
            chat_text, input_len = build_prompt(tokenizer, task, sample, args.datalen, args.enable_thinking)
            gt = get_ground_truth(task, sample)
            set_lserve_keep_ratio(engine, input_len, args.keep_ratio)

            max_gen = args.max_gen
            if max_gen <= 0:
                if args.enable_thinking:
                    max_gen = max_position - input_len
                else:
                    max_gen = INFINITEBENCH_GEN_LEN[task]
            max_gen = min(max_gen, max_position - input_len)

            t0 = time.time()
            raw_text = generate_with_lserve(engine, chat_text, eos_token_ids, max_gen)
            elapsed = time.time() - t0

            pred_text = strip_thinking(raw_text)
            score = infinitebench_metric(pred_text, gt, task)

            record = {
                "idx": i,
                "prediction": pred_text,
                "ground_truth": gt,
                "score": score,
                "input_len": input_len,
                "elapsed_sec": round(elapsed, 2),
            }
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        records = [json.loads(l) for l in open(out_path, encoding="utf-8")]
        if records:
            avg = sum(r["score"] for r in records) / len(records)
            all_summaries[task] = {"n": len(records), "score": round(avg, 4)}
            print(f"[{task}] n={len(records)} score={avg:.4f}", flush=True)

    summary = {"run": run_tag, "model": model_short, "datalen": args.datalen, "tasks": all_summaries}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(os.path.join(out_dir, f"infinitebench-{run_tag}.summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
