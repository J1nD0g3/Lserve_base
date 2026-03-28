"""
RULER benchmark metrics.
Adapted from ShadowKV (ByteDance) metrics for use with LServe.
"""

import re
import string
from collections import Counter


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
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
    predict_str = predict_str.strip()
    for tok in ['<|eot_id|>', '</s>', '</s', '</', '<|im_end|>', '<|endoftext|>']:
        predict_str = predict_str.replace(tok, '')
    np_pattern = re.compile(r'[\x00-\x1f]')
    predict_str = np_pattern.sub('\n', predict_str).strip()
    return predict_str


def needle_score(prediction, ground_truth):
    assert isinstance(prediction, str)
    assert isinstance(ground_truth, str)
    prediction = normalize_answer(postprocess_pred(prediction))
    ground_truth = normalize_answer(ground_truth)
    min_length = len(ground_truth)
    score = float(prediction[:min_length] == ground_truth[:min_length])
    pred_list = prediction.split()
    score = max(float(ground_truth in pred_list), score)
    return score


def multi_number(prediction: str, ground_truth: list) -> float:
    prediction = normalize_answer(prediction)
    prediction_list = re.findall(r'\d+', prediction)
    hits = [item for item in ground_truth if item in prediction_list]
    return len(hits) / len(ground_truth) if ground_truth else 0.0


def multi_words(prediction: str, ground_truth: list) -> float:
    prediction = prediction.lower()
    ground_truth = [gt.lower() for gt in ground_truth]
    prediction_list = re.findall(r'\b\w+\b', prediction)
    hits = [item for item in ground_truth if item in prediction_list]
    return len(hits) / len(ground_truth) if ground_truth else 0.0


def string_match_part(preds, refs):
    preds = postprocess_pred(preds)
    if not preds:
        return 0.0
    if isinstance(refs, str):
        refs = [refs]
    score_ref_in_pred = max([1.0 if r.lower() in preds.lower() else 0.0 for r in refs])
    score_pred_in_ref = max([1.0 if preds.lower() in r.lower() else 0.0 for r in refs])
    return round(max(score_ref_in_pred, score_pred_in_ref), 2)


# Task name -> metric function mapping
RULER_TASK_METRICS = {
    "niah_single_1": ("needle_score", needle_score),
    "niah_single_2": ("needle_score", needle_score),
    "niah_multikey_1": ("needle_score", needle_score),
    "niah_multikey_2": ("needle_score", needle_score),
    "niah_multivalue": ("needle_score", needle_score),
    "niah_multiquery": ("needle_score", needle_score),
    "vt": ("multi_words", multi_words),
    "fwe": ("multi_words", multi_words),
    "qa_1": ("string_match_part", string_match_part),
    "qa_2": ("string_match_part", string_match_part),
}

# Task grouping
RULER_GROUPS = {
    "NIAH Single": ["niah_single_1", "niah_single_2"],
    "NIAH Multi": ["niah_multikey_1", "niah_multikey_2", "niah_multivalue", "niah_multiquery"],
    "Aggregation": ["vt", "fwe"],
    "QA": ["qa_1", "qa_2"],
}


def compute_ruler_score(prediction, ground_truth, task_name):
    """Compute RULER metric for a single sample."""
    metric_name, metric_fn = RULER_TASK_METRICS[task_name]

    # needle_score expects single string ground_truth
    if metric_name == "needle_score":
        if isinstance(ground_truth, list):
            return max(metric_fn(prediction, gt) for gt in ground_truth)
        return metric_fn(prediction, ground_truth)
    elif metric_name in ("multi_number", "multi_words"):
        if isinstance(ground_truth, str):
            ground_truth = [ground_truth]
        return metric_fn(prediction, ground_truth)
    elif metric_name == "string_match_part":
        return metric_fn(prediction, ground_truth)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")
