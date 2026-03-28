import glob



def add_lbench_args(parser):
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--quant_model", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", required=True)
    parser.add_argument("--method", type=str, default="full")
    parser.add_argument("--model_name_suffix", type=str, default=None, help="name of model")
    parser.add_argument("--enable-thinking", action="store_true", default=False,
                        help="Enable Qwen3 thinking mode (default: off)")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples per task (0 = all). Useful for quick testing.")


# for lServe
import argparse
from typing import List, Tuple
import random

import datasets

import omniserve.utils.constants
from omniserve import EngineArgs, LLMEngine, SamplingParams
from omniserve.conversation import get_conv_template_name, get_conv_template

def _initialize_engine(args: argparse.Namespace) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    engine_args = EngineArgs.from_cli_args(args)
    return LLMEngine.from_engine_args(engine_args)

def initialize_engine(model) -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    parser = argparse.ArgumentParser(description="Demo on using the LLMEngine class directly")
    parser = EngineArgs.add_cli_args(parser)
    args, _ = parser.parse_known_args()
    args.model = model
    return _initialize_engine(args), args

max_seq_len = omniserve.utils.constants.max_seq_len
BG_BLUE = "\033[44m"
BG_GREEN = "\033[42m"
BG_PINK = "\033[45m"
RESET = "\033[0m"

_global_request_id = 0

def process_requests(engine: LLMEngine, test_prompts: List[str], stop_token_ids, max_gen_length: int=512):
    """Continuously process a list of prompts and handle the outputs."""
    global _global_request_id
    request_id_start = _global_request_id
    request_id = _global_request_id
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1.0, stop_token_ids=stop_token_ids, max_tokens=max_gen_length
    )
    test_prompts = [(test_prompt, sampling_params) for test_prompt in test_prompts]
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt, sampling_params = test_prompts.pop(0)
            succeeded = engine.add_request(str(request_id), prompt, sampling_params)
            if succeeded:
                request_id += 1
            else:
                pass
        num_test_prompts = request_id

        if not test_prompts:
            break

    if engine.ifb_mode == False:
        # We need to pre-caulcate the block table size for initialization
        block_size = engine.cache_config.block_size
        max_context_length = 128
        max_gen_length = 384
        tot_length = (
            max_context_length + max_gen_length
        )  # Set the upper bound for (prompt + gen) length
        init_num_blocks = (tot_length + block_size - 1) // block_size
        engine.update_init_num_blocks(init_num_blocks)

    # seq_group_metadata_list, scheduler_outputs = engine.step()
    iter = 1
    finished = 0
    outputs = {}
    import sys
    while engine.has_unfinished_requests():
        ### Schedule iteration 1 (context stage)
        requests_outputs = engine.step()
        if len(requests_outputs) == 0:
            break
        # print(
        #     BG_BLUE
        #     + "*" * 5
        #     + "Iteration %d (remaining req.s = %d)"
        #     % (iter, len(requests_outputs) + len(engine.scheduler.waiting))
        #     + "*" * 5
        #     + RESET
        # )
        for request_output in requests_outputs:
            if request_output["finished"]:
                finished += 1
                outputs[request_output['id']] = request_output['text']
                # print(
                #     f"{BG_GREEN}[Conversation {request_output['id']} output]{RESET} {request_output['text']}"
                # )
        iter += 1
        if engine.ifb_mode == False:
            if iter == max_gen_length:  # Early exit
                # for request_output in requests_outputs:
                    # print(
                    #     f"{BG_GREEN}[Conversation {request_output['id']} output]{RESET} {request_output['tokens']}"
                    # )
                break
    assert num_test_prompts - request_id_start == finished
    _global_request_id = request_id

    # Remap to 0-based index for caller
    result = {}
    for i in range(finished):
        key = request_id_start + i
        # Engine may return id as int or str
        if key in outputs:
            result[i] = outputs[key]
        elif str(key) in outputs:
            result[i] = outputs[str(key)]
    return result

