import torch
from itertools import count
from block_sparse_attn import (
    token_streaming_attn_func,
    block_streaming_attn_func,
)
from flash_attn import flash_attn_varlen_func

def attention_wrapper(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0, causal=True,
    head_mask_type = None,
    streaming_info = None,
):
    if head_mask_type is not None and streaming_info is not None:
        attn_output = token_static_sparse_attn(
                        q_unpad, k_unpad, v_unpad,
                        cu_seqlens_q, cu_seqlens_k,
                        max_seqlen_q, max_seqlen_k,
                        head_mask_type, streaming_info
                    )
    else: #dense case
        attn_output = dense_context_attn(
                        q_unpad, k_unpad, v_unpad,
                        cu_seqlens_q, cu_seqlens_k,
                        max_seqlen_q, max_seqlen_k,
                        dropout_p, causal
                    )
    return attn_output
        
def dense_context_attn(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_p, causal
):
    attn_output = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                dropout_p=dropout_p,
                causal=causal,
            )
    return attn_output

def block_static_sparse_attn(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    head_mask_type, streaming_info
):
    attn_output = block_streaming_attn_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens_q, cu_seqlens_k,
                head_mask_type, streaming_info,
                max_seqlen_q, max_seqlen_k,
            )
    return attn_output

def token_static_sparse_attn(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    head_mask_type, streaming_info
):
    attn_output = token_streaming_attn_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens_q, cu_seqlens_k,
                head_mask_type, streaming_info,
                max_seqlen_q, max_seqlen_k,
            )
    return attn_output