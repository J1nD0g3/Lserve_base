from typing import Dict, List, Optional
import torch
import omniserve_backend.fused_attention_fine_grained_dense as fused_attention_fine_grained_dense
import omniserve_backend.fused_attention_per_tensor_dense as fused_attention_per_tensor_dense
import omniserve_backend.fused_attention_ctx_pool as fused_attention_ctx_pool

class ApplyBiasRopeUpdateKVCacheWrapper(torch.nn.Module):
    def __init__(
        self,
        layer_idx: int,
        num_heads: int, num_kv_heads: int, tokens_per_block: int,
        head_dim: int, rope_theta: float, rope_scaling: Dict,
        max_position_embeddings: int, neox_rotary_style: bool,
        kv_quant_granularity: str,
        kv_cache_config: Dict,
        use_int8: bool,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.tokens_per_block = tokens_per_block
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        if self.rope_scaling is not None:
            self.rope_scaling_factor = self.rope_scaling["factor"]
            assert self.rope_scaling["type"] in ("linear", "yarn"), f"Unsupported rope scaling type {self.rope_scaling['type']}"
        else:
            self.rope_scaling_factor = 1.0
        import os as _os
        if _os.environ.get("LSERVE_ROPE_DEBUG") == "1" and layer_idx == 0:
            print(f"[rope-debug][ctx] layer0 theta={rope_theta} scale_factor={self.rope_scaling_factor} max_pos={max_position_embeddings}", flush=True)
        self.max_position_embeddings = max_position_embeddings
        self.neox_rotary_style = neox_rotary_style
        self.kv_quant_granularity = kv_quant_granularity
        self.kv_cache_config = kv_cache_config
        self.use_int8 = use_int8

        # NOTE (Shang): dense or sparse does not matter here, since we are using the same kernel
        # Howerver, per-tensor or fine-grained does matter.
        if kv_quant_granularity == "per_tensor":
            self.forward = self.forward_per_tensor
        elif kv_quant_granularity == "fine_grained":
            self.forward = self.forward_fine_grained
        else:
            raise NotImplementedError(f"Unsupported kv_quant_granularity {kv_quant_granularity}")


    @torch.no_grad()
    def forward_per_tensor(
        self, 
        qkv_proj_act_buffer, 
        input_metadata,
        retrieval_head_flags,
        head_rank_table,
        sink_size, local_size, sink_blocks, local_blocks,
        num_retrieval_kv_heads, num_streaming_kv_heads,
        kv_scale_orig_quant = None
    ):
        if kv_scale_orig_quant is not None:
            kv_scale_orig_quant = kv_scale_orig_quant.float()

        size_per_retrieval_token = num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        size_per_streaming_token = num_streaming_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)

        fused_attention_per_tensor_dense.apply_bias_rope_update_kv_cache(
            qkv_proj_act_buffer,
            kv_scale_orig_quant,
            input_metadata.retrieval_context_lens,
            input_metadata.streaming_context_lens,
            input_metadata.padding_offsets,  # size [batch_size, max_seq_len]
            input_metadata.retrieval_block_tables[self.layer_idx],
            input_metadata.streaming_block_tables[self.layer_idx],
            retrieval_head_flags,
            head_rank_table,
            self.num_heads,
            self.num_kv_heads,
            input_metadata.max_seq_len,
            self.tokens_per_block,
            size_per_retrieval_token, 
            size_per_streaming_token,
            sink_size, local_size,
            sink_blocks, local_blocks,
            num_retrieval_kv_heads,
            num_streaming_kv_heads,
            self.head_dim,
            self.rope_theta,
            self.rope_scaling_factor,
            self.max_position_embeddings,
            self.neox_rotary_style,
            self.kv_cache_config["INT4_ENABLED"],  # int4_kv
            self.kv_cache_config["ZEROS_ENABLED"],  # kv_cache_with_zeros
        )


    @torch.no_grad()
    def forward_fine_grained(
        self, 
        qkv_proj_act_buffer, 
        input_metadata,
        retrieval_head_flags,
        head_rank_table,
        sink_size, local_size, sink_blocks, local_blocks,
        num_retrieval_kv_heads, num_streaming_kv_heads,
        kv_scale_orig_quant = None  # Of no use, only keep for interface consistency
    ):
        size_per_retrieval_token = num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        size_per_streaming_token = num_streaming_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)

        fused_attention_fine_grained_dense.apply_bias_rope_update_kv_cache(
            qkv_proj_act_buffer,
            input_metadata.retrieval_context_lens,
            input_metadata.streaming_context_lens,
            input_metadata.padding_offsets,  # size [batch_size, max_seq_len]
            input_metadata.retrieval_block_tables[self.layer_idx],
            input_metadata.streaming_block_tables[self.layer_idx],
            retrieval_head_flags,
            head_rank_table,
            self.num_heads,
            self.num_kv_heads,
            input_metadata.max_seq_len,
            self.tokens_per_block,
            size_per_retrieval_token, 
            size_per_streaming_token,
            sink_size, local_size,
            sink_blocks, local_blocks,
            num_retrieval_kv_heads,
            num_streaming_kv_heads,
            self.head_dim,
            self.rope_theta,
            self.rope_scaling_factor,
            self.max_position_embeddings,
            self.neox_rotary_style,
            self.kv_cache_config["INT4_ENABLED"],   # int4_kv
            self.kv_cache_config["ZEROS_ENABLED"],  # kv_cache_with_zeros
        )

class PagedMinMaxPoolWrapper(torch.nn.Module):
    def __init__(
        self,
        layer_idx: int,
        tokens_per_block: int,
        sub_chunk_per_block: int,
        head_dim: int,
        kv_cache_config: Dict,
        use_int8: bool,
    ):
        super().__init__()

        self.layer_idx = layer_idx
        self.tokens_per_block = tokens_per_block
        self.sub_chunk_per_block = sub_chunk_per_block
        self.tokens_per_sub_chunk = tokens_per_block // sub_chunk_per_block
        self.head_dim = head_dim
        self.kv_cache_config = kv_cache_config
        self.use_int8 = use_int8
        assert tokens_per_block % sub_chunk_per_block == 0, f"tokens_per_block {tokens_per_block} must be divisible by sub_chunk_per_block {sub_chunk_per_block}"

    @torch.no_grad()
    def forward(
        self, 
        keys, 
        input_metadata,
        pooling_heads_idx,
        num_retrieval_kv_heads
    ):
        size_per_retrieval_token = num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        
        fused_attention_ctx_pool.paged_min_max_pool(
            keys,
            input_metadata.retrieval_block_tables[self.layer_idx],
            input_metadata.cu_seqlens,
            pooling_heads_idx,
            input_metadata.max_seq_len,
            self.tokens_per_sub_chunk,
            self.tokens_per_block,
            size_per_retrieval_token,
            True # self.kv_cache_config["ZEROS_ENABLED"]    # TODO: Fix this error for buffer offset.
        )

