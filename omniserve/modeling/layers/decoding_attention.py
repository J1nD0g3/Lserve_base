from typing import Dict, List, Optional

import os
import torch
import omniserve_backend.fused_attention_fine_grained_dense as fused_attention_fine_grained_dense
import omniserve_backend.fused_attention_fine_grained_sparse as fused_attention_fine_grained_sparse
import omniserve_backend.fused_attention_per_tensor_dense as fused_attention_per_tensor_dense
import omniserve_backend.fused_attention_per_tensor_sparse as fused_attention_per_tensor_sparse
import omniserve_backend.fused_attention_selector as fused_attention_selector
import omniserve_backend.fused_attention_pure_dense as fused_attention_pure_dense


class DecodingAttentionWrapper(torch.nn.Module):
    def __init__(
        self,
        layer_idx: int,
        static_sparsity_enabled: bool,
        head_dim: int,
        alibi_slopes: int,
        memory_max_len: int,
        tokens_per_block: int,
        rotary_embedding_dim: int,
        rotary_base: int, 
        rope_scaling: Dict,
        neox_rotary_style: bool,
        kv_quant_granularity: str,
        kv_cache_config: Dict,
        use_int8: bool,
        sparse_decode_mode: int,
        sub_chunk_size: int,
        dynamic_sparse_token_budget: int,
        multiblock_switch: int,
        selector_update_interval: int,
        ):
        super().__init__()

        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.alibi_slopes = alibi_slopes
        self.memory_max_len = memory_max_len
        self.tokens_per_block = tokens_per_block
        self.rotary_embedding_dim = rotary_embedding_dim
        self.rotary_base = rotary_base
        self.rope_scaling = rope_scaling
        if self.rope_scaling is not None:
            self.rope_scaling_factor = self.rope_scaling["factor"]
            assert self.rope_scaling["type"] in ("linear", "yarn"), f"Unsupported rope scaling type {self.rope_scaling['type']}"
        else:
            self.rope_scaling_factor = 1.0
        if os.environ.get("LSERVE_ROPE_DEBUG") == "1" and layer_idx == 0:
            print(f"[rope-debug][dec] layer0 base={rotary_base} scale_factor={self.rope_scaling_factor} memory_max_len={memory_max_len} rotary_dim={rotary_embedding_dim}", flush=True)
        self.neox_rotary_style = neox_rotary_style
        self.kv_quant_granularity = kv_quant_granularity
        self.kv_cache_config = kv_cache_config
        self.use_int8 = use_int8

        self.sparse_decode_mode = sparse_decode_mode
        self.sub_chunk_size = sub_chunk_size
        self.dynamic_sparse_token_budget = dynamic_sparse_token_budget
        self.multiblock_switch = multiblock_switch
        self.selector_update_interval = selector_update_interval
        
        if self.sparse_decode_mode != 0:
            if kv_quant_granularity == "per_tensor":
                self.forward = self.forward_w_dynamic_sparse_per_tensor
            elif kv_quant_granularity == "fine_grained":
                self.forward = self.forward_w_dynamic_sparse_fine_grained
            else:
                raise NotImplementedError(f"Unsupported kv_quant_granularity {kv_quant_granularity}")
        else:
            if static_sparsity_enabled:
                if kv_quant_granularity == "per_tensor":
                    self.forward = self.forward_wo_dynamic_sparse_per_tensor
                elif kv_quant_granularity == "fine_grained":
                    self.forward = self.forward_wo_dynamic_sparse_fine_grained
                else:
                    raise NotImplementedError(f"Unsupported kv_quant_granularity {kv_quant_granularity}")
            else:
                if kv_quant_granularity == "fine_grained":
                    # NOTE: fused_attention_pure_dense.single_query_attention has no multiblock
                    # support (smem overflow / CUDA invalid argument beyond ~28k context) and
                    # ignores rope_scaling_factor (breaks YaRN). Route through the
                    # wo_dynamic_sparse kernel, which is identical full attention when all
                    # heads are retrieval heads (static_sparsity=0.0).
                    self.forward = self.forward_wo_dynamic_sparse_fine_grained
                elif kv_quant_granularity == "per_tensor":
                    # raise NotImplementedError("per_tensor kv_quant_granularity is not supported for pure dense attention")
                    self.forward = self.forward_wo_dynamic_sparse_per_tensor    # NOTE: Per_tensor pure dense is has not been implemented yet. Just use the forward_wo_dynamic_sparse_per_tensor sparse for now.
                else:
                    raise NotImplementedError(f"Unsupported kv_quant_granularity {kv_quant_granularity}")
                


    @torch.no_grad()
    def dynamic_select_topk_pages(
        self, 
        q, k, v,
        retrieval_block_tables, streaming_block_tables, retrieval_head_flags, head_rank_table,
        lengths_per_sample, sink_size, local_size, sink_blocks, local_blocks,
        size_per_retrieval_token, size_per_streaming_token,
        num_retrieval_kv_heads, num_streaming_kv_heads, timestep, hidden_dim_per_retrieval_token
    ):
        if timestep <= self.dynamic_sparse_token_budget:
            selected_page_idx = torch.range(0, timestep // self.tokens_per_block, device=q.device, dtype=torch.int32).unsqueeze(0).unsqueeze(0).expand(q.shape[0], q.shape[1], -1).contiguous()
        
        else:
            dynamic_sparse_token_budget = min(self.dynamic_sparse_token_budget, timestep)
            selected_page_stats = fused_attention_selector.single_query_page_selector(
                q,
                k,
                v,                               # Actually of no use. (just keep for the interface)
                retrieval_block_tables,
                streaming_block_tables,          # Actually of no use. (just keep for the interface)
                retrieval_head_flags,
                head_rank_table,
                None,                            # selected_page_idx: This is of no use (just keep for the interface)
                lengths_per_sample,
                self.alibi_slopes,
                self.memory_max_len,
                self.tokens_per_block,
                size_per_retrieval_token,
                size_per_streaming_token,        # Actually of no use. (just keep for the interface)
                sink_size, local_size,           # Actually of no use. (just keep for the interface)
                sink_blocks, local_blocks,       # Actually of no use. (just keep for the interface)
                num_retrieval_kv_heads, 
                num_streaming_kv_heads,          # Actually of no use. (just keep for the interface)
                timestep,                        # NOTE (shang): timestep is the length of history, not including the current token! 
                self.rotary_embedding_dim,
                self.rotary_base,
                self.rope_scaling_factor,
                self.neox_rotary_style,
                self.kv_cache_config["INT4_ENABLED"],
                True, # self.kv_cache_config["ZEROS_ENABLED"],     # TODO: Fix this error for buffer offset.
                self.sub_chunk_size,
                hidden_dim_per_retrieval_token,
                1000000,                         # const int multiblock_switch  # FIXME: Currently never activate it in page selector!
            )

            selected_page_stats = selected_page_stats.view(q.shape[0], q.shape[1], -1, self.tokens_per_block // self.sub_chunk_size)
            selected_page_stats = torch.max(selected_page_stats, dim=-1).values        # max over sub-chunk-dim


            total_page_num = selected_page_stats.size(-1)
            _, selected_page_idx = selected_page_stats[:,:,:-1].topk(
                k=(min(max(3, dynamic_sparse_token_budget // self.tokens_per_block), total_page_num) - 1), dim=-1
            )
            selected_page_idx = torch.cat([selected_page_idx, torch.ones_like(selected_page_idx[..., :1]) * (total_page_num - 1)], dim=-1).contiguous()      # Make sure the most recent page is chosen and concatenated in the last
            selected_page_idx = selected_page_idx.to(torch.int32)

        return selected_page_idx

    @torch.no_grad()
    def forward_pure_dense(
        self,
        q, k, v,
        input_metadata,
        retrieval_head_flags, head_rank_table,
        sink_size, local_size, sink_blocks, local_blocks,
        num_retrieval_kv_heads, num_streaming_kv_heads,
        cached_dynamic_sparse_page_idx,
        kv_scale_quant_orig,
    ):
        timestep = input_metadata.max_seq_len
        # Shang's important fix, but it might cause problem in the end of a block...
        lengths_per_sample = input_metadata.retrieval_context_lens  # + 1

        size_per_retrieval_token = num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        
        attn_output = fused_attention_pure_dense.single_query_attention(
                q,
                k,
                v,
                input_metadata.retrieval_block_tables[self.layer_idx],
                lengths_per_sample,
                self.alibi_slopes,
                self.memory_max_len,
                self.tokens_per_block,
                size_per_retrieval_token,
                timestep,
                self.rotary_embedding_dim,
                self.rotary_base,
                # self.rope_scaling_factor, # TODO: Fix rope scaling factor
                self.neox_rotary_style,
                self.kv_cache_config["INT4_ENABLED"],
                self.kv_cache_config["ZEROS_ENABLED"],
            )

        selected_page_idx = None
        return attn_output, selected_page_idx

    @torch.no_grad()
    def forward_wo_dynamic_sparse_per_tensor(
        self, 
        q, k, v,
        input_metadata,
        retrieval_head_flags, head_rank_table,
        sink_size, local_size, sink_blocks, local_blocks, 
        num_retrieval_kv_heads, num_streaming_kv_heads,
        cached_dynamic_sparse_page_idx,  # NOTE: cached_dynamic_sparse_page_idx is of no use. Just keep for the interface consistency.
        kv_scale_quant_orig,
    ):
        timestep = input_metadata.max_seq_len
        # Shang's important fix, but it might cause problem in the end of a block...
        lengths_per_sample = input_metadata.retrieval_context_lens  # + 1

        size_per_retrieval_token = num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        size_per_streaming_token = num_streaming_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)

        kv_scale_quant_orig = kv_scale_quant_orig.float()
        kv_scale_orig_quant = 1 / kv_scale_quant_orig
        
        attn_output = fused_attention_per_tensor_dense.single_query_attention(
            q,
            k,
            v,
            kv_scale_quant_orig,
            kv_scale_orig_quant,
            input_metadata.retrieval_block_tables[self.layer_idx],
            input_metadata.streaming_block_tables[self.layer_idx],
            retrieval_head_flags,
            head_rank_table,
            lengths_per_sample,
            self.alibi_slopes,
            self.memory_max_len,
            self.tokens_per_block,
            size_per_retrieval_token, 
            size_per_streaming_token,
            sink_size, local_size,
            sink_blocks, local_blocks,
            num_retrieval_kv_heads,
            num_streaming_kv_heads,
            timestep,
            self.rotary_embedding_dim,
            self.rotary_base,
            self.rope_scaling_factor,
            self.neox_rotary_style,
            self.kv_cache_config["INT4_ENABLED"],
            self.kv_cache_config["ZEROS_ENABLED"],
            2048,  # const int multiblock_switch
        )

        selected_page_idx = None
        return attn_output, selected_page_idx

    @torch.no_grad()
    def forward_w_dynamic_sparse_per_tensor(
        self,
        q, k, v,
        input_metadata,
        retrieval_head_flags, head_rank_table,
        sink_size, local_size, sink_blocks, local_blocks, 
        num_retrieval_kv_heads, num_streaming_kv_heads,
        cached_dynamic_sparse_page_idx,
        kv_scale_quant_orig,
    ):
        timestep = input_metadata.max_seq_len
        # Shang's important fix, but it might cause problem in the end of a block...
        lengths_per_sample = input_metadata.retrieval_context_lens  # + 1

        size_per_retrieval_token = num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        size_per_streaming_token = num_streaming_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        hidden_dim_per_retrieval_token = num_retrieval_kv_heads * self.head_dim

        if ((timestep) % self.selector_update_interval != 0) and cached_dynamic_sparse_page_idx is not None:      # Since timestep is the length of history, not including the current token. No need to -1 here.
            dynamic_sparse_page_idx = cached_dynamic_sparse_page_idx
        else:
            dynamic_sparse_page_idx = self.dynamic_select_topk_pages(    
                q, k, v,
                input_metadata.retrieval_block_tables[self.layer_idx],
                input_metadata.streaming_block_tables[self.layer_idx],
                retrieval_head_flags, head_rank_table,
                lengths_per_sample, sink_size, local_size, sink_blocks, local_blocks,
                size_per_retrieval_token, size_per_streaming_token,
                num_retrieval_kv_heads, num_streaming_kv_heads, timestep, hidden_dim_per_retrieval_token
            )
        
        kv_scale_quant_orig = kv_scale_quant_orig.float()
        kv_scale_orig_quant = 1 / kv_scale_quant_orig

        attn_output = fused_attention_per_tensor_sparse.single_query_attention(
            q,
            k,
            v,
            kv_scale_quant_orig,
            kv_scale_orig_quant,
            input_metadata.retrieval_block_tables[self.layer_idx],
            input_metadata.streaming_block_tables[self.layer_idx],
            retrieval_head_flags,
            head_rank_table,
            dynamic_sparse_page_idx,
            lengths_per_sample,
            self.alibi_slopes,
            self.memory_max_len,
            self.tokens_per_block,
            size_per_retrieval_token,
            size_per_streaming_token,
            sink_size, local_size,
            sink_blocks, local_blocks,
            num_retrieval_kv_heads, 
            num_streaming_kv_heads,
            timestep,
            self.rotary_embedding_dim,
            self.rotary_base,
            self.rope_scaling_factor,
            self.neox_rotary_style,
            self.kv_cache_config["INT4_ENABLED"],
            self.kv_cache_config["ZEROS_ENABLED"],
            self.sub_chunk_size,
            hidden_dim_per_retrieval_token,
            self.multiblock_switch,
        )

        return attn_output, dynamic_sparse_page_idx
    
    @torch.no_grad()
    def forward_wo_dynamic_sparse_fine_grained(
        self,
        q, k, v,
        input_metadata,
        retrieval_head_flags, head_rank_table,
        sink_size, local_size, sink_blocks, local_blocks, 
        num_retrieval_kv_heads, num_streaming_kv_heads,
        cached_dynamic_sparse_page_idx,
        kv_scale_quant_orig,        # NOTE: kv_scale_quant_orig and cached_dynamic_sparse_page_idx is of no use. Just keep for the interface consistency.
    ):
        timestep = input_metadata.max_seq_len
        # Shang's important fix, but it might cause problem in the end of a block...
        lengths_per_sample = input_metadata.retrieval_context_lens  # + 1

        size_per_retrieval_token = num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        size_per_streaming_token = num_streaming_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        
        attn_output = fused_attention_fine_grained_dense.single_query_attention(
            q,
            k,
            v,
            input_metadata.retrieval_block_tables[self.layer_idx],
            input_metadata.streaming_block_tables[self.layer_idx],
            retrieval_head_flags,
            head_rank_table,
            lengths_per_sample,
            self.alibi_slopes,
            self.memory_max_len,
            self.tokens_per_block,
            size_per_retrieval_token, 
            size_per_streaming_token,
            sink_size, local_size,
            sink_blocks, local_blocks,
            num_retrieval_kv_heads,
            num_streaming_kv_heads,
            timestep,
            self.rotary_embedding_dim,
            self.rotary_base,
            self.rope_scaling_factor,
            self.neox_rotary_style,
            self.kv_cache_config["INT4_ENABLED"],
            self.kv_cache_config["ZEROS_ENABLED"],
            2048,  # const int multiblock_switch
        )

        selected_page_idx = None
        return attn_output, selected_page_idx
    
    @torch.no_grad()
    def forward_w_dynamic_sparse_fine_grained(
        self,
        q, k, v,
        input_metadata,
        retrieval_head_flags, head_rank_table,
        sink_size, local_size, sink_blocks, local_blocks, 
        num_retrieval_kv_heads, num_streaming_kv_heads,
        cached_dynamic_sparse_page_idx,
        kv_scale_quant_orig,        # NOTE: kv_scale_quant_orig is of no use. Just keep for the interface consistency.
    ):
        timestep = input_metadata.max_seq_len
        # Shang's important fix, but it might cause problem in the end of a block...
        lengths_per_sample = input_metadata.retrieval_context_lens  # + 1

        size_per_retrieval_token = num_retrieval_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        size_per_streaming_token = num_streaming_kv_heads * self.head_dim * (1 if self.use_int8 else 2) // (2 if self.kv_cache_config["INT4_ENABLED"] else 1)
        hidden_dim_per_retrieval_token = num_retrieval_kv_heads * self.head_dim

        if ((timestep) % self.selector_update_interval != 0) and cached_dynamic_sparse_page_idx is not None:      # Since timestep is the length of history, not including the current token. No need to -1 here.
            dynamic_sparse_page_idx = cached_dynamic_sparse_page_idx
        else:
            dynamic_sparse_page_idx = self.dynamic_select_topk_pages(    
                q, k, v,
                input_metadata.retrieval_block_tables[self.layer_idx],
                input_metadata.streaming_block_tables[self.layer_idx],
                retrieval_head_flags, head_rank_table,
                lengths_per_sample, sink_size, local_size, sink_blocks, local_blocks,
                size_per_retrieval_token, size_per_streaming_token,
                num_retrieval_kv_heads, num_streaming_kv_heads, timestep, hidden_dim_per_retrieval_token
            )

        attn_output = fused_attention_fine_grained_sparse.single_query_attention(
            q,
            k,
            v,
            input_metadata.retrieval_block_tables[self.layer_idx],
            input_metadata.streaming_block_tables[self.layer_idx],
            retrieval_head_flags,
            head_rank_table,
            dynamic_sparse_page_idx,
            lengths_per_sample,
            self.alibi_slopes,
            self.memory_max_len,
            self.tokens_per_block,
            size_per_retrieval_token,
            size_per_streaming_token,
            sink_size, local_size,
            sink_blocks, local_blocks,
            num_retrieval_kv_heads, 
            num_streaming_kv_heads,
            timestep,
            self.rotary_embedding_dim,
            self.rotary_base,
            self.rope_scaling_factor,
            self.neox_rotary_style,
            self.kv_cache_config["INT4_ENABLED"],
            self.kv_cache_config["ZEROS_ENABLED"],
            self.sub_chunk_size,
            hidden_dim_per_retrieval_token,
            self.multiblock_switch,
        )

        return attn_output, dynamic_sparse_page_idx