# pytype: skip-file
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU-compatible DeepSeek-V4 Lightning Indexer."""

from typing import Optional, Tuple

import jax.numpy as jnp
import torch
import torch.nn as nn
from torchax.interop import jax_view
from torchax.interop import torch_view
from tpu_inference.kernels.experimental.deepseek_v4.streamindex_topk import streamindex_topk
from tpu_inference.layers.common import quantization

# =====================================================================
# IMPORT TPU CUSTOM OPS TO TRIGGER vLLM @register_oot DECORATORS
# =====================================================================
import tpu_inference.layers.vllm.custom_ops.rope
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Indexer


def fused_indexer_q_rope_quant(
    q: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Applies RoPE and dynamically quantizes the queries

  Args:
      q: Un-rotated query tensor of shape [num_tokens, num_heads, head_dim]
      positions: Token positions of shape [num_tokens]
      rotary_emb: The vLLM RoPE CustomOp module (Intercepted by TPU rope.py)

  Returns:
      q_quant: Int8 quantized, RoPE-applied queries
      q_scales: Quantization scales
  """

  # Apply the custom rotary embedding op directly in-place on the query tensor.
  q, _ = rotary_emb(positions, q)

  # Bridge to JAX, call JAX quantize_tensor, and bridge back to PyTorch
  q_jax = jax_view(q)
  q_quant_jax, q_scales_jax = quantization.quantize_tensor(
      q_jax, jnp.float8_e4m3fn
  )
  q_quant = torch_view(q_quant_jax)
  q_scales = torch_view(q_scales_jax)

  # Note: vLLM's implementation rounds the scale factors up to the
  # next power of 2, but the standard division scale returned by quantize_tensor
  # is sufficient here.
  return q_quant, q_scales.squeeze(-1)


class VllmDeepseekV4Indexer(DeepseekV4Indexer):
  """TPU-compatible DeepSeek-V4 Lightning Indexer with StreamIndex.

  This class overrides the forward method of DeepseekV4Indexer to provide a
  TPU-compatible implementation using JAX interop. It uses
  `streamindex_topk` to compute top-k token indices over a
  PagedAttention KV cache.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    # TODO(hwanginho): Attach TPU-native compressor here when needed.
    self.compressor = getattr(self, "compressor", None)  # Placeholder

  # pylint: disable=unused-argument
  def forward(
      self,
      hidden_states: torch.Tensor,
      query: torch.Tensor,
      compressed_kv_score: torch.Tensor,
      indexer_weights: torch.Tensor,
      positions: torch.Tensor,
      rotary_emb: nn.Module,
      slot_mapping: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:

    actual_num_tokens = hidden_states.shape[0]

    q, _ = self.wq_b(query)
    q = q.view(-1, self.n_head, self.head_dim)

    q_quant, q_scales = fused_indexer_q_rope_quant(q, positions, rotary_emb)

    # Fold the query quantization scales into the weights
    weights = (
        indexer_weights.to(q.dtype)
        * self.softmax_scale
        * (self.head_dim**-0.5)
        * q_scales
    )

    attn_metadata_dict = get_forward_context().attn_metadata
    attn_metadata = attn_metadata_dict[self.k_cache.prefix]

    # ---------------------------------------------------------
    # 1. EXECUTE COMPRESSOR & SCATTER TO KV CACHE
    # ---------------------------------------------------------
    # TODO(hwanginho): Execute the TPU-Native Compressor to compute the current
    # token's keys, and write a native TPU scatter kernel to save those keys
    # into `self.k_cache.kv_cache`.
    # (Note: MUST happen before jax_view to guarantee XLA read-after-write ordering!)

    # if self.compressor is not None:
    #   if slot_mapping is None:
    #     current_slot_mapping = (
    #         attn_metadata.slot_mapping.flatten()
    #     )  # pytype: disable=file-attribute-error
    #   else:
    #     current_slot_mapping = slot_mapping
    #
    #   active_slot_mapping = current_slot_mapping[:actual_num_tokens]
    #   self.compressor(...) # Uncomment when compressor is attached

    # ---------------------------------------------------------
    # 2. EXTRACT KV CACHE FOR JAX KERNEL
    # ---------------------------------------------------------
    kv_cache_tensor = self.k_cache.kv_cache

    # TODO(hwanginho): Support DSv4 FP8 index cache format.
    # Once vLLM's FP8 cache is enabled, extract the scale tensor here if it
    # is a separate tensor, or pack it with cache_kv. Need to align with
    # alynie@ on https://github.com/vllm-project/tpu-inference/pull/2858
    # (The exact attribute name depends on vLLM's FP8 implementation, e.g., `kv_scale`)
    # kv_scales_tensor = self.k_cache.kv_scale
    # assert kv_scales_tensor.is_contiguous(), "Scales must be contiguous"
    # kv_scales_jax = jax_view(kv_scales_tensor)

    # ---------------------------------------------------------
    # 3. DIRECT JAX KERNEL CALL
    # ---------------------------------------------------------
    bt_slice = attn_metadata.block_table  # pytype: disable=attribute-error
    sl_slice = attn_metadata.seq_lens  # pytype: disable=attribute-error
    csl_slice = attn_metadata.cu_seq_lens  # pytype: disable=attribute-error

    topk_indices = streamindex_topk(
        query_projection=q_quant[:actual_num_tokens],
        kv_cache=kv_cache_tensor,
        page_indices=bt_slice,
        seq_lens=sl_slice,
        cu_q_lens=csl_slice,
        indexer_weights=weights,
        k=self.topk_tokens,
        compression_ratio=self.compress_ratio,
        # TODO(hwanginho): Tune these block configurations later for performance
        num_kv_pages_per_block=1,
        num_queries_per_block=1,
    )

    return topk_indices
