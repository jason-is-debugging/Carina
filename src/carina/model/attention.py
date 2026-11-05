"""Multi-head causal self-attention with RoPE and KV-cache decoding."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .checks import (
    check_divides,
    check_hidden,
    check_not_none,
    check_pos_int,
    check_query_span,
    check_seq_at_least,
)
from .kv_cache import CarinaKVCache
from .rope import RotaryEmbedding


class CarinaMultiHeadAttention(nn.Module):
    """Scaled dot-product causal attention with prefill/decode support."""

    d_model: int
    n_heads: int
    head_dim: int
    query_proj: nn.Linear
    key_proj: nn.Linear
    value_proj: nn.Linear
    out_proj: nn.Linear

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        check_pos_int("MHA.init", "d_model", d_model)
        check_pos_int("MHA.init", "n_heads", n_heads)
        check_divides("MHA.init", d_model, n_heads)
        self.d_model: int = d_model
        self.n_heads: int = n_heads
        self.head_dim: int = d_model // n_heads
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def heads(self, hidden_states: Tensor) -> Tensor:
        """Split [batch_size, seq_len, d_model] into [batch_size, n_heads, seq_len, head_dim]."""
        batch_size: int = hidden_states.size(0)
        seq_len: int = hidden_states.size(1)
        wide_states: Tensor = hidden_states.view(batch_size, seq_len, self.n_heads, self.head_dim)
        return wide_states.transpose(1, 2)

    def forward(
        self,
        hidden_states: Tensor,
        rope: RotaryEmbedding,
        kvcache: CarinaKVCache | None = None,
        use_cache: bool = False,
    ) -> Tensor:
        """Run one attention step: full, prefill, or decode.

        Args:
            hidden_states: Tensor [batch_size, seq_len, d_model].
            rope: RotaryEmbedding instance.
            kvcache: KVCache for cached prefill/decoding (required if use_cache).
            use_cache: If True, decode one token via cache.

        Returns:
            Tensor [batch_size, seq_len, d_model].
        """
        check_hidden("MHA.forward", "hidden_states", hidden_states, self.d_model)

        query_heads: Tensor = self.heads(self.query_proj(hidden_states))
        key_heads: Tensor = self.heads(self.key_proj(hidden_states))
        value_heads: Tensor = self.heads(self.value_proj(hidden_states))

        seq_len: int = hidden_states.size(1)

        if kvcache is None and not use_cache:
            return self._full_forward(
                query_heads, key_heads, value_heads, rope, hidden_states, seq_len
            )

        return self._cached_forward(
            query_heads, key_heads, value_heads, rope, hidden_states, kvcache, seq_len
        )

    def _full_forward(
        self,
        query_heads: Tensor,
        key_heads: Tensor,
        value_heads: Tensor,
        rope: RotaryEmbedding,
        hidden_states: Tensor,
        seq_len: int,
    ) -> Tensor:
        """Full-sequence forward without any cache (training path)."""
        query_heads, key_heads = self._full_rope(
            query_heads, key_heads, rope, hidden_states, seq_len, offset=0
        )

        causal_mask: Tensor = self._causal_mask(seq_len, hidden_states.device)

        attention_scores: Tensor = self._attn_scores(query_heads, key_heads, causal_mask)
        attention_weights: Tensor = F.softmax(attention_scores, dim=-1)

        attended_values: Tensor = attention_weights @ value_heads

        merged_heads: Tensor = attended_values.transpose(1, 2).reshape(
            hidden_states.size(0), seq_len, self.d_model
        )

        return self.out_proj(merged_heads)

    def _cached_forward(
        self,
        query_heads: Tensor,
        key_heads: Tensor,
        value_heads: Tensor,
        rope: RotaryEmbedding,
        hidden_states: Tensor,
        kvcache: CarinaKVCache | None,
        seq_len: int,
    ) -> Tensor:
        """Prefill or decode with KV-cache."""
        check_not_none("MHA.forward", "kvcache", kvcache)
        check_seq_at_least("MHA.forward", seq_len, 1)

        cached_length: int = kvcache.cached_len()

        rotated_query, rotated_key = self._full_rope(
            query_heads, key_heads, rope, hidden_states, seq_len, offset=cached_length
        )

        full_key, full_value = kvcache.update(rotated_key, value_heads)

        if seq_len == 1:
            return self._decode_step(rotated_query, full_key, full_value, hidden_states)

        return self._prefill_step(
            rotated_query, full_key, full_value, hidden_states, seq_len
        )

    def _decode_step(
        self,
        rotated_query: Tensor,
        full_key: Tensor,
        full_value: Tensor,
        hidden_states: Tensor,
    ) -> Tensor:
        """Single-token decode: attend over all cached keys, no mask needed."""
        attention_scores: Tensor = self._attn_scores(rotated_query, full_key, None)
        attention_weights: Tensor = F.softmax(attention_scores, dim=-1)

        attended_values: Tensor = torch.matmul(attention_weights, full_value)

        merged_heads: Tensor = attended_values.transpose(1, 2).reshape(
            hidden_states.size(0), 1, self.d_model
        )

        return self.out_proj(merged_heads)

    def _prefill_step(
        self,
        rotated_query: Tensor,
        full_key: Tensor,
        full_value: Tensor,
        hidden_states: Tensor,
        seq_len: int,
    ) -> Tensor:
        """Multi-token prefill: attend over cached keys with extended causal mask."""
        extended_mask: Tensor = self._causal_mask_cached(
            seq_len, full_key.size(2), hidden_states.device
        )

        attention_scores: Tensor = self._attn_scores(rotated_query, full_key, extended_mask)
        attention_weights: Tensor = F.softmax(attention_scores, dim=-1)

        attended_values: Tensor = torch.matmul(attention_weights, full_value)

        merged_heads: Tensor = attended_values.transpose(1, 2).reshape(
            hidden_states.size(0), seq_len, self.d_model
        )

        return self.out_proj(merged_heads)

    def _full_rope(
        self,
        query_heads: Tensor,
        key_heads: Tensor,
        rope: RotaryEmbedding,
        hidden_states: Tensor,
        seq_len: int,
        offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Apply RoPE for positions [offset, offset+seq_len)."""
        cos_table: Tensor
        sin_table: Tensor
        cos_table, sin_table = rope.angles(seq_len, hidden_states.device, hidden_states.dtype, offset=offset)
        return rope(query_heads, cos_table, sin_table), rope(key_heads, cos_table, sin_table)

    def _attn_scores(
        self,
        query_heads: Tensor,
        key_heads: Tensor,
        attention_mask: Tensor | None,
    ) -> Tensor:
        """Scaled dot-product scores with optional additive mask."""
        scale: float = 1.0 / math.sqrt(self.head_dim)
        scores: Tensor = torch.matmul(query_heads, key_heads.transpose(-2, -1)) * scale
        if attention_mask is not None:
            scores = scores + attention_mask.to(scores.dtype)
        return scores

    def _causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Return additive causal mask [1, 1, seq_len, seq_len]."""
        check_pos_int("MHA.mask", "seq_len", seq_len)
        row_positions: Tensor = torch.arange(seq_len, device=device).view(seq_len, 1)
        col_positions: Tensor = torch.arange(seq_len, device=device).view(1, seq_len)
        allowed_positions: Tensor = col_positions <= row_positions
        mask: Tensor = torch.zeros(seq_len, seq_len, device=device)
        mask = mask.masked_fill(~allowed_positions, float("-inf"))
        return mask.view(1, 1, seq_len, seq_len)

    def _causal_mask_cached(
        self, query_seq_len: int, kv_seq_len: int, device: torch.device
    ) -> Tensor:
        """Causal mask for queries against cached keys [1,1,query_seq_len,kv_seq_len].

        Query global pos = (kv_seq_len - query_seq_len + i); key j allowed iff j <= pos.
        """
        check_query_span("MHA.mask_cached", query_seq_len, kv_seq_len)
        position_offset: int = kv_seq_len - query_seq_len
        row_positions: Tensor = (torch.arange(query_seq_len, device=device) + position_offset).view(query_seq_len, 1)
        col_positions: Tensor = torch.arange(kv_seq_len, device=device).view(1, kv_seq_len)
        allowed_positions: Tensor = col_positions <= row_positions
        mask: Tensor = torch.zeros(query_seq_len, kv_seq_len, device=device)
        mask = mask.masked_fill(~allowed_positions, float("-inf"))
        return mask.view(1, 1, query_seq_len, kv_seq_len)
