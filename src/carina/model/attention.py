"""Multi-head self-attention with optional KV cache and GQA support.

The class keeps separate projection matrices so
``num_key_value_heads`` can be tuned independently of
``num_attention_heads`` — enabling grouped-query attention when the
two differ. We use :func:`torch.nn.functional.scaled_dot_product_attention`
because it dispatches to the optimal backend for the device.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .rmsnorm import CarinaRMSNorm
from .rotary import CarinaRotaryEmbedding


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``num_kv_heads`` → ``num_heads`` by repeating along heads."""
    if n_rep == 1:
        return x
    bsz, slen, num_kv, head_dim = x.shape
    x = x[:, :, :, None, :].expand(bsz, slen, num_kv, n_rep, head_dim)
    return x.reshape(bsz, slen, num_kv * n_rep, head_dim)


def _prepare_padding_mask(
    attention_mask: torch.Tensor | None,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Convert a 2-D padding mask into a 4-D additive mask for SDPA."""
    if attention_mask is None:
        return None
    mask = attention_mask[:, None, None, :].to(dtype)
    additive = (1.0 - mask) * torch.finfo(dtype).min
    return additive.to(device)


class CarinaAttention(nn.Module):
    """Multi-head self-attention with optional KV cache.

    Args:
        hidden_size: Residual-stream width.
        num_attention_heads: Number of query heads.
        num_key_value_heads: Number of key/value heads (must divide
            ``num_attention_heads``).
        head_dim: Per-head dimension.
        max_position_embeddings: Maximum context length (RoPE budget).
        rope_theta: RoPE base frequency.
        rms_norm_eps: Epsilon for the per-head RMSNorms.
        dropout: Attention / residual dropout probability.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rms_norm_eps: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.n_rep = num_attention_heads // num_key_value_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = CarinaRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = CarinaRMSNorm(head_dim, eps=rms_norm_eps)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.rotary = CarinaRotaryEmbedding(
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
        )

    def _project(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project to q/k/v and apply per-head RMSNorm."""
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        return self.q_norm(q), self.k_norm(k), v

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        """Run one attention layer.

        Args:
            x: Input hidden states ``(B, S, hidden)``.
            past_key_value: Optional ``(k, v)`` cache from a previous call.
            use_cache: Whether to return updated ``(k, v)`` for reuse.
            attention_mask: Optional additive mask ``(B, S_total)``
                (``1`` → keep, ``0`` → mask).

        Returns:
            Output hidden states and an optional updated ``(k, v)`` pair.
        """
        bsz, seq_len, _ = x.shape
        q, k, v = self._project(x)
        past_len = past_key_value[0].shape[1] if past_key_value is not None else 0
        position_ids = torch.arange(
            past_len, past_len + seq_len, device=x.device, dtype=torch.long
        )
        cos, sin = self.rotary(q, position_ids=position_ids)
        q, k = self.rotary.apply(q, k, cos, sin)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        new_past = (k, v) if use_cache else None
        q_t = q.transpose(1, 2)
        k_t = _repeat_kv(k, self.n_rep).transpose(1, 2)
        v_t = _repeat_kv(v, self.n_rep).transpose(1, 2)
        if past_key_value is not None:
            full_len = past_len + seq_len
            causal = torch.ones(seq_len, full_len, device=x.device, dtype=torch.bool)
            causal = torch.triu(causal, diagonal=past_len + 1)
            attn_mask_4d = torch.zeros(
                seq_len, full_len, dtype=q_t.dtype, device=x.device
            )
            attn_mask_4d.masked_fill_(causal, torch.finfo(q_t.dtype).min)
            attn_mask_4d = attn_mask_4d[None, None, :, :].expand(
                bsz, self.num_heads, seq_len, full_len
            )
        else:
            attn_mask_4d = _prepare_padding_mask(attention_mask, q_t.dtype, q_t.device)
        out = F.scaled_dot_product_attention(
            q_t,
            k_t,
            v_t,
            attn_mask=attn_mask_4d,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=attn_mask_4d is None,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        out = self.resid_dropout(self.o_proj(out))
        return out, new_past


class CarinaMultiHeadAttention(CarinaAttention):
    """Alias for :class:`CarinaAttention` (spec-named entry point)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)


__all__ = ["CarinaAttention", "CarinaMultiHeadAttention"]
