"""Single transformer decoder block (pre-norm) for Carina."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from ..config import CarinaConfig
from .attention import CarinaAttention
from .mlp import CarinaMLP
from .rmsnorm import CarinaRMSNorm


class CarinaDecoderLayer(nn.Module):
    """Pre-norm decoder block with optional gradient checkpointing.

    Args:
        config: A :class:`carina.config.CarinaConfig` instance.
    """

    def __init__(self, config: CarinaConfig) -> None:
        super().__init__()
        self.self_attn = CarinaAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim_eff,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rms_norm_eps=config.rms_norm_eps,
            dropout=config.dropout,
        )
        self.input_layernorm = CarinaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = CarinaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = CarinaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.dropout,
        )

    def _forward_body(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
        attention_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        residual = hidden_states
        ln_out = self.input_layernorm(hidden_states)
        attn_out, new_past = self.self_attn(
            ln_out,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden_states = residual + attn_out
        residual = hidden_states
        mlp_out = self.mlp(self.post_attention_layernorm(hidden_states))
        hidden_states = residual + mlp_out
        return hidden_states, new_past

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """Run the block; return ``(hidden_states, new_past)``."""
        # pyright: ignore[has-type] — _gradient_checkpointing is set by
        # _setup_run() via torch.nn.Module.__setattr__ semantics.
        if self.training and getattr(self, "_gradient_checkpointing", False):
            return checkpoint(
                self._forward_body,
                hidden_states,
                past_key_value,
                use_cache,
                attention_mask,
                use_reentrant=False,
            )
        return self._forward_body(
            hidden_states, past_key_value, use_cache, attention_mask
        )


__all__ = ["CarinaDecoderLayer"]
