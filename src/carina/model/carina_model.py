"""Shared decoder looped K times with per-group depth complements."""
from __future__ import annotations

from torch import Tensor, nn
import torch

from carina.model.attention import CarinaMultiHeadAttention
from carina.model.kv_cache import CarinaKVCache
from carina.model.lora import DepthComplement
from carina.model.mlp import CarinaMLP
from carina.model.rope import RotaryEmbedding

from .checks import (
    check_cache_list,
    check_divides,
    check_hidden,
    check_len,
    check_pos_int,
    check_pos_number,
    check_unit_interval,
)


class CarinaModel(nn.Module):
    """One shared attention + one shared MLP, looped max_loop_times.

    Per-group DepthComplement supplies the depth-specific part for the
    shared MLP's gate/up/down projections. Groups are reused every
    lora_reusage iterations: group_id = loop_index // lora_reusage.

    Each loop iteration owns one CarinaKVCache (see new_caches): K/V
    streams of different depths must never share one cache slot.
    """

    max_loop_times: int
    d_model: int
    d_hidden: int
    n_heads: int
    lora_reusage: int
    max_seq_len: int

    def __init__(
        self,
        max_loop_times: int = 16,
        d_model: int = 512,
        d_hidden: int = 2048,
        n_heads: int = 8,
        lora_reusage: int = 4,
        ranks: tuple[int, ...] = (8, 16, 32, 64),
        alpha: float = 16.0,
        dropout: float = 0.2,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        check_pos_int("CarinaModel.init", "max_loop_times", max_loop_times)
        check_pos_int("CarinaModel.init", "d_model", d_model)
        check_pos_int("CarinaModel.init", "d_hidden", d_hidden)
        check_pos_int("CarinaModel.init", "n_heads", n_heads)
        check_pos_int("CarinaModel.init", "lora_reusage", lora_reusage)
        check_pos_int("CarinaModel.init", "max_seq_len", max_seq_len)
        check_pos_number("CarinaModel.init", "alpha", alpha)
        check_unit_interval("CarinaModel.init", "dropout", dropout)
        check_divides("CarinaModel.init", max_loop_times, lora_reusage)
        check_divides("CarinaModel.init", d_model, n_heads)
        self.max_loop_times = max_loop_times
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.lora_reusage = lora_reusage
        num_groups: int = max_loop_times // lora_reusage
        check_len("CarinaModel.init", "ranks", ranks, num_groups)
        self.num_groups: int = num_groups
        self.ranks: tuple[int, ...] = tuple(ranks)
        self.attn_norm = nn.LayerNorm(self.d_model)
        self.mlp_norm = nn.LayerNorm(self.d_model)
        # LayerNorm at the residual stream entrance: looping the same block
        # K times accumulates the hidden norm until it explodes (measured
        # 33 -> 980), drowning out the attention output of later rounds.
        # None disables it (ablation).
        self.stream_norm: nn.LayerNorm | None = nn.LayerNorm(self.d_model)
        self.rope = RotaryEmbedding(
            self.d_model // self.n_heads, self.max_seq_len
        )
        self.mlp = CarinaMLP(
            d_model=self.d_model, d_hidden=self.d_hidden, dropout=dropout
        )
        self.loras = nn.ModuleList(
            DepthComplement(
                d_model=self.d_model,
                d_hidden=self.d_hidden,
                rank=rank,
                alpha=alpha,
            )
            for rank in ranks
        )
        self.attention = CarinaMultiHeadAttention(
            d_model=self.d_model, n_heads=self.n_heads
        )
        # Per-round residual gains (one scalar per round): hidden += gain[i] * branch.
        # Rationale: under plain bf16 the gains never moved off 1.0 (bf16 has a
        # ~0.004 resolution around 1.0, so smaller updates get rounded away).
        # With an fp32 master weight copy they now decay normally (1.0 -> 0.16),
        # showing the model does want to shrink the contribution of later rounds.
        # Set to None to disable (ablation).
        self.attn_gain: nn.Parameter | None = nn.Parameter(
            torch.ones(self.max_loop_times))
        self.mlp_gain: nn.Parameter | None = nn.Parameter(
            torch.ones(self.max_loop_times))

    def new_caches(self) -> list[CarinaKVCache]:
        """Create one empty cache per loop iteration."""
        return [CarinaKVCache() for _ in range(self.max_loop_times)]

    def forward(
        self, hidden_states: Tensor, caches: list[CarinaKVCache] | None = None,
    ) -> Tensor:
        """Loop shared attention + shared MLP with complements.

        Args:
            hidden_states: Tensor [batch_size, seq_len, d_model] hidden states.
            caches: One cache per loop iteration, or None for pure
                full forward (training). Non-None means prefill
                (seq_len >= 1, empty caches) or decode (seq_len == 1).

        Returns:
            Tensor [batch_size, seq_len, d_model].
        """
        check_hidden("CarinaModel.forward", "hidden_states", hidden_states, self.d_model)
        if caches is not None:
            check_cache_list(
                "CarinaModel.forward", caches, self.max_loop_times, CarinaKVCache)
        for loop_index in range(self.max_loop_times):
            active_cache: CarinaKVCache | None = caches[loop_index] if caches is not None else None
            attn_out = self.attention(
                self.attn_norm(hidden_states), rope=self.rope, kvcache=active_cache
            )
            if self.attn_gain is not None:
                attn_out = attn_out * self.attn_gain[loop_index]
            hidden_states = hidden_states + attn_out
            normed_states: Tensor = self.mlp_norm(hidden_states)
            complement: DepthComplement = self.loras[loop_index // self.lora_reusage]
            mlp_out = self.mlp(normed_states, complement)
            if self.mlp_gain is not None:
                mlp_out = mlp_out * self.mlp_gain[loop_index]
            hidden_states = hidden_states + mlp_out
            if self.stream_norm is not None:
                # Normalize the residual stream: looping K times makes the hidden
                # norm grow linearly (measured 33 -> 980), so the attention
                # output of later rounds is drowned out and the benefit of
                # depth saturates around round 4.
                hidden_states = self.stream_norm(hidden_states)
        return hidden_states
