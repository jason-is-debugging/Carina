"""Rotary position embedding with plain cos/sin multiply."""
from __future__ import annotations

import torch
from torch import nn

from .checks import (
    check_even,
    check_nonneg_int,
    check_pos_int,
    check_pos_number,
    check_rope_tables,
)


class RotaryEmbedding(nn.Module):
    """Precompute cos/sin tables, apply rotation per even/odd pair.

    Formula per pair (first, second) at position m with angle t:
        rotated_first = first * cos(t) - second * sin(t)
        rotated_second = first * sin(t) + second * cos(t)

    Args:
        head_dim: Per-head feature size (must be even).
        max_len: Maximum cached sequence length.
        base: Rotary frequency base (must be > 0).
    """

    inv_freq: torch.Tensor
    max_len: int
    head_dim: int

    def __init__(self, head_dim: int, max_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        check_even("RoPE.init", "head_dim", head_dim)
        check_pos_int("RoPE.init", "max_len", max_len)
        check_pos_number("RoPE.init", "base", base)
        half_dim: int = head_dim // 2
        dim_ratios: torch.Tensor = torch.arange(half_dim, dtype=torch.float32) / float(half_dim)
        inv_freq: torch.Tensor = 1.0 / (float(base) ** dim_ratios)
        self.register_buffer("inv_freq", inv_freq)
        self.max_len: int = max_len
        self.head_dim: int = head_dim

    def angles(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin tables for positions [offset, offset+seq_len).

        Args:
            seq_len: Number of positions (must be > 0).
            device: Target device.
            dtype: Target dtype.
            offset: Starting position (cached length during decode).
        """
        check_pos_int("RoPE.angles", "seq_len", seq_len)
        check_nonneg_int("RoPE.angles", "offset", offset)
        position_ids: torch.Tensor = torch.arange(
            offset, offset + seq_len, device=device, dtype=torch.float32)
        inv_freq_device: torch.Tensor = self.inv_freq.to(device=device, dtype=torch.float32)
        angle_table: torch.Tensor = torch.outer(position_ids, inv_freq_device).to(dtype=torch.float32)
        cos_table: torch.Tensor = torch.cos(angle_table).to(dtype=dtype)
        sin_table: torch.Tensor = torch.sin(angle_table).to(dtype=dtype)
        return cos_table, sin_table

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_table: torch.Tensor,
        sin_table: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate last-dim pairs of hidden_states with cos/sin tables.

        Args:
            hidden_states: Tensor [..., seq_len, head_dim].
            cos_table: Tensor [seq_len, head_dim // 2].
            sin_table: Tensor [seq_len, head_dim // 2].

        Returns:
            Rotated tensor with same shape as hidden_states.
        """
        check_rope_tables("RoPE.forward", hidden_states, cos_table, sin_table, self.head_dim)
        even_dims: torch.Tensor = hidden_states[..., 0::2]
        odd_dims: torch.Tensor = hidden_states[..., 1::2]
        broadcast_shape: list[int] = [1] * (even_dims.dim() - 2) + [cos_table.size(0), cos_table.size(1)]
        cos_broadcast: torch.Tensor = cos_table.view(broadcast_shape)
        sin_broadcast: torch.Tensor = sin_table.view(broadcast_shape)
        rotated_even: torch.Tensor = even_dims * cos_broadcast - odd_dims * sin_broadcast
        rotated_odd: torch.Tensor = even_dims * sin_broadcast + odd_dims * cos_broadcast
        rotated_states: torch.Tensor = torch.empty_like(hidden_states)
        rotated_states[..., 0::2] = rotated_even
        rotated_states[..., 1::2] = rotated_odd
        return rotated_states
