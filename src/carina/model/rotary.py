"""Rotary positional embeddings (RoPE) for Carina.

We pre-compute ``(cos, sin)`` tables once and slice them by
``position_ids`` at forward time. The rotation itself is just
``x * cos + rotate_half(x) * sin``.
"""
from __future__ import annotations

import torch
from torch import nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by swapping halves with sign flip."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class CarinaRotaryEmbedding(nn.Module):
    """Pre-computed cos/sin tables for RoPE.

    Args:
        head_dim: Per-head dimension (must be even).
        max_position_embeddings: Maximum sequence length to pre-compute.
        rope_theta: Base for the geometric frequency schedule.
    """

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        cos, sin = self._build(max_position_embeddings)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _build(self, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the cos/sin tables for positions ``[0, end)``."""
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        t = torch.arange(end, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(torch.float32), emb.sin().to(torch.float32)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` slices covering the active positions.

        Args:
            x: A query or key tensor of shape ``(B, S, H, D)``; only
                ``S`` is used to determine the active range.
            position_ids: Optional explicit positions; defaults to
                ``[0, S)``.
        """
        seq_len = x.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=x.device)
        cos = self.cos[position_ids].to(x.device)
        sin = self.sin[position_ids].to(x.device)
        return cos, sin

    @staticmethod
    def apply(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE rotation to ``q`` and ``k``.

        ``q``/``k`` have shape ``(B, S, H, D)``; ``cos``/``sin`` have
        shape ``(S, D)``. We unsqueeze to broadcast over batch and
        heads.
        """
        cos_b = cos.unsqueeze(0).unsqueeze(2)
        sin_b = sin.unsqueeze(0).unsqueeze(2)
        q_rot = (q * cos_b) + (_rotate_half(q) * sin_b)
        k_rot = (k * cos_b) + (_rotate_half(k) * sin_b)
        return q_rot.to(q.dtype), k_rot.to(k.dtype)


__all__ = ["CarinaRotaryEmbedding"]
