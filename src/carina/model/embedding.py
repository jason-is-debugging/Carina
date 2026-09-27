"""Token embedding layer for Carina.

A thin wrapper around :class:`torch.nn.Embedding` that exposes the
underlying matrix as ``self.token_embedding.weight`` and a convenience
``self.weight`` alias so the parent model can tie the LM head.
"""
from __future__ import annotations

import torch
from torch import nn


class CarinaEmbedding(nn.Module):
    """Token embedding table used by :class:`CarinaModel`.

    Args:
        vocab_size: Number of rows in the table.
        hidden_size: Embedding width.
        pad_token_id: Id whose row is forced to zero (and whose gradient
            is excluded from Adam updates by PyTorch's built-in logic).
        dtype: Optional storage dtype.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        pad_token_id: int = 0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_size,
            padding_idx=pad_token_id,
            dtype=dtype,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up token ids and return the matching hidden vectors."""
        return self.token_embedding(input_ids)

    @property
    def weight(self) -> nn.Parameter:
        """Alias to the underlying embedding matrix (used for tying)."""
        return self.token_embedding.weight


__all__ = ["CarinaEmbedding"]
