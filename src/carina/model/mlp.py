"""SwiGLU feed-forward block used by every Carina layer.

The activation is ``SiLU(gate_proj(x)) * up_proj(x)`` followed by
``down_proj``. The ``gate`` and ``up`` projections share the same
output shape; we deliberately omit biases to match the LLaMA recipe.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CarinaMLP(nn.Module):
    """SwiGLU MLP with ``(gate_proj, up_proj, down_proj)``.

    Args:
        hidden_size: Input/output width.
        intermediate_size: Hidden width of the gated projection.
        dropout: Dropout probability applied to the output.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if intermediate_size is None:
            intermediate_size = max(32, int(round(hidden_size * 8 / 3 / 32) * 32))
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the gated MLP transformation."""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


__all__ = ["CarinaMLP"]
