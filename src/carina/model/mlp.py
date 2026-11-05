"""SwiGLU feed-forward block: down(silu(gate(hidden_states)) * up(hidden_states))."""
from __future__ import annotations

from typing import Protocol

import torch.nn.functional as F
from torch import Tensor, nn

from .checks import check_last_dim, check_pos_int, check_unit_interval


class ComplementProto(Protocol):
    """Structural complement: gate/up/down deltas plus enabled flag."""

    enabled: bool

    def gate_delta(self, hidden_states: Tensor) -> Tensor: ...
    def up_delta(self, hidden_states: Tensor) -> Tensor: ...
    def down_delta(self, gated_product: Tensor) -> Tensor: ...


class CarinaMLP(nn.Module):
    """Gated SwiGLU MLP with dropout on the hidden product.

    Args:
        d_model: Model width (must be > 0).
        d_hidden: Hidden width (must be > 0).
        dropout: Dropout probability in [0, 1).
    """

    d_model: int
    d_hidden: int
    gate_proj: nn.Linear
    up_proj: nn.Linear
    down_proj: nn.Linear
    dropout_layer: nn.Dropout

    def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.2) -> None:
        super().__init__()
        check_pos_int("MLP.init", "d_model", d_model)
        check_pos_int("MLP.init", "d_hidden", d_hidden)
        check_unit_interval("MLP.init", "dropout", dropout)
        self.d_model: int = d_model
        self.d_hidden: int = d_hidden
        self.gate_proj = nn.Linear(d_model, d_hidden)
        self.up_proj = nn.Linear(d_model, d_hidden)
        self.down_proj = nn.Linear(d_hidden, d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self, hidden_states: Tensor, complement: ComplementProto | None = None
    ) -> Tensor:
        """Apply SwiGLU transform with optional depth complement.

        Args:
            hidden_states: Tensor [..., d_model].
            complement: One DepthComplement group or None.

        Returns:
            Tensor [..., d_model].
        """
        check_last_dim("MLP.forward", "hidden_states", hidden_states, self.d_model)
        gate_states: Tensor = self.gate_proj(hidden_states)
        up_states: Tensor = self.up_proj(hidden_states)
        if complement is not None and complement.enabled:
            gate_states = gate_states + complement.gate_delta(hidden_states)
            up_states = up_states + complement.up_delta(hidden_states)
        gated_product: Tensor = self.dropout_layer(F.silu(gate_states) * up_states)
        output_states: Tensor = self.down_proj(gated_product)
        if complement is not None and complement.enabled:
            output_states = output_states + complement.down_delta(gated_product)
        return output_states
