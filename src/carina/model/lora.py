"""Depth complement for shared CarinaMLP: one rank per module."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .checks import check_last_dim, check_pos_int, check_pos_number


class DepthComplement(nn.Module):
    """One single-rank complement for all three MLP projections.

    LoRA-style delta = scale * up_matrix @ down_matrix @ hidden_states
    with scale = alpha/rank. The up matrices are zero-init so delta = 0
    at start. Multi-scale coverage comes from stacking several complements
    with different ranks, not from stuffing many ranks into one module.
    """

    d_model: int
    d_hidden: int
    rank: int
    scale: float
    enabled: bool
    gate_down: nn.Parameter
    gate_up: nn.Parameter
    up_down: nn.Parameter
    up_up: nn.Parameter
    down_down: nn.Parameter
    down_up: nn.Parameter

    def __init__(self, d_model: int, d_hidden: int,
                 rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        check_pos_int("DepthComp.init", "d_model", d_model)
        check_pos_int("DepthComp.init", "d_hidden", d_hidden)
        check_pos_int("DepthComp.init", "rank", rank)
        check_pos_number("DepthComp.init", "alpha", alpha)
        self.d_model: int = d_model
        self.d_hidden: int = d_hidden
        self.rank: int = rank
        self.scale: float = float(alpha) / float(rank)
        self.enabled: bool = True
        self.gate_down = nn.Parameter(torch.zeros(rank, d_model))
        self.gate_up = nn.Parameter(torch.zeros(d_hidden, rank))
        self.up_down = nn.Parameter(torch.zeros(rank, d_model))
        self.up_up = nn.Parameter(torch.zeros(d_hidden, rank))
        self.down_down = nn.Parameter(torch.zeros(rank, d_hidden))
        self.down_up = nn.Parameter(torch.zeros(d_model, rank))
        for down_matrix in (self.gate_down, self.up_down, self.down_down):
            nn.init.kaiming_uniform_(down_matrix, a=5 ** 0.5)

    def gate_delta(self, hidden_states: Tensor) -> Tensor:
        """Delta for gate_proj input hidden_states [..., d_model] -> [..., d_hidden]."""
        check_last_dim("DepthComp.gate", "hidden_states", hidden_states, self.d_model)
        return (hidden_states @ self.gate_down.T @ self.gate_up.T) * self.scale

    def up_delta(self, hidden_states: Tensor) -> Tensor:
        """Delta for up_proj input hidden_states [..., d_model] -> [..., d_hidden]."""
        check_last_dim("DepthComp.up", "hidden_states", hidden_states, self.d_model)
        return (hidden_states @ self.up_down.T @ self.up_up.T) * self.scale

    def down_delta(self, gated_product: Tensor) -> Tensor:
        """Delta for down_proj input gated_product [..., d_hidden] -> [..., d_model]."""
        check_last_dim("DepthComp.down", "gated_product", gated_product, self.d_hidden)
        return (gated_product @ self.down_down.T @ self.down_up.T) * self.scale
