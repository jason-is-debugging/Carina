"""Root-mean-square layer norm used by every Carina block.

The implementation matches the standard formulation
``x * rsqrt(mean(x^2) + eps) * weight`` and performs the heavy lifting
in fp32 to avoid numerical drift on bf16 weights.
"""
from __future__ import annotations

import torch
from torch import nn


class CarinaRMSNorm(nn.Module):
    """RMSNorm — mean-free layer norm.

    Args:
        dim: Channel size to normalise over.
        eps: Numerical-stability epsilon.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm in fp32 and cast back to the input dtype."""
        in_dtype = x.dtype
        x_fp32 = x.float()
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(var + self.eps)
        return (self.weight * x_normed).to(in_dtype)


__all__ = ["CarinaRMSNorm"]
