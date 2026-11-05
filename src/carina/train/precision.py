"""BF16-parameter AdamW with FP32 optimizer state.

Why this module exists
----------------------
Carina's backbone is trained entirely in bf16 (single route, no AMP
branch). That is a good fit for an 8GB card: bf16 halves the weight,
gradient and activation footprint versus fp32.

The risk of bf16 *parameters* is not the forward pass (which is fine in
bf16) but the *optimizer update*: bf16 keeps 8 mantissa bits, so the
relative resolution of a weight is 2**-8 ~ 3.9e-3. A weight of
magnitude 0.02 therefore has an absolute resolution of ~1.7e-4, and a
LayerNorm gain of magnitude 1.0 has a resolution of ~3.9e-3.

For *online* learning the update magnitude is deliberately tiny (a user
edited one sentence), so the update can fall below the bf16 resolution
and be rounded away. Momentum/adaptive state kept in fp32 is what makes
the accumulated direction survive; the fp32 ``master`` copy is what
makes each individual update survive.

This optimizer keeps:
  * parameters: bf16 (unchanged, so inference path is untouched)
  * gradients : bf16 (produced by the bf16 forward pass)
  * exp_avg / exp_avg_sq / master: fp32

Usage::

    opt = BFloat16AdamW(model.parameters(), lr=1e-4)
    opt.step()

Memory cost for the current Carina config (230.5M params):
  bf16 params + bf16 grads + fp32 exp_avg + fp32 exp_avg_sq = 1.72 GB
  (+ fp32 master copy = +0.86 GB when use_master=True)
"""
from __future__ import annotations

import math
from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer

__all__ = ["BFloat16AdamW"]


class BFloat16AdamW(Optimizer):
    """AdamW for bf16 parameters with fp32 optimizer state.

    Args:
        params: Iterable of parameters (bf16 tensors).
        lr: Learning rate.
        betas: Adam beta coefficients.
        eps: Term added to the denominator for numerical stability.
        weight_decay: Decoupled weight decay (applied in fp32).
        use_master: Keep an fp32 master copy of each parameter. Costs
            4 bytes/param but guarantees the per-step update survives
            the bf16 rounding. Recommended for online learning.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        use_master: bool = True,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"[BFloat16AdamW] expected lr>0, got {lr!r}")
        if eps <= 0.0:
            raise ValueError(f"[BFloat16AdamW] expected eps>0, got {eps!r}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"[BFloat16AdamW] expected betas in [0,1), got {betas!r}")
        if weight_decay < 0.0:
            raise ValueError(
                f"[BFloat16AdamW] expected weight_decay>=0, "
                f"got {weight_decay!r}")
        super().__init__(params, dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, use_master=use_master,
        ))

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one AdamW update on every parameter group."""
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = float(closure())
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr: float = float(group["lr"])
            eps: float = float(group["eps"])
            weight_decay: float = float(group["weight_decay"])
            use_master: bool = bool(group["use_master"])
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad: Tensor = param.grad.detach().float()
                state = self.state[param]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(param, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(
                        param, dtype=torch.float32)
                    state["step"] = 0
                    if use_master:
                        state["master"] = param.detach().float().clone()
                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]
                state["step"] += 1
                step: int = int(state["step"])

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                bias_correction1: float = 1.0 - beta1 ** step
                bias_correction2: float = 1.0 - beta2 ** step
                denom: Tensor = (exp_avg_sq / bias_correction2).sqrt().add_(eps)
                update: Tensor = (exp_avg / bias_correction1) / denom

                target: Tensor = state["master"] if use_master else param.float()
                if weight_decay != 0.0:
                    update.add_(target, alpha=weight_decay)
                target.add_(update, alpha=-lr)
                param.copy_(target.to(param.dtype))
                if not use_master:
                    state["master"] = None
        return loss
