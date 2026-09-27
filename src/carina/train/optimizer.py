"""Optimizer and learning-rate scheduler builders for Carina."""
from __future__ import annotations

import math

from torch import nn, optim


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    betas: tuple[float, float] = (0.9, 0.95),
) -> optim.AdamW:
    """Build AdamW with weight-decay applied to 2-D weights only.

    Args:
        model: The model whose parameters will be optimised.
        lr: Peak learning rate (callers schedule this over time).
        weight_decay: AdamW weight decay for linear/conv weights.
        betas: Adam betas (defaults mirror LLaMA recipes).

    Returns:
        A configured :class:`torch.optim.AdamW`.
    """
    decay: list = []
    no_decay: list = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "embed" not in name:
            decay.append(param)
        else:
            no_decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)


def build_scheduler(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Cosine-decay schedule with linear warmup.

    Args:
        optimizer: Wrapped optimiser.
        warmup_steps: Number of linear warmup steps from 0 → peak lr.
        total_steps: Total training steps (warmup + decay).
        min_lr_ratio: Floor of the cosine (fraction of the initial lr).

    Returns:
        A :class:`torch.optim.lr_scheduler.LambdaLR` instance.
    """

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


__all__ = ["build_optimizer", "build_scheduler"]
