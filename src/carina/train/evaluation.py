"""Validation helpers for Carina SFT/pretraining."""
from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    autocast_ctx: Any,
    max_batches: int | None = None,
) -> float:
    """Average cross-entropy loss on ``loader``.

    Args:
        model: A :class:`CarinaForCausalLM`.
        loader: DataLoader producing ``(input_ids, labels)`` batches.
        device: Device the batches should be moved to.
        autocast_ctx: Mixed-precision context manager.
        max_batches: Optional cap on number of evaluated batches.

    Returns:
        Mean loss over the (possibly truncated) loader.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for step, (input_ids, labels) in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            with autocast_ctx:
                out = model(input_ids=input_ids, labels=labels)
            loss = out["loss"]
            if loss is None:
                continue
            total_loss += float(loss.item()) * input_ids.size(0)
            total_tokens += input_ids.size(0)
    model.train()
    if total_tokens == 0:
        return float("nan")
    return total_loss / total_tokens


__all__ = ["evaluate"]
