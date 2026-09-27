"""Shared helpers used by the SFT and pretraining loops.

* :func:`init_distributed_mode` — detect ``torchrun`` env vars.
* :class:`SkipBatchSampler` — yield batches but skip the first ``N``,
  used for mid-epoch resume.
"""
from __future__ import annotations

import math
import os
from collections.abc import Iterator, Sequence

import torch
from torch.utils.data import Sampler


def init_distributed_mode() -> int:
    """Detect ``torchrun`` env vars and initialise NCCL when present.

    Returns:
        The local rank (``0`` when not running under ``torchrun``).
    """
    rank = int(os.environ.get("RANK", "-1"))
    if rank == -1:
        return 0
    if not torch.distributed.is_available():
        return 0
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    torch.distributed.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def cosine_warmup_lr(step: int, total_steps: int, lr: float, warmup_ratio: float = 0.05) -> float:
    """Standalone cosine-with-warmup LR helper (kept for backward use)."""
    if total_steps <= 0:
        return lr
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return lr * 0.1 + (lr - lr * 0.1) * 0.5 * (1.0 + math.cos(math.pi * progress))


class SkipBatchSampler(Sampler[list[int]]):
    """Yield ``batch_size``-sized lists, skipping the first ``skip_batches``.

    Used for mid-epoch resume: keep dropping batches until the sampler
    has advanced past the previously completed step count.
    """

    def __init__(
        self,
        sampler: Sampler[int],
        batch_size: int,
        skip_batches: int = 0,
    ) -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = max(0, int(skip_batches))

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        skipped = 0
        for idx in self.sampler:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if batch and skipped >= self.skip_batches:
            yield batch

    def __len__(self) -> int:
        total = len(self.sampler) if isinstance(self.sampler, Sequence) else 0
        n_batches = (total + self.batch_size - 1) // self.batch_size
        return max(0, n_batches - self.skip_batches)


__all__ = ["SkipBatchSampler", "cosine_warmup_lr", "init_distributed_mode"]
