"""Shared utilities: seeding, device, parameter counting."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    """Fix Python / NumPy / Torch RNGs for reproducibility."""
    assert isinstance(seed, int), (
        f"[utils.set_seed] expected int seed, got {type(seed)}={seed!r}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return cuda if available else cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str) -> str:
    """Create a directory if it does not exist."""
    assert isinstance(path, str) and len(path) > 0, (
        f"[utils.ensure_dir] expected non-empty str path, got {path!r}")
    os.makedirs(path, exist_ok=True)
    return path


def count_parameters(module: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters of a module."""
    assert isinstance(module, nn.Module), (
        f"[utils.count_parameters] expected nn.Module, got {type(module)}")
    total: int = sum(p.numel() for p in module.parameters())
    trainable: int = sum(
        p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def peak_memory_gb(device: torch.device) -> float:
    """Return current CUDA peak allocation in GB (0.0 on CPU)."""
    assert isinstance(device, torch.device), (
        f"[utils.peak_memory_gb] expected torch.device, got {type(device)}")
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    peak: int = torch.cuda.max_memory_allocated(device)
    return peak / (1024 ** 3)
