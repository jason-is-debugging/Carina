"""Shared utilities used by the Carina training and inference stacks."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def is_main_process() -> bool:
    """Return ``True`` when running on the rank-0 process (or non-DDP)."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def setup_seed(seed: int) -> None:
    """Seed every relevant RNG so runs are reproducible.

    Args:
        seed: The seed value; rank-dependent offsets should be applied
            by the caller for distributed jobs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def format_metrics(metrics: dict[str, Any]) -> str:
    """Pretty-print a metrics dict for console logging."""
    parts: list[str] = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        else:
            parts.append(f"{k}={v}")
    return " | ".join(parts)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file, ignoring blank lines and decoding errors."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write ``records`` as JSONL, creating parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


__all__ = [
    "format_metrics",
    "is_main_process",
    "read_jsonl",
    "setup_seed",
    "write_jsonl",
]
