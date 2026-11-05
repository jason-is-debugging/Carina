"""Memory-lean pretrain corpus: one flat int32 token stream on disk.

Why: `PretrainDataset` keeps two copies of the corpus in RAM (a list of
per-doc token lists, plus the sliced windows). At 22M tokens that is
~2GB of Python ints, which on a 14GB box gets the process silently
SIGKILLed by the OOM killer (no traceback, no exception).

This module instead pre-tokenizes the corpus into a single flat int32
numpy array (22M tokens = 90MB) and serves fixed-size windows from it
via mmap. Windows are non-overlapping, one token is dropped between
consecutive windows so every token is a label exactly once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


def build_token_cache(
    corpus_path: Path,
    cache_path: Path,
    tokenizer: object,
    eos_id: int,
    max_chars: int = 2048,
) -> Path:
    """Tokenize a {"text": ...} JSONL corpus into a flat int32 array."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[np.ndarray] = []
    total: int = 0
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj: dict = json.loads(line)
            text: str = (obj.get("text") or obj.get("content")
                         or obj.get("story") or "")
            if not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()[:max_chars]
            ids: list[int] = list(tokenizer.encode(text))  # type: ignore[attr-defined]
            chunks.append(np.asarray(ids + [eos_id], dtype=np.int32))
            total += len(ids) + 1
            if len(chunks) % 100000 == 0:
                print(f"  tokenized {total:,} tokens", flush=True)
    flat: np.ndarray = np.concatenate(chunks)
    np.save(cache_path, flat)
    print(f"token cache: {cache_path} tokens={len(flat):,} docs={len(chunks):,}")
    return cache_path


class TokenWindowDataset(Dataset):
    """Fixed-size (input, label) windows sliced from a flat token stream.

    Args:
        flat: int token stream (numpy array or mmap).
        seq_len: context length; each item holds seq_len+1 tokens so that
            inputs and labels are both seq_len long.
        lo: inclusive start offset of this split.
        hi: exclusive end offset of this split.
    """

    flat: np.ndarray
    seq_len: int
    lo: int
    n: int

    def __init__(self, flat: np.ndarray, seq_len: int, lo: int, hi: int) -> None:
        super().__init__()
        self.flat = flat
        self.seq_len = seq_len
        self.lo = lo
        self.n = max(0, (hi - lo) // seq_len - 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        start: int = self.lo + idx * self.seq_len
        window: np.ndarray = np.asarray(
            self.flat[start:start + self.seq_len + 1], dtype=np.int64)
        import torch
        return torch.from_numpy(window[:-1]), torch.from_numpy(window[1:])
