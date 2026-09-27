"""Tokenizer training configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .base import BaseConfig


@dataclass
class TokenizerConfig(BaseConfig):
    """Hyperparameters for :mod:`scripts.train_tokenizer`.

    Attributes:
        corpus_paths: Extra JSONL files to include in the training corpus
            (one conversation per line). May be empty.
        output_dir: Where ``tokenizer.json`` and ``tokenizer_meta.json``
            are written.
        vocab_size: Target vocabulary size (8k–20k per spec).
        max_lines: Maximum corpus lines to consume (caps training time).
        min_frequency: Minimum pair frequency to keep during BPE merges.
    """

    corpus_paths: list[Path] = field(default_factory=list)
    output_dir: Path = field(default_factory=lambda: Path("models/carina_tokenizer"))
    vocab_size: int = 8000
    max_lines: int = 20000
    min_frequency: int = 2


__all__ = ["TokenizerConfig"]
