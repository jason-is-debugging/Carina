"""Carina architectural hyperparameters.

Defines the single dataclass that fully describes a ``CarinaForCausalLM``
checkpoint. Default values land around ~50M parameters (8 layers,
hidden 768, 8 attention heads, 4 KV heads) — small enough to train on
a single CPU/GPU yet large enough to hold a useful chat assistant.

The class is serialised via :meth:`save` and :meth:`load`. Configs are
**never** stored as JSON for in-repo state (see ``AGENT_PROMPT.md``);
JSON is only used to *exchange* architecture with the inference path.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


@dataclass
class CarinaConfig:
    """Architectural hyperparameters for ``CarinaForCausalLM``.

    Attributes:
        vocab_size: Tokenizer vocabulary size (8k–20k per spec).
        hidden_size: Token-embedding / residual-stream width.
        num_hidden_layers: Transformer block count.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value heads (defaults to
            ``num_attention_heads`` for plain MHA; smaller values enable
            grouped-query attention).
        head_dim: Per-head dimension; inferred when ``0``.
        intermediate_size: SwiGLU MLP hidden width; defaults via
            ``__post_init__`` when ``0``.
        max_position_embeddings: Maximum context length (RoPE budget).
        rms_norm_eps: Epsilon for RMSNorm.
        rope_theta: Base frequency for RoPE.
        tie_word_embeddings: Share weights between token embedding and
            LM head (parameter-efficient for small models).
        dropout: Dropout probability (kept low; the model is tiny).
        pad_token_id: Padding token id (defaults to ``0``).
        bos_token_id: Beginning-of-sequence id.
        eos_token_id: End-of-sequence id.
        dtype: Default weight dtype (typically ``"bfloat16"``).
    """

    vocab_size: int = 8192
    hidden_size: int = 768
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int = 96
    intermediate_size: int = 2048
    max_position_embeddings: int = 1024
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        """Validate and default fields that depend on other fields."""
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads "
                f"(got hidden_size={self.hidden_size}, heads={self.num_attention_heads})"
            )
        if self.num_key_value_heads <= 0:
            self.num_key_value_heads = self.num_attention_heads
        if self.num_key_value_heads > self.num_attention_heads:
            raise ValueError(
                "num_key_value_heads cannot exceed num_attention_heads "
                f"(got {self.num_key_value_heads} > {self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim <= 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.intermediate_size <= 0:
            # LLaMA-style 8/3 ratio rounded to a multiple of 32.
            self.intermediate_size = max(
                32,
                int(round(self.hidden_size * 8 / 3 / 32) * 32),
            )

    @property
    def head_dim_eff(self) -> int:
        """Effective per-head dimension (handles ``head_dim=0``)."""
        return self.head_dim if self.head_dim > 0 else self.hidden_size // self.num_attention_heads

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow dict copy of the dataclass."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Persist the config to ``path`` as JSON for interchange.

        Args:
            path: Destination file. Parent directories are created.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> CarinaConfig:
        """Load a config previously written by :meth:`save`.

        Unknown keys are silently ignored so newer checkpoints can be
        loaded by older code without crashing.
        """
        with Path(path).open("r", encoding="utf-8") as fh:
            raw: dict[str, Any] = json.load(fh)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})


__all__ = ["CarinaConfig"]
