"""Pretraining configuration (causal-LM objective, no label masking).

All values are hard-coded here so ``python scripts/pretrain.py`` runs
without any --set overrides.  Override by editing this file or passing
--fresh if you need a clean start (resume is the default).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .base import BaseConfig


@dataclass
class PretrainConfig(BaseConfig):
    """Hyperparameters for causal-language-model pretraining.

    Defaults target a ~150 M-parameter model (12 layers × 1024 hidden)
    trained on ~500 M-token corpus with a 8 192-token BPE vocabulary.
    Effective batch size = batch_size × accumulation_steps = 8 × 1 = 8.
    """

    # ── Data ────────────────────────────────────────────────────────────────
    data_path: Path = field(default_factory=lambda: Path("datasets/carina_pretrain/train.jsonl"))
    tokenizer_dir: Path = field(default_factory=lambda: Path("models/carina_tokenizer"))
    # ── Resume ──────────────────────────────────────────────────────────────
    resume_dir: Path | None = None   # auto-detected below if checkpoint exists
    # ── Architecture ────────────────────────────────────────────────────────
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: int = 8       # GQA: 16 query / 8 KV
    head_dim: int = 64
    intermediate_size: int = 0          # 0 → auto (8/3 ratio)
    max_seq_len: int = 2048
    # ── Training ────────────────────────────────────────────────────────────
    epochs: int = 6
    batch_size: int = 2
    accumulation_steps: int = 4        # effective BS = 2 × 4 = 8
    learning_rate: float = 3e-4        # 3e-4 … 5e-4 (cosine, no restart)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03        # 3 % warmup
    min_lr_ratio: float = 0.1         # cosine floor = 10 % of peak LR
    grad_clip: float = 1.0
    val_split: float = 0.005          # 0.5 % held-out for val loss
    # ── Miscellaneous ──────────────────────────────────────────────────────
    log_interval: int = 10             # steps between console metrics
    save_interval: int = 500          # steps between checkpoint writes
    use_wandb: bool = False
    wandb_project: str = "carina"
    seed: int = 42


__all__ = ["PretrainConfig"]
