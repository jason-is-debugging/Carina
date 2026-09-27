"""Supervised fine-tuning configuration (assistant-token label masking).

All values are hard-coded here so ``python scripts/train_sft.py`` runs
without any --set overrides.  Override by editing this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .base import BaseConfig


@dataclass
class SFTConfig(BaseConfig):
    """Hyperparameters for supervised fine-tuning.

    Effective batch size = batch_size × accumulation_steps = 4 × 4 = 16.
    SFT starts from the pretrain final checkpoint (auto-detected).
    """

    # ── Data ────────────────────────────────────────────────────────────────
    data_path: Path = field(default_factory=lambda: Path("datasets/carina_sft/train.jsonl"))
    tokenizer_dir: Path = field(default_factory=lambda: Path("models/carina_tokenizer"))
    # ── Resume ──────────────────────────────────────────────────────────────
    resume_dir: Path | None = None   # auto-detected below if checkpoint exists
    # ── Architecture (must match pretrain) ─────────────────────────────────
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    num_key_value_heads: int = 8       # GQA: matches pretrain config
    head_dim: int = 64
    intermediate_size: int = 0
    max_seq_len: int = 2048
    # ── Training ────────────────────────────────────────────────────────────
    epochs: int = 3                   # 3 is enough; overfitting starts around epoch 3
    batch_size: int = 2               # keep micro-BS = 2 (same as pretrain); accum=8 → effective=16
    accumulation_steps: int = 8       # effective BS = 2 × 8 = 16
    learning_rate: float = 1e-5       # from 2e-5; lower LR is more stable after pretrain
    dropout: float = 0.1              # applied in attention / residual / MLP / embedding
    weight_decay: float = 0.1        # from 0.01; stronger regularisation for small model
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    val_split: float = 0.005
    patience: int = 2                 # from 4; stop earlier when overfitting begins
    label_smoothing: float = 0.1     # soft targets; reduces overconfidence
    min_delta: float = 1e-3
    # ── System prompt ───────────────────────────────────────────────────────
    system_prompt: str = "You are Carina, a warm and supportive virtual companion."
    # ── Miscellaneous ──────────────────────────────────────────────────────
    log_interval: int = 10
    save_interval: int = 500
    use_wandb: bool = False
    wandb_project: str = "carina"
    seed: int = 42


__all__ = ["SFTConfig"]
