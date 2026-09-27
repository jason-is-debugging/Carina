"""Carina: a small, fully-typed virtual-companion language model."""
from __future__ import annotations

from .config import CarinaConfig
from .model import CarinaForCausalLM, CarinaModel
from .tokenizer import CarinaTokenizer

__all__ = [
    "CarinaConfig",
    "CarinaForCausalLM",
    "CarinaModel",
    "CarinaTokenizer",
    "main",
]


def main() -> None:
    """Print a brief summary of the package and the default model."""
    cfg = CarinaConfig()
    model = CarinaForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Carina default config: hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
        f"heads={cfg.num_attention_heads} vocab={cfg.vocab_size}"
    )
    print(f"Total parameters: {total / 1e6:.2f}M")


if __name__ == "__main__":
    main()
