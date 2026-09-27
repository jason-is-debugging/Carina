"""Shared training infrastructure for SFT and pretraining.

Two builders live here:

* :func:`build_tokenizer_and_config` — turn a config dataclass into a
  tokenizer + matching :class:`CarinaConfig`.
* :func:`build_amp` — produce the right autocast context / gradient
  scaler for the requested dtype.

The training loops in :mod:`src.carina.train.sft` and
:mod:`src.carina.train.pretrain` both use these helpers so the
dataclass configuration drives everything.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ..config import CarinaConfig
from ..tokenizer import CarinaTokenizer


def build_tokenizer_and_config(cfg) -> tuple[CarinaTokenizer, CarinaConfig]:
    """Build a tokenizer + matching :class:`CarinaConfig` from a config object.

    Args:
        cfg: Either a :class:`configs.sft.SFTConfig` or a
            :class:`configs.pretrain.PretrainConfig` instance.

    Returns:
        A ``(tokenizer, config)`` tuple.
    """
    tokenizer = CarinaTokenizer.from_file(Path(cfg.tokenizer_dir))
    config = CarinaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads or cfg.num_attention_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        max_position_embeddings=cfg.max_seq_len,
        dropout=getattr(cfg, "dropout", 0.0),
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        dtype=cfg.resolve_dtype(),
    )
    return tokenizer, config


def build_amp(dtype: str, device_type: str) -> tuple[Any, Any]:
    """Return ``(autocast_ctx, grad_scaler)`` for ``dtype`` on ``device_type``.

    On CPU or with ``float32`` we use a ``nullcontext`` and a ``None``
    scaler so the training loop never branches on device.
    """
    if device_type != "cuda":
        from contextlib import nullcontext

        return nullcontext(), None
    if dtype == "bfloat16":
        return (
            torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16),
            None,
        )
    if dtype == "float16":
        return (
            torch.amp.autocast(device_type=device_type, dtype=torch.float16),
            torch.amp.GradScaler(device_type),
        )
    from contextlib import nullcontext

    return nullcontext(), None


def make_arg_parser(description: str) -> argparse.ArgumentParser:
    """Build an :class:`argparse.ArgumentParser` accepting config overrides.

    The parser accepts a single ``--config`` flag plus ``--set key=value``
    pairs. ``--config`` points at a Python file under ``configs/``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=str,
        default="configs.sft",
        help="Python config module to load (e.g. configs.sft).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a single config field (repeatable).",
    )
    return parser


def load_config_from_args(
    parser: argparse.ArgumentParser,
    default_module: str,
    config_class: type,
) -> Any:
    """Parse argv, import the config module, and apply overrides.

    Args:
        parser: An :func:`make_arg_parser` instance.
        default_module: Module to load when ``--config`` is not provided.
        config_class: Dataclass to instantiate (e.g. ``SFTConfig``).
    """
    import importlib
    from dataclasses import replace

    args = parser.parse_args()
    module_name = args.config or default_module
    module = importlib.import_module(module_name)
    instance = module.__dict__[config_class.__name__]()
    for raw in args.set:
        if "=" not in raw:
            parser.error(f"--set expects key=value, got {raw!r}")
        key, value = raw.split("=", 1)
        key = key.strip()
        value = _coerce(value.strip())
        if not hasattr(instance, key):
            parser.error(f"Unknown config field: {key}")
        instance = replace(instance, **{key: value})
    return instance


def _coerce(value: str) -> object:
    """Best-effort literal parser for ``--set`` values."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


__all__ = ["build_amp", "build_tokenizer_and_config", "load_config_from_args", "make_arg_parser"]
