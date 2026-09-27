"""Configs package — Python dataclass configs for every Carina stage.

All configuration lives here as importable Python modules. There are
**no** JSON config files in the repo (see ``AGENT_PROMPT.md`` §2.1);
training scripts accept either a config module path or CLI overrides.
"""
from __future__ import annotations

from .base import BaseConfig
from .pretrain import PretrainConfig
from .sft import SFTConfig
from .tokenizer import TokenizerConfig

__all__ = ["BaseConfig", "PretrainConfig", "SFTConfig", "TokenizerConfig"]
