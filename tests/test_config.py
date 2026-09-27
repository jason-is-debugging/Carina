"""Tests for :mod:`carina.config` and the ``configs/`` dataclasses."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from configs import PretrainConfig, SFTConfig, TokenizerConfig
from configs.base import BaseConfig

from carina.config import CarinaConfig


def test_base_defaults_are_set() -> None:
    cfg = BaseConfig()
    assert cfg.seed == 42
    assert cfg.use_wandb is False
    assert cfg.dtype == "bfloat16"


def test_base_resolves_device_dtype() -> None:
    cfg = BaseConfig(device="cpu")
    assert cfg.resolve_device() == "cpu"
    assert cfg.resolve_dtype() == "float32"


def test_carina_config_defaults() -> None:
    cfg = CarinaConfig()
    assert cfg.vocab_size == 8192
    assert cfg.hidden_size == 768
    assert cfg.num_hidden_layers == 8
    assert cfg.num_attention_heads == 8
    assert cfg.tie_word_embeddings is True
    assert cfg.dtype == "bfloat16"


def test_carina_config_field_validation() -> None:
    with pytest.raises(ValueError):
        CarinaConfig(hidden_size=100, num_attention_heads=8)
    with pytest.raises(ValueError):
        CarinaConfig(num_attention_heads=4, num_key_value_heads=8)


def test_carina_config_save_load_roundtrip(tmp_path: Path) -> None:
    cfg = CarinaConfig(
        vocab_size=4096,
        hidden_size=512,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=64,
        intermediate_size=1024,
        max_position_embeddings=512,
    )
    target = tmp_path / "config.json"
    cfg.save(target)
    assert target.is_file()
    loaded = CarinaConfig.load(target)
    assert loaded.vocab_size == 4096
    assert loaded.hidden_size == 512
    assert loaded.num_hidden_layers == 6
    assert loaded.tie_word_embeddings == cfg.tie_word_embeddings


def test_carina_config_load_ignores_unknown_keys(tmp_path: Path) -> None:
    payload = CarinaConfig().as_dict()
    payload["future_setting"] = 42
    path = tmp_path / "config.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    loaded = CarinaConfig.load(path)
    assert loaded.vocab_size == CarinaConfig().vocab_size


def test_carina_config_head_dim_defaulting() -> None:
    cfg = CarinaConfig(
        hidden_size=384, num_attention_heads=6, num_key_value_heads=6, head_dim=0
    )
    assert cfg.head_dim == 384 // 6


def test_sft_config_inherits_base() -> None:
    cfg = SFTConfig()
    assert isinstance(cfg, BaseConfig)
    assert cfg.dummy_data is False
    assert cfg.system_prompt.startswith("You are Carina")


def test_pretrain_config_inherits_base() -> None:
    cfg = PretrainConfig()
    assert isinstance(cfg, BaseConfig)
    assert cfg.learning_rate > 0


def test_tokenizer_config_inherits_base() -> None:
    cfg = TokenizerConfig()
    assert isinstance(cfg, BaseConfig)
    assert cfg.vocab_size >= 8000
