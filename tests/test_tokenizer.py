"""Tests for :mod:`carina.tokenizer`."""
from __future__ import annotations

from pathlib import Path

import pytest

from carina.tokenizer import (
    SPECIAL_TOKENS,
    CarinaTokenizer,
    build_default_tokenizer,
)

TOKENIZER_DIR = Path("models/carina_tokenizer")


@pytest.fixture(scope="module")
def tokenizer() -> CarinaTokenizer:
    if not TOKENIZER_DIR.is_dir():
        pytest.skip(f"Tokenizer not trained at {TOKENIZER_DIR}")
    return CarinaTokenizer.from_file(TOKENIZER_DIR)


def test_vocab_in_range(tokenizer: CarinaTokenizer) -> None:
    assert 500 <= tokenizer.vocab_size <= 20000


def test_special_token_ids(tokenizer: CarinaTokenizer) -> None:
    assert tokenizer.bos_token_id in range(tokenizer.vocab_size)
    assert tokenizer.eos_token_id in range(tokenizer.vocab_size)
    assert tokenizer.pad_token_id in range(tokenizer.vocab_size)
    assert tokenizer.assistant_token_id != tokenizer.user_token_id


def test_encode_decode_roundtrip(tokenizer: CarinaTokenizer) -> None:
    text = "Hello, Carina! How are you today?"
    ids = tokenizer.encode(text)
    assert all(0 <= i < tokenizer.vocab_size for i in ids)
    back = tokenizer.decode(ids)
    assert "Hello" in back


def test_chat_template_format(tokenizer: CarinaTokenizer) -> None:
    msgs = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    text = tokenizer.apply_chat_template(msgs)
    assert text.startswith(tokenizer.BOS)
    assert tokenizer.SYSTEM in text
    assert tokenizer.USER in text
    assert tokenizer.ASSISTANT in text
    assert text.endswith(tokenizer.EOS)


def test_build_labels_masks_non_assistant() -> None:
    tok = build_default_tokenizer()
    tok._tokenizer.add_special_tokens(list(SPECIAL_TOKENS))
    text = tok.apply_chat_template(
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    )
    ids = tok.encode(text)
    labels = tok.build_labels(ids)
    assert len(labels) == len(ids)
    n_supervised = sum(1 for x in labels if x != -100)
    assert n_supervised >= len("a")
    n_masked = sum(1 for x in labels if x == -100)
    assert n_masked > 0
