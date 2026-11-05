"""Load / save / fetch a pretrained HF tokenizer (Qwen2.5 by default).

Values are passed in as arguments; this module never imports
``carina.config`` (see config.py design decisions).

Typical use::

    from carina.config import TokenizerConfig
    from carina.tokenizer import ensure_tokenizer

    tok = ensure_tokenizer(
        hf_id=TokenizerConfig.hf_id,
        local_dir=TokenizerConfig.tokenizer_dir,
    )
    ids: list[int] = tok.encode("hello world")
    text: str = tok.decode(ids)
"""
from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer


def fetch_tokenizer(
    hf_id: str,
    local_dir: str | Path | None = None,
    trust_remote_code: bool = False,
) -> AutoTokenizer:
    """Download tokenizer ``hf_id`` from the Hub, optionally save to ``local_dir``.

    Args:
        hf_id: Hub id, e.g. ``"Qwen/Qwen2.5-0.5B"``.
        local_dir: Directory to ``save_pretrained`` into (created if needed).
        trust_remote_code: Passed to ``from_pretrained`` (Qwen2.5 needs False).

    Returns:
        Loaded ``AutoTokenizer`` (fast tokenizer when available).
    """
    tok = AutoTokenizer.from_pretrained(
        hf_id, trust_remote_code=trust_remote_code, use_fast=True
    )
    if local_dir is not None:
        save_tokenizer(tok, local_dir)
    return tok


def save_tokenizer(tok: AutoTokenizer, local_dir: str | Path) -> Path:
    """Save ``tok`` into ``local_dir`` via ``save_pretrained``; return the dir."""
    out = Path(local_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(out))
    return out


def load_tokenizer(
    local_dir: str | Path, trust_remote_code: bool = False
) -> AutoTokenizer:
    """Load a tokenizer previously saved with :func:`save_tokenizer`."""
    return AutoTokenizer.from_pretrained(
        str(local_dir), trust_remote_code=trust_remote_code, use_fast=True
    )


def ensure_tokenizer(
    hf_id: str,
    local_dir: str | Path,
    trust_remote_code: bool = False,
) -> AutoTokenizer:
    """Return local tokenizer if present, else fetch ``hf_id`` and save it.

    Local hit = ``local_dir`` exists and contains ``tokenizer.json`` or
    ``tokenizer_config.json``. Otherwise downloads from the Hub and saves
    for next time.
    """
    local = Path(local_dir)
    if local.is_dir() and (
        (local / "tokenizer.json").exists()
        or (local / "tokenizer_config.json").exists()
    ):
        return load_tokenizer(local, trust_remote_code=trust_remote_code)
    return fetch_tokenizer(
        hf_id, local_dir=local, trust_remote_code=trust_remote_code
    )
