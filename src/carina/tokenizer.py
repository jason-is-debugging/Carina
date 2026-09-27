"""Carina tokenizer wrapper.

Provides a thin :class:`CarinaTokenizer` facade around a HuggingFace
``tokenizers`` BPE tokenizer so the rest of the project can rely on a
small, stable surface area. The wrapper supports the role-tagged chat
template used throughout Carina
(``<|system|>...<|user|>...<|assistant|>...``) and exposes a helper
that builds label tensors with non-assistant tokens masked to ``-100``.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPre

SPECIAL_TOKENS: tuple[str, ...] = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
)


class CarinaTokenizerCore:
    """Encode/decode/save/load surface for the Carina BPE tokenizer.

    Args:
        inner: A :class:`tokenizers.Tokenizer` instance. When ``None`` a
            placeholder BPE tokenizer is created so the object can be
            constructed without external files (useful for tests).
        pad_to_multiple: When set, ``encode_pair`` will pad outputs to a
            multiple of this many tokens.
    """

    PAD = "<|pad|>"
    BOS = "<|bos|>"
    EOS = "<|eos|>"
    USER = "<|user|>"
    ASSISTANT = "<|assistant|>"
    SYSTEM = "<|system|>"

    def __init__(
        self,
        inner: Tokenizer | None = None,
        pad_to_multiple: int | None = None,
    ) -> None:
        self._tokenizer: Tokenizer = inner if inner is not None else Tokenizer(BPE())
        self.pad_to_multiple = pad_to_multiple

    @classmethod
    def from_file(cls, path: str | Path) -> CarinaTokenizerCore:
        """Load a tokenizer that was previously saved with :meth:`save`."""
        path = Path(path)
        tok_path = path / "tokenizer.json" if path.is_dir() else path
        inner = Tokenizer.from_file(str(tok_path))
        return cls(inner=inner)

    @classmethod
    def from_json_string(cls, json_str: str) -> CarinaTokenizerCore:
        """Build a tokenizer from an in-memory ``tokenizer.json`` string."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            fh.write(json_str.encode("utf-8"))
            tmp = fh.name
        try:
            inner = Tokenizer.from_file(tmp)
        finally:
            os.unlink(tmp)
        return cls(inner=inner)

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return self._tokenizer.get_vocab_size()

    def _id(self, token: str) -> int | None:
        """Return the id of ``token`` or ``None`` if missing."""
        return self._tokenizer.token_to_id(token)

    def _require(self, token: str) -> int:
        """Return the id of ``token`` or raise ``ValueError``."""
        idx = self._id(token)
        if idx is None:
            raise ValueError(f"Tokenizer is missing {token!r}")
        return idx

    @property
    def bos_token_id(self) -> int:
        """BOS token id (raises if not registered)."""
        return self._require(self.BOS)

    @property
    def eos_token_id(self) -> int:
        """EOS token id (raises if not registered)."""
        return self._require(self.EOS)

    @property
    def pad_token_id(self) -> int:
        """Pad token id (falls back to EOS when pad is missing)."""
        idx = self._id(self.PAD)
        return self.eos_token_id if idx is None else idx

    @property
    def user_token_id(self) -> int:
        """Id of the ``<|user|>`` role tag."""
        return self._require(self.USER)

    @property
    def assistant_token_id(self) -> int:
        """Id of the ``<|assistant|>`` role tag."""
        return self._require(self.ASSISTANT)

    @property
    def system_token_id(self) -> int:
        """Id of the ``<|system|>`` role tag."""
        return self._require(self.SYSTEM)

    def add_special_tokens(self, tokens: Sequence[str]) -> None:
        """Register extra special tokens in the underlying tokenizer."""
        trainer_tokens = [t for t in tokens if self._id(t) is None]
        if trainer_tokens:
            self._tokenizer.add_special_tokens(trainer_tokens)

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into a list of token ids."""
        return list(self._tokenizer.encode(text).ids)

    def decode(self, ids: Iterable[int]) -> str:
        """Decode a sequence of token ids back to text."""
        return self._tokenizer.decode(list(ids))

    def encode_pair(self, text: str) -> tuple[list[int], list[int]]:
        """Return ``(ids, attention_mask)`` for ``text``."""
        ids = self.encode(text)
        attn = [1] * len(ids)
        if self.pad_to_multiple:
            pad_n = (-len(ids)) % self.pad_to_multiple
            if pad_n:
                ids = ids + [self.pad_token_id] * pad_n
                attn = attn + [0] * pad_n
        return ids, attn

    def save(self, directory: str | Path) -> None:
        """Persist the tokenizer to ``directory`` as JSON."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(directory / "tokenizer.json"))
        meta = {
            "tokenizer_class": "CarinaTokenizer",
            "model_max_length": 2048,
            "pad_token": self.PAD,
            "bos_token": self.BOS,
            "eos_token": self.EOS,
            "additional_special_tokens": [
                self.USER,
                self.ASSISTANT,
                self.SYSTEM,
            ],
            "special_tokens": list(SPECIAL_TOKENS),
            "vocab_size": self.vocab_size,
        }
        with (directory / "tokenizer_meta.json").open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False)

    def __repr__(self) -> str:
        return (
            f"CarinaTokenizer(vocab_size={self.vocab_size}, "
            f"pad_id={self.pad_token_id}, bos_id={self.bos_token_id}, "
            f"eos_id={self.eos_token_id})"
        )


class CarinaChatMixin:
    """Chat-template and label-mask helpers layered on a tokenizer."""

    def apply_chat_template(
        self,
        messages: Sequence[dict],
        system: str | None = None,
        add_generation_prompt: bool = False,
    ) -> str:
        """Render ``messages`` into the Carina role-tagged format.

        Args:
            messages: Ordered list of ``{"role", "content"}`` dicts.
            system: Default system prompt if no system turn is present.
            add_generation_prompt: When True, append ``<|assistant|>`` at
                the end so the model knows where to start generating.
                Training data omits this; inference typically enables it.
        """
        rendered: list[str] = [self.BOS]
        system_text = system
        conversation: list[tuple[str, str]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                if system_text is None:
                    system_text = content
                continue
            if role not in ("user", "assistant"):
                role = "user"
            conversation.append((role, content))
        if system_text is not None:
            rendered.append(self.SYSTEM)
            rendered.append(system_text)
        for role, content in conversation:
            tag = self.ASSISTANT if role == "assistant" else self.USER
            rendered.append(tag)
            rendered.append(content)
            rendered.append(self.EOS)
        if add_generation_prompt and (not conversation or conversation[-1][0] != "assistant"):
            rendered.append(self.ASSISTANT)
        return "".join(rendered)

    def build_labels(
        self,
        input_ids: Sequence[int],
        assistant_ranges: Sequence[tuple[int, int]] | None = None,
    ) -> list[int]:
        """Build a label tensor with non-assistant tokens masked to ``-100``."""
        ids = list(input_ids)
        labels = [-100] * len(ids)
        if assistant_ranges is None:
            assistant_ranges = self._infer_assistant_ranges(ids)
        for start, end in assistant_ranges:
            s = max(0, start)
            e = min(len(ids), end)
            for i in range(s, e):
                labels[i] = ids[i]
        return labels

    def _infer_assistant_ranges(
        self,
        input_ids: Sequence[int],
    ) -> list[tuple[int, int]]:
        """Infer assistant-supervised spans from role tags."""
        try:
            assistant_id = self.assistant_token_id
            eos_id = self.eos_token_id
        except ValueError:
            return []
        ranges: list[tuple[int, int]] = []
        i = 0
        n = len(input_ids)
        while i < n:
            if input_ids[i] == assistant_id:
                start = i + 1
                j = start
                while j < n and input_ids[j] != eos_id:
                    j += 1
                end = j + 1 if j < n else n
                ranges.append((start, end))
                i = end
            else:
                i += 1
        return ranges


class CarinaTokenizer(CarinaTokenizerCore, CarinaChatMixin):
    """Full-featured tokenizer used throughout Carina."""


def build_default_tokenizer() -> CarinaTokenizer:
    """Construct an empty BPE tokenizer with the Carina special tokens."""
    inner = Tokenizer(BPE())
    inner.pre_tokenizer = ByteLevelPre(add_prefix_space=False)
    inner.add_special_tokens(list(SPECIAL_TOKENS))
    return CarinaTokenizer(inner=inner)


__all__ = [
    "SPECIAL_TOKENS",
    "CarinaChatMixin",
    "CarinaTokenizer",
    "CarinaTokenizerCore",
    "build_default_tokenizer",
]
