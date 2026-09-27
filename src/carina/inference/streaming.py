"""Streaming helpers for incremental decode during generation."""
from __future__ import annotations

from collections.abc import Callable, Iterable


class TextStreamer:
    """Buffered text streamer that flushes one ``str`` per token.

    The streamer integrates with the generator loop: callers push token
    ids with :meth:`put` and then call :meth:`end`. The buffered text
    uses the bound tokenizer to decode incrementally, which avoids
    ``\\ufffd`` replacement characters when possible.

    Args:
        tokenizer: A :class:`carina.tokenizer.CarinaTokenizer`.
        skip_prompt: When ``True`` the first call to :meth:`put` ignores
            the prompt tokens (they are emitted only once at the start).
        skip_special_tokens: Strip special tokens from surfaced text.
    """

    def __init__(
        self,
        tokenizer,
        skip_prompt: bool = False,
        skip_special_tokens: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.skip_special_tokens = skip_special_tokens
        self._cache: list[int] = []
        self._first = True

    def put(self, token_ids: Iterable[int]) -> str:
        """Push a batch of token ids; return any newly-decodable text."""
        ids = list(token_ids)
        if not ids:
            return ""
        if self._first and self.skip_prompt:
            self._first = False
            return ""
        self._first = False
        new_text = ""
        for tid in ids:
            self._cache.append(int(tid))
            decoded = self.tokenizer.decode(self._cache)
            if "\ufffd" not in decoded:
                new_text += decoded
                self._cache = []
        if self.skip_special_tokens:
            for tok in (
                self.tokenizer.BOS,
                self.tokenizer.EOS,
                self.tokenizer.PAD,
            ):
                new_text = new_text.replace(tok, "")
        return new_text

    def end(self) -> str:
        """Final flush; returns any remaining buffered text."""
        if self._cache:
            tail = self.tokenizer.decode(self._cache)
            self._cache = []
            return tail
        return ""


def make_simple_consumer(print_fn: Callable[[str], None] | None = None):
    """Return a callback ``(text: str) -> None`` that prints as it arrives."""
    printer = print_fn or (lambda s: print(s, end="", flush=True))

    def consume(text: str) -> None:
        if text:
            printer(text)

    return consume


__all__ = ["TextStreamer", "make_simple_consumer"]
