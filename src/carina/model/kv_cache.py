"""Key/value cache for autoregressive decoding."""
from __future__ import annotations

import torch
from torch import Tensor

from .checks import check_kv_append, check_kv_step, check_same_shape


class CarinaKVCache:
    """Single-slot cache storing past keys and values.

    One instance holds the K/V stream of ONE attention pass
    (i.e. one loop iteration). The looped backbone owns one
    cache per iteration.
    """

    keys: Tensor | None
    values: Tensor | None

    def __init__(self) -> None:
        self.keys: Tensor | None = None
        self.values: Tensor | None = None

    def reset(self) -> None:
        """Clear all cached states."""
        self.keys = None
        self.values = None

    def update(
        self,
        new_key: Tensor,
        new_value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Append new steps of keys/values.

        Args:
            new_key: Tensor [batch_size, n_heads, new_seq_len, head_dim], new_seq_len >= 1.
            new_value: Tensor [batch_size, n_heads, new_seq_len, head_dim], new_seq_len >= 1.

        Returns:
            Tuple of concatenated (keys, values) [batch_size, n_heads, cached_seq_len, head_dim].
        """
        check_kv_step("KVCache.update", "new_key", new_key)
        check_kv_step("KVCache.update", "new_value", new_value)
        check_same_shape("KVCache.update", new_key, new_value)
        cached_key: Tensor | None = self.keys
        cached_value: Tensor | None = self.values
        if cached_key is None or cached_value is None:
            full_key: Tensor = new_key
            full_value: Tensor = new_value
        else:
            check_kv_append("KVCache.update", cached_key, new_key)
            full_key = torch.cat([cached_key, new_key], dim=2)
            full_value = torch.cat([cached_value, new_value], dim=2)
        self.keys = full_key
        self.values = full_value
        return full_key, full_value

    def read(self) -> tuple[Tensor | None, Tensor | None]:
        """Read cached keys/values (may be None)."""
        return self.keys, self.values

    def cached_len(self) -> int:
        """Return cached sequence length (0 when empty)."""
        if self.keys is None:
            return 0
        return int(self.keys.size(2))
