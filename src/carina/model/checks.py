"""One-line precondition checks for model modules.

Call sites just call ``check_*(...)``. The message (location / expected /
actual) lives here so the forward path is not a wall of asserts.
Each helper asserts and returns True.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _is_pos_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def check_pos_int(location: str, name: str, value: object) -> bool:
    """value is a positive int (bool excluded)."""
    assert _is_pos_int(value), (
        f"[{location}] expected {name}>0 int, got {value!r}")
    return True


def check_nonneg_int(location: str, name: str, value: object) -> bool:
    """value is an int >= 0 (bool excluded)."""
    ok = isinstance(value, int) and not isinstance(value, bool) and value >= 0
    assert ok, f"[{location}] expected {name}>=0 int, got {value!r}"
    return True


def check_pos_number(location: str, name: str, value: object) -> bool:
    """value is an int/float > 0."""
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    ok = ok and float(value) > 0
    assert ok, f"[{location}] expected {name}>0, got {value!r}"
    return True


def check_unit_interval(location: str, name: str, value: object) -> bool:
    """value is a float in [0, 1)."""
    ok = isinstance(value, float) and 0.0 <= value < 1.0
    assert ok, f"[{location}] expected {name} in [0, 1), got {value!r}"
    return True


def check_divides(location: str, whole: int, part: int) -> bool:
    """whole % part == 0."""
    assert whole % part == 0, (
        f"[{location}] expected {whole} % {part} == 0, got {whole % part}")
    return True


def check_even(location: str, name: str, value: object) -> bool:
    """value is a positive even int."""
    ok = _is_pos_int(value) and int(value) % 2 == 0
    assert ok, f"[{location}] expected even {name}, got {value!r}"
    return True


def check_len(location: str, name: str, values: object, expected: int) -> bool:
    """len(values) == expected."""
    got = len(values) if hasattr(values, "__len__") else None
    assert got == expected, (
        f"[{location}] expected len({name})=={expected}, got {got!r}")
    return True


def check_tensor(location: str, name: str, value: object, ndim: int) -> bool:
    """value is a Tensor with ndim dimensions."""
    ok = isinstance(value, Tensor) and value.dim() == ndim
    shape = tuple(value.shape) if isinstance(value, Tensor) else None
    assert ok, (
        f"[{location}] expected {ndim}D Tensor {name}, "
        f"got {type(value)} shape={shape}")
    return True


def check_last_dim(location: str, name: str, value: object, expected: int) -> bool:
    """value is a Tensor whose last dim equals expected."""
    assert isinstance(value, Tensor), (
        f"[{location}] expected Tensor {name}, got {type(value)}")
    got = int(value.size(-1))
    assert got == expected, (
        f"[{location}] expected {name} last dim {expected}, "
        f"got {got} shape={tuple(value.shape)}")
    return True


def check_hidden(location: str, name: str, value: object, d_model: int) -> bool:
    """value is a 3D [batch_size, seq_len, d_model] Tensor."""
    check_tensor(location, name, value, 3)
    assert isinstance(value, Tensor)
    check_last_dim(location, name, value, d_model)
    return True


def check_seq_at_least(location: str, seq_len: int, minimum: int) -> bool:
    """seq_len >= minimum."""
    assert seq_len >= minimum, (
        f"[{location}] expected seq_len>={minimum}, got {seq_len}")
    return True


def check_not_none(location: str, name: str, value: object) -> bool:
    """value is not None."""
    assert value is not None, f"[{location}] expected {name}, got None"
    return True


def check_is(location: str, name: str, value: object, expected_type: type) -> bool:
    """isinstance(value, expected_type)."""
    assert isinstance(value, expected_type), (
        f"[{location}] expected {name} {expected_type.__name__}, got {type(value)}")
    return True


def check_id_batch(location: str, ids: object, vocab_size: int) -> bool:
    """ids is [batch_size, seq_len] long-like with values in [0, vocab_size)."""
    check_tensor(location, "ids", ids, 2)
    assert isinstance(ids, Tensor)
    low, high = int(ids.min()), int(ids.max())
    assert low >= 0 and high < vocab_size, (
        f"[{location}] expected ids in [0, {vocab_size}), got [{low}, {high}]")
    return True


def check_loss_pair(location: str, logits: Tensor, labels: Tensor) -> bool:
    """logits [batch_size, seq_len, vocab_size] and labels [batch_size, seq_len]."""
    assert logits.dim() == 3 and labels.dim() == 2, (
        f"[{location}] expected logits 3D and labels 2D, "
        f"got {tuple(logits.shape)} / {tuple(labels.shape)}")
    assert logits.size(0) == labels.size(0) and logits.size(1) == labels.size(1), (
        f"[{location}] batch_size/seq_len mismatch logits {tuple(logits.shape)} "
        f"vs labels {tuple(labels.shape)}")
    return True


def check_single_token(location: str, ids: Tensor) -> bool:
    """ids shape [batch_size, 1]."""
    assert ids.size(1) == 1, (
        f"[{location}] expected seq_len=1, got seq_len={ids.size(1)} "
        f"shape={tuple(ids.shape)}")
    return True


def check_cache_list(location: str, caches: object, expected_count: int, expected_type: type) -> bool:
    """caches is a list of expected_count instances of expected_type."""
    assert isinstance(caches, list) and len(caches) == expected_count, (
        f"[{location}] expected {expected_count} caches, got "
        f"{len(caches) if isinstance(caches, list) else type(caches)}")
    for item in caches:
        assert isinstance(item, expected_type), (
            f"[{location}] expected {expected_type.__name__}, got {type(item)}")
    return True


def check_kv_step(location: str, name: str, tensor: object) -> bool:
    """tensor is [batch_size, n_heads, seq_len>=1, head_dim]."""
    assert isinstance(tensor, Tensor), (
        f"[{location}] expected Tensor {name}, got {type(tensor)}")
    assert tensor.dim() == 4 and tensor.size(2) >= 1, (
        f"[{location}] expected {name} [batch_size,n_heads,seq_len>=1,head_dim], "
        f"got {tuple(tensor.shape)}")
    return True


def check_same_shape(location: str, left: Tensor, right: Tensor) -> bool:
    """left.shape == right.shape."""
    assert left.shape == right.shape, (
        f"[{location}] shape mismatch {tuple(left.shape)} vs {tuple(right.shape)}")
    return True


def check_kv_append(location: str, cached: Tensor, incoming: Tensor) -> bool:
    """incoming matches cached on batch_size, n_heads, head_dim, device, dtype."""
    assert cached.shape[:2] + cached.shape[3:] == incoming.shape[:2] + incoming.shape[3:], (
        f"[{location}] batch_size/n_heads/head_dim mismatch cached {tuple(cached.shape)} "
        f"vs new {tuple(incoming.shape)}")
    assert cached.device == incoming.device, (
        f"[{location}] device mismatch {cached.device} vs {incoming.device}")
    assert cached.dtype == incoming.dtype, (
        f"[{location}] dtype mismatch {cached.dtype} vs {incoming.dtype}")
    return True


def check_rope_tables(
    location: str, hidden_states: Tensor, cos_table: Tensor, sin_table: Tensor, head_dim: int,
) -> bool:
    """hidden_states last dim is head_dim; cos/sin are [seq_len, head_dim // 2]."""
    check_last_dim(location, "hidden_states", hidden_states, head_dim)
    half_dim = head_dim // 2
    expected_shape = (hidden_states.size(-2), half_dim)
    assert isinstance(cos_table, Tensor) and isinstance(sin_table, Tensor), (
        f"[{location}] expected Tensor cos/sin, got {type(cos_table)}/{type(sin_table)}")
    assert tuple(cos_table.shape) == expected_shape, (
        f"[{location}] expected cos {expected_shape}, got {tuple(cos_table.shape)}")
    assert tuple(sin_table.shape) == expected_shape, (
        f"[{location}] expected sin {expected_shape}, got {tuple(sin_table.shape)}")
    return True


def check_query_span(location: str, query_seq_len: int, kv_seq_len: int) -> bool:
    """1 <= query_seq_len <= kv_seq_len for a cached causal mask."""
    assert query_seq_len >= 1 and kv_seq_len >= query_seq_len, (
        f"[{location}] expected 1<=query_seq_len<=kv_seq_len, "
        f"got query_seq_len={query_seq_len} kv_seq_len={kv_seq_len}")
    return True
