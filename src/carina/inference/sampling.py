"""Sampling helpers for Carina generation.

Three controls are exposed:

* ``temperature`` — collapses to greedy when ``<= 0``.
* ``top_k`` — keep only the top-``k`` logits (``0`` disables).
* ``top_p`` — nucleus sampling; keep the smallest set whose cumulative
  probability exceeds ``top_p`` (``1.0`` disables).
"""
from __future__ import annotations

import torch


def sample_top_k_top_p(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample a single token from the (possibly filtered) distribution.

    Args:
        logits: Unnormalised log-probabilities of shape ``(B, V)``.
        top_k: If ``> 0``, keep only the top-``k`` logits, setting the
            rest to ``-inf`` before softmax.
        top_p: If ``< 1.0``, perform nucleus (top-p) filtering on the
            sorted logits.
        temperature: Softmax temperature. ``temperature <= 0`` collapses
            the distribution to argmax (greedy decoding).

    Returns:
        A ``(B, 1)`` tensor of sampled token ids.
    """
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    scaled = logits / temperature
    vocab_size = scaled.shape[-1]
    if top_k and top_k > 0:
        k = min(top_k, vocab_size)
        kth = torch.topk(scaled, k, dim=-1)[0][..., -1, None]
        scaled = torch.where(
            scaled < kth,
            torch.full_like(scaled, float("-inf")),
            scaled,
        )
    if top_p and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        sorted_mask = cumulative > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        scaled = torch.zeros_like(scaled).scatter_(-1, sorted_indices, sorted_logits)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Penalise already-generated token ids.

    Args:
        logits: Tensor of shape ``(B, V)``.
        input_ids: Generated token history ``(B, T)``.
        penalty: Multiplicative factor (``>1`` penalises repeats,
            ``<1`` encourages them). ``1.0`` is a no-op.
    """
    if penalty == 1.0:
        return logits
    for b in range(logits.shape[0]):
        seen = torch.unique(input_ids[b])
        scores = logits[b, seen]
        logits[b, seen] = torch.where(
            scores > 0,
            scores / penalty,
            scores * penalty,
        )
    return logits


__all__ = ["apply_repetition_penalty", "sample_top_k_top_p"]
