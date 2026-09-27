"""Tests for the Carina model package."""
from __future__ import annotations

import torch

from carina.config import CarinaConfig
from carina.model import CarinaForCausalLM, CarinaModel

SMALL_CFG = CarinaConfig(
    vocab_size=2048,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=32,
    intermediate_size=256,
    max_position_embeddings=128,
)


def _model(cfg: CarinaConfig | None = None) -> CarinaForCausalLM:
    return CarinaForCausalLM(cfg or SMALL_CFG).eval()


def test_forward_shape_and_finite_logits() -> None:
    torch.manual_seed(0)
    m = _model()
    ids = torch.randint(0, SMALL_CFG.vocab_size, (2, 32))
    out = m(input_ids=ids)
    logits = out["logits"]
    assert logits.shape == (2, 32, SMALL_CFG.vocab_size)
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf"


def test_backward_runs() -> None:
    torch.manual_seed(0)
    m = _model()
    ids = torch.randint(0, SMALL_CFG.vocab_size, (1, 16))
    labels = ids.clone()
    out = m(input_ids=ids, labels=labels)
    assert out["loss"] is not None
    out["loss"].backward()
    grad_sum = sum(
        p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None
    )
    assert grad_sum > 0


def test_kv_cache_equivalence() -> None:
    torch.manual_seed(0)
    m = _model()
    ids = torch.randint(0, SMALL_CFG.vocab_size, (1, 16))
    with torch.no_grad():
        full = m(input_ids=ids, use_cache=False)
        chunked_logits = []
        pkv = None
        for start in range(0, 16, 4):
            out = m(
                input_ids=ids[:, start : start + 4],
                past_key_values=pkv,
                use_cache=True,
            )
            pkv = out["past_key_values"]
            chunked_logits.append(out["logits"])
        chunked = torch.cat(chunked_logits, dim=1)
    assert torch.allclose(full["logits"], chunked, atol=1e-4)


def test_parameter_count_in_range() -> None:
    cfg = CarinaConfig()
    m = CarinaForCausalLM(cfg)
    total = sum(p.numel() for p in m.parameters())
    assert 30_000_000 <= total <= 200_000_000, (
        f"Default model has {total / 1e6:.2f}M params (expected 30-200M)"
    )


def test_tied_embeddings_have_single_storage() -> None:
    cfg = CarinaConfig(tie_word_embeddings=True)
    m = CarinaForCausalLM(cfg)
    assert m.lm_head.weight is m.model.embed_tokens.token_embedding.weight


def test_no_cache_path_produces_none_past() -> None:
    m = _model()
    ids = torch.randint(0, SMALL_CFG.vocab_size, (1, 8))
    out = m(input_ids=ids, use_cache=False)
    assert out["past_key_values"] is None


def test_model_body_forward() -> None:
    """``CarinaModel`` alone should produce hidden states + KV cache list."""
    torch.manual_seed(0)
    body = CarinaModel(SMALL_CFG).eval()
    ids = torch.randint(0, SMALL_CFG.vocab_size, (1, 16))
    hidden, presents = body(input_ids=ids, use_cache=True)
    assert hidden.shape == (1, 16, SMALL_CFG.hidden_size)
    assert presents is not None
    assert len(presents) == SMALL_CFG.num_hidden_layers
