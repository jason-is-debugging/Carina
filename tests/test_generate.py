"""Tests for the generation pipeline."""
from __future__ import annotations

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from carina.config import CarinaConfig
from carina.inference import generate
from carina.inference.sampling import sample_top_k_top_p
from carina.model import CarinaForCausalLM
from carina.tokenizer import SPECIAL_TOKENS, CarinaTokenizer

SMALL_CFG = CarinaConfig(
    vocab_size=128,
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    head_dim=16,
    intermediate_size=128,
    max_position_embeddings=64,
)

CORPUS = [
    "Hello, how are you today?",
    "I had a long day at work.",
    "Tell me a joke please.",
    "Why did the chicken cross the road?",
    "I love to read books in the morning.",
    "Let's plan a trip to the mountains.",
    "Music helps me focus on tasks.",
    "Coffee is essential for mornings.",
] * 8


def _train_test_tokenizer(tmp_path_factory) -> CarinaTokenizer:
    """Train a small (vocab≈128) tokenizer with all special tokens."""
    inner = Tokenizer(models.BPE(unk_token=SPECIAL_TOKENS[1]))
    inner.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=128,
        show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=list(SPECIAL_TOKENS),
    )
    inner.train_from_iterator(CORPUS, trainer=trainer)
    return CarinaTokenizer(inner=inner)


@pytest.fixture(scope="module")
def model_and_tokenizer(tmp_path_factory):
    tok = _train_test_tokenizer(tmp_path_factory)
    cfg = CarinaConfig(
        vocab_size=tok.vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    model = CarinaForCausalLM(cfg).eval()
    return model, tok, cfg


def test_generate_non_empty(model_and_tokenizer) -> None:
    model, tok, _ = model_and_tokenizer
    prompt = tok.apply_chat_template([{"role": "user", "content": "Hi"}])
    text = "".join(list(generate(model, tok, prompt, max_new_tokens=8, temperature=0.7)))
    assert text.strip() != ""


def test_generate_respects_max_new_tokens(model_and_tokenizer) -> None:
    model, tok, _ = model_and_tokenizer
    prompt = tok.apply_chat_template([{"role": "user", "content": "Hi"}])
    text = "".join(generate(model, tok, prompt, max_new_tokens=5, temperature=0.0))
    ids = tok.encode(text)
    assert len(ids) <= 5


def test_generate_temperature_variation(model_and_tokenizer) -> None:
    model, tok, _ = model_and_tokenizer
    prompt = tok.apply_chat_template([{"role": "user", "content": "Hi"}])
    outputs = set()
    for seed in range(5):
        torch.manual_seed(seed)
        text = "".join(generate(model, tok, prompt, max_new_tokens=6, temperature=1.0))
        outputs.add(text)
    assert len(outputs) >= 2


def test_sampling_top_k_top_p_greedy() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[1.0, 5.0, 3.0, 0.0]])
    greedy = sample_top_k_top_p(logits.clone(), temperature=0.0)
    assert int(greedy.item()) == 1
    top_k1 = sample_top_k_top_p(logits.clone(), top_k=1, temperature=1.0)
    assert int(top_k1.item()) == 1


def test_generate_only_valid_ids(model_and_tokenizer) -> None:
    model, tok, _ = model_and_tokenizer
    prompt = tok.apply_chat_template([{"role": "user", "content": "Hello"}])
    text = "".join(list(generate(model, tok, prompt, max_new_tokens=4, temperature=0.6)))
    ids = tok.encode(text)
    for i in ids:
        assert 0 <= int(i) < tok.vocab_size
