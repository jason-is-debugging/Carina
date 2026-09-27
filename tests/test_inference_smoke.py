"""Smoke test for inference: load checkpoint, run, check determinism."""
from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from carina.config import CarinaConfig
from carina.inference.generate import _load_model_and_tokenizer, generate
from carina.model import CarinaForCausalLM
from carina.tokenizer import SPECIAL_TOKENS, CarinaTokenizer
from carina.train.checkpoint import CarinaCheckpoint

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


def _train_small_tokenizer() -> CarinaTokenizer:
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


def _save_checkpoint(tmp_path: Path) -> Path:
    tokenizer = _train_small_tokenizer()
    cfg = CarinaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=64,
    )
    model = CarinaForCausalLM(cfg)
    save_dir = tmp_path / "ckpt"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CarinaCheckpoint(
        model_state=model.state_dict(),
        config_json={"vocab_size": cfg.vocab_size},
    )
    ckpt.save(save_dir)
    torch.save(model.state_dict(), save_dir / "model.pt")
    cfg.save(save_dir / "carina_config.json")
    tokenizer.save(str(save_dir / "tokenizer"))
    return save_dir


def test_inference_smoke(tmp_path: Path) -> None:
    ckpt = _save_checkpoint(tmp_path)
    model, tokenizer, _ = _load_model_and_tokenizer(ckpt)

    messages = [{"role": "user", "content": "Hi"}]
    prompt = tokenizer.apply_chat_template(messages)
    torch.manual_seed(0)
    text_a = "".join(generate(model, tokenizer, prompt, max_new_tokens=8, temperature=0.7))
    torch.manual_seed(0)
    text_b = "".join(generate(model, tokenizer, prompt, max_new_tokens=8, temperature=0.7))
    assert text_a == text_b, "Same RNG state must produce identical outputs"
    assert text_a.strip() != "", "Sampling produced empty text"


def test_greedy_decoding_deterministic(tmp_path: Path) -> None:
    """Greedy decoding must be deterministic regardless of seed."""
    ckpt = _save_checkpoint(tmp_path)
    model, tokenizer, _ = _load_model_and_tokenizer(ckpt)
    messages = [{"role": "user", "content": "Hi"}]
    prompt = tokenizer.apply_chat_template(messages)
    text_a = "".join(generate(model, tokenizer, prompt, max_new_tokens=8, temperature=0.0))
    text_b = "".join(generate(model, tokenizer, prompt, max_new_tokens=8, temperature=0.0))
    assert text_a == text_b, "Greedy decoding must be deterministic"


def test_inference_can_be_called_with_only_tokenizer_dir(tmp_path: Path) -> None:
    """The loader should also accept a directory containing only the tokenizer."""
    tokenizer = _train_small_tokenizer()
    cfg = CarinaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=64,
    )
    model = CarinaForCausalLM(cfg)
    save_dir = tmp_path / "ckpt"
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_dir))
    cfg.save(save_dir / "carina_config.json")
    torch.save(model.state_dict(), save_dir / "model.pt")
    loaded, tok2, cfg2 = _load_model_and_tokenizer(save_dir)
    assert cfg2.vocab_size == tokenizer.vocab_size
    assert tok2.vocab_size == tokenizer.vocab_size
    assert loaded is not None
