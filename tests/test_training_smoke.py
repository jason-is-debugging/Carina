"""End-to-end smoke test for SFT training."""
from __future__ import annotations

from pathlib import Path

import torch

from carina.config import CarinaConfig
from carina.model import CarinaForCausalLM
from carina.tokenizer import build_default_tokenizer
from carina.train.checkpoint import CarinaCheckpoint
from carina.train.dataset import DummySFTDataset
from carina.train.optimizer import build_optimizer, build_scheduler


def _tiny_setup():
    tokenizer = build_default_tokenizer()
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
    optimizer = build_optimizer(model, lr=1e-3)
    scheduler = build_scheduler(optimizer, warmup_steps=1, total_steps=2)
    dataset = DummySFTDataset(tokenizer, num_samples=8, max_length=32)
    return model, optimizer, scheduler, dataset, cfg


def _run_two_steps(model, optimizer, scheduler, dataset) -> list[float]:
    losses: list[float] = []
    for step in range(2):
        input_ids, labels = dataset[step]
        out = model(input_ids=input_ids.unsqueeze(0), labels=labels.unsqueeze(0))
        loss = out["loss"]
        assert torch.isfinite(loss), f"step {step} loss is not finite"
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.item()))
    return losses


def _assert_params_changed(model, before: torch.Tensor) -> None:
    after = model.model.embed_tokens.token_embedding.weight.detach().clone()
    assert torch.any(before != after), "Embedding did not change across two SFT steps"


def _checkpoint_roundtrip(model, optimizer, scheduler, cfg, tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CarinaCheckpoint(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        step=2,
        epoch=1,
        wandb_run_id=None,
        config_json={"vocab_size": cfg.vocab_size},
    )
    ckpt.save(ckpt_dir)
    assert (ckpt_dir / "checkpoint.pt").is_file()
    new_model = CarinaForCausalLM(cfg)
    CarinaCheckpoint.load(ckpt_dir, model=new_model)
    for k, v in model.state_dict().items():
        assert torch.allclose(new_model.state_dict()[k], v), f"Mismatch after reload: {k}"


def test_sft_smoke_two_steps(tmp_path: Path) -> None:
    """Two SFT steps must finish, produce a finite loss, and update params."""
    model, optimizer, scheduler, dataset, cfg = _tiny_setup()
    before = model.model.embed_tokens.token_embedding.weight.detach().clone()
    losses = _run_two_steps(model, optimizer, scheduler, dataset)
    _assert_params_changed(model, before)
    assert all(torch.isfinite(torch.tensor(losses))), "Loss is not finite"
    _checkpoint_roundtrip(model, optimizer, scheduler, cfg, tmp_path)
