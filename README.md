# Carina

A small, fully-typed virtual-companion language model.

Carina is a decoder-only transformer (RMSNorm + RoPE + SwiGLU + GQA) built
in pure PyTorch for training a warm, helpful chat companion on a single
GPU or CPU. Configurations live as Python dataclasses; the trainer ships
with checkpointing, resume, early stopping, optional wandb logging, and a
Gradio demo.

## Layout

```
src/carina/
  config.py            CarinaConfig dataclass
  model/               rmsnorm, rotary, embedding, mlp, attention, layers, model
  tokenizer.py         CarinaTokenizer (BPE + chat template + label mask)
  train/               dataset, optimizer, checkpoint, logging, sft, pretrain
  inference/           sampling, streaming, generate, web_demo (Gradio)
  utils/               shared RNG / metrics / JSONL helpers
configs/               BaseConfig + PretrainConfig + SFTConfig + TokenizerConfig
tests/                 unit + smoke tests
scripts/               CLI entry points
```

## Quick start

```bash
# Train the BPE tokenizer (offline synthetic corpus by default)
uv run python scripts/train_tokenizer.py --output-dir models/carina_tokenizer

# Run SFT on a dummy dataset for 2 steps (CPU-friendly smoke test)
uv run python scripts/train_sft.py --config configs.sft --set max_steps=2 --set dummy_data=true

# Chat against a trained checkpoint
uv run python scripts/generate.py --checkpoint checkpoints/carina_sft/final

# Launch the Gradio chat UI
uv run python scripts/web_demo.py --checkpoint checkpoints/carina_sft/final
```

## Status checks

```bash
uv run pytest tests/ -v
uv run python scripts/check_style.py
uv run ruff check src tests scripts configs
```

## Configuration

Configs are pure Python dataclasses — no JSON for in-repo state.
Override any field from the CLI:

```bash
uv run python scripts/train_sft.py \
    --config configs.sft \
    --set training.epochs=3 \
    --set optimizer.learning_rate=1.0e-4 \
    --set dummy_data=true
```

## Architecture highlights

* Pre-norm transformer decoder with `RMSNorm`
* Rotary positional embeddings (RoPE) with pre-computed cos/sin tables
* SwiGLU MLP (`down_proj(SiLU(gate_proj) * up_proj)`)
* Multi-head attention with grouped-query attention support
  (`num_key_value_heads` can be < `num_attention_heads`)
* KV cache support for incremental decoding
* Optional tied input/output embeddings

## License

See `LICENSE`.
