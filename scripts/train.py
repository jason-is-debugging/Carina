"""Pretrain/SFT entry: config-driven, uses new model/datasets/train.

Single route: SGD + whole model bf16, no precision/optimizer branches.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from carina.config import (
    DataConfig,
    MiscConfig,
    ModelConfig,
    TokenizerConfig,
    TrainingConfig,
)
from carina.datasets import (
    SftDataset,
    TokenWindowDataset,
    build_token_cache,
)
from carina.model.carina_casual_model import CarinaCausalModel
from carina.tokenizer import ensure_tokenizer
from carina.train.trainer import Trainer, TrainerOptions
from carina.utils import (
    count_parameters,
    get_device,
    peak_memory_gb,
    set_seed,
)
from carina.utils.logger import make_logger


def parse_args() -> argparse.Namespace:
    """Parse mode + step overrides."""
    parser = argparse.ArgumentParser(description="Train Carina causal LM")
    parser.add_argument(
        "--mode", choices=["pretrain", "finetune"], default="pretrain")
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def build_datasets(mode: str, tok, seq_len: int, pad_id: int, ign: int,
                   logger=None):
    """Build train/val datasets from processed JSONL per mode."""
    log = logger or print
    eos = int(tok.eos_token_id)
    pack = bool(DataConfig.pack)
    if mode == "pretrain":
        # Pretraining goes through the token cache: with 22M tokens
        # PretrainDataset would keep two copies of Python int lists in RAM
        # (~2GB), and a 14GB box gets the process silently SIGKILLed by the
        # OOM killer (no error message). The cache stores one int32 numpy
        # stream (90MB) and windows are cut from it.
        cache = Path(DataConfig.pretrain_token_cache)
        corpus = Path(DataConfig.token_cache_corpus)
        if not cache.exists() or corpus.stat().st_mtime > cache.stat().st_mtime:
            log(f"building token cache -> {cache}")
            build_token_cache(corpus, cache, tok, eos,
                              max_chars=int(DataConfig.token_cache_max_chars))
        import numpy as np
        flat = np.load(cache, mmap_mode="r")
        n_val_tok = int(len(flat) * DataConfig.val_ratio)
        train_ds = TokenWindowDataset(flat, seq_len, 0, len(flat) - n_val_tok)
        val_ds = TokenWindowDataset(flat, seq_len, len(flat) - n_val_tok, len(flat))
        log(f"token cache windows: train={len(train_ds)} val={len(val_ds)}")
        return train_ds, val_ds
    path = Path(DataConfig.finetune_processed_saved_path)
    full = SftDataset(path, seq_len, tok, pad_id=pad_id, ignore_index=ign,
                      pack=pack, eos_id=eos)
    assert len(full) > 1, f"[train] dataset too small: {path}"
    n_val = max(1, int(len(full) * DataConfig.val_ratio))
    n_train = len(full) - n_val
    gen = torch.Generator().manual_seed(MiscConfig.seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=gen)
    return train_ds, val_ds


def build_model(vocab_size: int, ign: int) -> CarinaCausalModel:
    """Build CarinaCausalModel from ModelConfig."""
    return CarinaCausalModel(
        vocab_size=vocab_size,
        d_model=ModelConfig.d_model,
        d_hidden=ModelConfig.d_ff,
        n_heads=ModelConfig.n_heads,
        max_loop_times=ModelConfig.max_loop_times,
        lora_reusage=ModelConfig.lora_reusage,
        ranks=tuple(ModelConfig.ranks),
        dropout=ModelConfig.dropout,
        tie=ModelConfig.tie_embeddings,
        ignore_index=ign,
    )


def main() -> None:
    """Run training, save checkpoints + metrics."""
    args = parse_args()
    set_seed(MiscConfig.seed)
    device = get_device()
    save_dir = Path(TrainingConfig.model_saved_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = make_logger("train", str(save_dir / "train.log"))
    tok = ensure_tokenizer(TokenizerConfig.hf_id,
                           TokenizerConfig.tokenizer_dir,
                           TokenizerConfig.trust_remote_code)
    pad_id = int(tok.pad_token_id)
    ign = int(TokenizerConfig.label_ignore_token)
    vocab_size = len(tok)
    logger.info(f"mode={args.mode} vocab={vocab_size} pad={pad_id} dtype=bf16")
    train_ds, val_ds = build_datasets(
        args.mode, tok, DataConfig.seq_len, pad_id, ign, logger)
    train_loader = DataLoader(train_ds, batch_size=DataConfig.batch_size,
                              shuffle=True,
                              num_workers=DataConfig.num_workers)
    val_loader = DataLoader(val_ds, batch_size=DataConfig.batch_size,
                            num_workers=DataConfig.num_workers)
    model = build_model(vocab_size, ign).to(device).to(torch.bfloat16)
    if TrainingConfig.init_from:
        state = torch.load(TrainingConfig.init_from, map_location=device,
                           weights_only=True)
        model.load_state_dict(state)
        logger.info(f"init_from={TrainingConfig.init_from}")
    total, trainable = count_parameters(model)
    logger.info(f"params total={total} trainable={trainable}")
    max_steps = args.max_steps or TrainingConfig.max_steps
    trainer = Trainer(model, TrainerOptions(
        lr=TrainingConfig.learning_rate,
        weight_decay=TrainingConfig.weight_decay,
        grad_clip=TrainingConfig.grad_clip,
        max_steps=max_steps,
        eval_interval=TrainingConfig.eval_interval,
        warmup_steps=TrainingConfig.warmup_steps,
        min_lr_ratio=TrainingConfig.min_lr_ratio,
        fp32_master=TrainingConfig.fp32_master,
        save_dir=str(save_dir)))
    start = time.time()
    try:
        history = trainer.fit(train_loader, val_loader, log_fn=logger.info)
    except torch.cuda.OutOfMemoryError as err:
        logger.error(f"OOM: {err}. Lower batch/seq.")
        raise
    elapsed = time.time() - start
    peak = peak_memory_gb(device)
    logger.info(f"done best_ppl={history['best_ppl']:.3f} "
                f"time={elapsed:.1f}s peak_gb={peak:.2f}")
    with open(save_dir / "metrics.md", "w") as fh:
        fh.write(f"# Metrics ({args.mode})\n\n")
        fh.write(f"best_ppl={history['best_ppl']:.3f}\n")
        fh.write(f"steps={history['steps']}\nlosses={history['losses']}\n")
        fh.write(f"ppls={history['ppls']}\n")
        fh.write(f"final_train_loss={history.get('final_train_loss', float('nan'))}\n")


if __name__ == "__main__":
    main()
