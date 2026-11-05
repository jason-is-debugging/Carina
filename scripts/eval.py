"""Evaluation: loss + perplexity over the val split, config-driven."""
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
from carina.datasets import PretrainDataset, SftDataset
from carina.model.carina_casual_model import CarinaCausalModel
from carina.tokenizer import ensure_tokenizer
from carina.train.trainer import Trainer, TrainerOptions
from carina.utils import get_device, peak_memory_gb, set_seed
from carina.utils.logger import make_logger


def parse_args() -> argparse.Namespace:
    """Parse dataset mode."""
    parser = argparse.ArgumentParser(description="Evaluate Carina causal LM")
    parser.add_argument(
        "--mode", choices=["pretrain", "finetune"], default="pretrain")
    return parser.parse_args()


def build_val_loader(mode: str, tok, seq_len: int, pad_id: int, ign: int):
    """Build val loader with the same seed/split as train."""
    eos = int(tok.eos_token_id)
    pack = bool(DataConfig.pack)
    if mode == "pretrain":
        path = Path(DataConfig.pretrain_processed_saved_path)
        full = PretrainDataset(path, seq_len, tok, pad_id=pad_id,
                               ignore_index=ign, pack=pack, eos_id=eos)
    else:
        path = Path(DataConfig.finetune_processed_saved_path)
        full = SftDataset(path, seq_len, tok, pad_id=pad_id, ignore_index=ign,
                          pack=pack, eos_id=eos)
    assert len(full) > 1, f"[eval] dataset too small: {path}"
    n_val = max(1, int(len(full) * DataConfig.val_ratio))
    n_train = len(full) - n_val
    gen = torch.Generator().manual_seed(MiscConfig.seed)
    _, val_ds = random_split(full, [n_train, n_val], generator=gen)
    return DataLoader(val_ds, batch_size=DataConfig.batch_size,
                      num_workers=DataConfig.num_workers)


def build_model(vocab_size: int, ign: int) -> CarinaCausalModel:
    """Build CarinaCausalModel from ModelConfig (same as train)."""
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
    """Load best checkpoint and report loss/ppl/speed."""
    args = parse_args()
    set_seed(MiscConfig.seed)
    device = get_device()
    save_dir = Path(TrainingConfig.model_saved_dir)
    logger = make_logger("eval", str(save_dir / "eval.log"))
    tok = ensure_tokenizer(TokenizerConfig.hf_id,
                           TokenizerConfig.tokenizer_dir,
                           TokenizerConfig.trust_remote_code)
    pad_id = int(tok.pad_token_id)
    ign = int(TokenizerConfig.label_ignore_token)
    loader = build_val_loader(args.mode, tok, DataConfig.seq_len,
                              pad_id, ign)
    model = build_model(len(tok), ign).to(device).to(torch.bfloat16)
    try:
        state = torch.load(save_dir / "best.pt", map_location=device,
                           weights_only=True)
        model.load_state_dict(state)
        logger.info("loaded best checkpoint")
    except FileNotFoundError:
        logger.warning("no checkpoint, evaluating random weights")
    trainer = Trainer(model, TrainerOptions(
        lr=TrainingConfig.learning_rate,
        save_dir=str(save_dir)))
    start = time.time()
    mean_loss, ppl = trainer.evaluate(loader)
    elapsed = max(time.time() - start, 1e-6)
    line = (f"mode={args.mode} loss={mean_loss:.4f} ppl={ppl:.3f} "
            f"time={elapsed:.1f}s peak_gb={peak_memory_gb(device):.2f}")
    logger.info(line)
    with open(save_dir / "eval.md", "w") as fh:
        fh.write("# Eval\n\n" + line + "\n")


if __name__ == "__main__":
    main()
