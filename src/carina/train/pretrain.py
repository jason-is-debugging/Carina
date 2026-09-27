"""Pretraining entry point for Carina.

Mirrors :mod:`carina.train.sft` but uses no label masking (every token
is supervised) and the :class:`DummyPretrainDataset`/JSONL loader for
unlabelled text. Run via::

    python -m carina.train.pretrain --config configs.pretrain
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ..model import CarinaForCausalLM
from ..utils import is_main_process
from .builder import (
    build_amp,
    build_tokenizer_and_config,
    load_config_from_args,
    make_arg_parser,
)
from .checkpoint import CarinaCheckpoint
from .dataset import DummyPretrainDataset, PretrainDataset
from .epoch import run_epoch
from .logging import finish, log, setup_logging, setup_wandb, shutdown_logging
from .optimizer import build_optimizer, build_scheduler
from .trainer_utils import init_distributed_mode


def _load_dataset(args, tokenizer):
    if getattr(args, "dummy_data", False):
        return DummyPretrainDataset(
            tokenizer, num_samples=args.dummy_samples, max_length=args.max_seq_len
        )
    return PretrainDataset(Path(args.data_path), max_length=args.max_seq_len)


def _save_final(model, optimizer, scheduler, step_global, epochs, save_dir, config, tokenizer, wandb_run):
    final_dir = save_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = CarinaCheckpoint(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        step=step_global,
        epoch=epochs,
        wandb_run_id=getattr(wandb_run, "id", None),
        config_json={"vocab_size": config.vocab_size},
    )
    final_ckpt.save(final_dir)
    torch.save(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        final_dir / "model.pt",
    )
    config.save(final_dir / "carina_config.json")
    tokenizer.save(str(final_dir / "tokenizer"))


def _build_sampler(train_ds):
    if torch.distributed.is_initialized():
        from torch.utils.data import DistributedSampler

        return DistributedSampler(train_ds)
    return torch.randperm(len(train_ds)).tolist()


def _build_optimizer_and_scheduler(model, args, train_ds):
    import math

    optimizer = build_optimizer(model, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_micro = max(1, math.ceil(len(train_ds) / max(1, args.batch_size)))
    total_steps = max(1, total_micro // max(1, args.accumulation_steps) * args.epochs)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    scheduler = build_scheduler(
        optimizer, warmup_steps, total_steps, min_lr_ratio=args.min_lr_ratio
    )
    return optimizer, scheduler


def _setup_run(args):
    init_distributed_mode()
    device = torch.device(args.resolve_device())
    tokenizer, config = build_tokenizer_and_config(args)
    model = CarinaForCausalLM(config).to(device)
    train_ds = _load_dataset(args, tokenizer)
    train_sampler = _build_sampler(train_ds)
    optimizer, scheduler = _build_optimizer_and_scheduler(model, args, train_ds)
    autocast_ctx, scaler = build_amp(args.resolve_dtype(), device.type)
    start_epoch, start_step, resume_ckpt = 0, 0, None
    if args.resume_dir:
        ckpt = CarinaCheckpoint.load(args.resume_dir, model=model, map_location=device)
        if ckpt.optimizer_state is not None:
            optimizer.load_state_dict(ckpt.optimizer_state)
        if ckpt.scheduler_state is not None:
            scheduler.load_state_dict(ckpt.scheduler_state)
        start_epoch, start_step = ckpt.epoch, ckpt.step
        log(f"Resumed from {args.resume_dir} @ step {start_step}")
        resume_ckpt = ckpt
    args = replace(args, start_epoch=start_epoch)
    wandb_run = setup_wandb(
        project=args.wandb_project,
        name=args.wandb_name or args.run_name,
        config={"args": args.as_dict()},
        run_id=resume_ckpt.wandb_run_id if resume_ckpt else args.wandb_run_id,
        enabled=args.use_wandb,
    )
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    config.save(save_dir / "carina_config.json")
    return {
        "device": device,
        "tokenizer": tokenizer,
        "config": config,
        "model": model,
        "train_ds": train_ds,
        "train_sampler": train_sampler,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "autocast_ctx": autocast_ctx,
        "scaler": scaler,
        "start_epoch": start_epoch,
        "start_step": start_step,
        "wandb_run": wandb_run,
        "save_dir": save_dir,
    }


def _train_epochs(args, ctx, epoch_state):
    for epoch in range(ctx["start_epoch"], args.epochs):
        epoch_state["step_global"] = run_epoch(
            epoch,
            args,
            ctx["model"],
            ctx["train_ds"],
            ctx["train_sampler"],
            ctx["optimizer"],
            ctx["scheduler"],
            ctx["autocast_ctx"],
            ctx["scaler"],
            ctx["device"],
            ctx["start_step"],
            ctx["wandb_run"],
            ctx["save_dir"],
            ctx["config"],
            epoch_state["step_global"],
        )
        if args.max_steps and epoch_state["step_global"] >= args.max_steps:
            break


def train(args) -> dict:
    setup_logging(args.log_dir, args.run_name or "carina-pretrain")
    ctx = _setup_run(args)
    epoch_state = {"step_global": ctx["start_step"]}
    _train_epochs(args, ctx, epoch_state)
    step_global = epoch_state["step_global"]
    if is_main_process():
        _save_final(
            ctx["model"], ctx["optimizer"], ctx["scheduler"],
            step_global, args.epochs, ctx["save_dir"], ctx["config"],
            ctx["tokenizer"], ctx["wandb_run"],
        )
    finish(ctx["wandb_run"])
    shutdown_logging()
    return {"final_step": step_global, "epochs_run": args.epochs, "save_dir": str(ctx["save_dir"] / "final")}


def main(argv: list[str] | None = None) -> int:
    parser = make_arg_parser(description="Carina pretraining.")
    args = load_config_from_args(
        parser,
        default_module="configs.pretrain",
        config_class=__import__("configs.pretrain", fromlist=["PretrainConfig"]).PretrainConfig,
    )
    summary = train(args)
    log(json.dumps(summary, indent=2))
    return 0


__all__ = ["DummyPretrainDataset", "PretrainDataset", "main", "train"]
