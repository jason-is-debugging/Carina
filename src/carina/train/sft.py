"""Supervised fine-tuning entry point for Carina.

Run via ``python -m carina.train.sft --config configs.sft --set epochs=3``.

The loop mirrors the MiniMind pattern: per-micro-batch forward /
backward, gradient accumulation, cosine LR schedule with warmup, and
periodic checkpointing with full resume support (model + optimizer +
scheduler + step + epoch).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Subset

from ..model import CarinaForCausalLM
from ..utils import is_main_process
from .builder import (
    build_amp,
    build_tokenizer_and_config,
    load_config_from_args,
    make_arg_parser,
)
from .checkpoint import CarinaCheckpoint
from .dataset import DummySFTDataset, load_jsonl_dataset
from .epoch import run_epoch
from .evaluation import evaluate
from .logging import finish, log, log_metrics, setup_logging, setup_wandb, shutdown_logging
from .optimizer import build_optimizer, build_scheduler
from .trainer_utils import init_distributed_mode


def _load_dataset(args, tokenizer):
    if getattr(args, "dummy_data", False):
        return DummySFTDataset(
            tokenizer,
            num_samples=args.dummy_samples,
            max_length=args.max_seq_len,
        )
    return load_jsonl_dataset(
        args.data_path,
        tokenizer,
        max_length=args.max_seq_len,
        system=args.system_prompt,
    )


def _maybe_split_val(train_ds, val_split: float):
    if val_split <= 0 or len(train_ds) <= 1:
        return train_ds, None
    n_val = max(1, int(len(train_ds) * val_split))
    indices = list(range(len(train_ds)))
    return Subset(train_ds, indices[n_val:]), Subset(train_ds, indices[:n_val])


def _load_resume(args, model, optimizer, scheduler, device):
    start_epoch, start_step = 0, 0
    if not args.resume_dir:
        return start_epoch, start_step, None
    try:
        ckpt = CarinaCheckpoint.load(args.resume_dir, model=model, map_location=device)
        if ckpt.optimizer_state is not None:
            optimizer.load_state_dict(ckpt.optimizer_state)
        if ckpt.scheduler_state is not None:
            scheduler.load_state_dict(ckpt.scheduler_state)
        log(f"Resumed from {args.resume_dir} @ step {ckpt.step}, ep {ckpt.epoch}")
        return ckpt.epoch, ckpt.step, ckpt
    except FileNotFoundError:
        log(f"No resume ckpt at {args.resume_dir}, starting fresh")
    return start_epoch, start_step, None


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


def _save_best(model, optimizer, scheduler, step_global, epoch, save_dir, config, wandb_run):
    """Save the model weights when val loss improves (no optimizer/scheduler overhead)."""
    from .checkpoint import CarinaCheckpoint
    best_dir = save_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CarinaCheckpoint(
        model_state=model.state_dict(),
        optimizer_state=None,
        scheduler_state=None,
        step=step_global,
        epoch=epoch,
        wandb_run_id=getattr(wandb_run, "id", None),
        config_json={"vocab_size": config.vocab_size},
    )
    ckpt.save(best_dir)


def _build_sampler(train_ds):
    if torch.distributed.is_initialized():
        from torch.utils.data import DistributedSampler

        return DistributedSampler(train_ds)
    return torch.randperm(len(train_ds)).tolist()


def _build_optimizer_and_scheduler(model, args, train_ds):
    import math

    optimizer = build_optimizer(
        model, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_micro = max(1, math.ceil(len(train_ds) / max(1, args.batch_size)))
    total_steps = max(
        1, total_micro // max(1, args.accumulation_steps) * args.epochs
    )
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=args.min_lr_ratio)
    return optimizer, scheduler


def _setup_run(args):
    """Construct the trainer components needed before the loop begins."""
    init_distributed_mode()
    device = torch.device(args.resolve_device())
    tokenizer, config = build_tokenizer_and_config(args)
    model = CarinaForCausalLM(config).to(device)
    for layer in model.model.layers:
        layer._gradient_checkpointing = True
    train_ds = _load_dataset(args, tokenizer)
    train_ds, val_ds = _maybe_split_val(train_ds, args.val_split)
    train_sampler = _build_sampler(train_ds)
    optimizer, scheduler = _build_optimizer_and_scheduler(model, args, train_ds)
    autocast_ctx, scaler = build_amp(args.resolve_dtype(), device.type)
    start_epoch, start_step, resume_ckpt = _load_resume(
        args, model, optimizer, scheduler, device
    )
    args.start_epoch = start_epoch
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
        "val_ds": val_ds,
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


def _should_stop(history, best_val, no_improve, args):
    if not history:
        return False, best_val, no_improve
    latest = history[-1]["val_loss"]
    if latest < best_val - args.min_delta:
        return False, latest, 0
    no_improve += 1
    return no_improve >= args.patience, best_val, no_improve


def _run_validation(args, model, val_ds, device, autocast_ctx, wandb_run, step_global):
    from torch.utils.data import DataLoader

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loss = evaluate(model, val_loader, device, autocast_ctx, max_batches=20)
    log_metrics(wandb_run, {"val/loss": val_loss}, step=step_global)
    return val_loss


def _train_epochs(args, ctx, epoch_state):
    """Drive the per-epoch loop; return True when early stopping kicks in."""
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
        if ctx["val_ds"] is None or args.epochs <= 0:
            continue
        val_loss = _run_validation(
            args, ctx["model"], ctx["val_ds"], ctx["device"],
            ctx["autocast_ctx"], ctx["wandb_run"], epoch_state["step_global"],
        )
        log(f"epoch={epoch + 1} val_loss={val_loss:.4f}")
        epoch_state["history"].append({"epoch": epoch + 1, "val_loss": val_loss})
        stop, best_val, epoch_state["no_improve"] = _should_stop(
            epoch_state["history"],
            epoch_state["best_val"],
            epoch_state["no_improve"],
            args,
        )
        # Save best model whenever val loss improves.
        if val_loss < epoch_state["best_val"]:
            epoch_state["best_val"] = val_loss
            _save_best(
                ctx["model"],
                ctx["optimizer"],
                ctx["scheduler"],
                epoch_state["step_global"],
                epoch + 1,
                ctx["save_dir"],
                ctx["config"],
                ctx["wandb_run"],
            )
            log(f"★ best model saved (val_loss={val_loss:.4f})")
        if stop:
            log("Early stopping triggered.")
            return True
        if args.max_steps and epoch_state["step_global"] >= args.max_steps:
            log("Reached max_steps, stopping.")
            return True
    return False


def train(args) -> dict:
    """Run the SFT loop described by ``args``; returns a summary dict."""
    setup_logging(args.log_dir, args.run_name or "carina-sft")
    ctx = _setup_run(args)
    epoch_state = {
        "step_global": ctx["start_step"],
        "best_val": float("inf"),
        "no_improve": 0,
        "history": [],
    }
    _train_epochs(args, ctx, epoch_state)
    step_global = epoch_state["step_global"]
    if is_main_process():
        _save_final(
            ctx["model"],
            ctx["optimizer"],
            ctx["scheduler"],
            step_global,
            args.epochs,
            ctx["save_dir"],
            ctx["config"],
            ctx["tokenizer"],
            ctx["wandb_run"],
        )
    finish(ctx["wandb_run"])
    shutdown_logging()
    return {
        "final_step": step_global,
        "epochs_run": args.epochs,
        "save_dir": str(ctx["save_dir"] / "final"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = make_arg_parser(description="Carina SFT training.")
    args = load_config_from_args(parser, default_module="configs.sft", config_class=__import__("configs.sft", fromlist=["SFTConfig"]).SFTConfig)
    summary = train(args)
    log(json.dumps(summary, indent=2))
    return 0


__all__ = ["main", "train"]
