"""Per-epoch SFT training helpers."""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils import is_main_process
from .checkpoint import CarinaCheckpoint
from .logging import log, log_metrics
from .trainer_utils import SkipBatchSampler


def _step_batch(
    model,
    batch,
    device,
    autocast_ctx,
    scaler,
    accumulation_steps: int,
    label_smoothing: float = 0.0,
) -> tuple[float, bool]:
    """Forward + backward pass for one micro-batch."""
    input_ids, labels = batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    with autocast_ctx:
        if label_smoothing > 0:
            # Bypass the model's internal cross-entropy so we can apply smoothing.
            out = model(input_ids=input_ids, labels=None)
            logits = out["logits"]  # (B, T, V)
            # Mask out user-token positions; only supervise assistant tokens.
            mask = labels.ne(-100)
            # Flatten for F.cross_entropy: (B*T,) targets, (B*T, V) logits.
            flat_logits = logits.view(-1, logits.size(-1))
            flat_labels = labels.clamp(min=0)  # replace -100 with 0 (won't be masked anyway)
            loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")
            loss = (loss * mask.view(-1).float()).sum() / mask.sum().clamp(min=1)
            # Label smoothing: mix uniform distribution with hard labels.
            # smoothed = (1 - smoothing) * hard + smoothing * uniform
            log_probs = F.log_softmax(flat_logits, dim=-1)
            smooth_loss = (-log_probs.mean(dim=-1) * mask.view(-1).float()).sum() / mask.sum().clamp(min=1)
            loss = (1 - label_smoothing) * loss + label_smoothing * smooth_loss
        else:
            out = model(input_ids=input_ids, labels=labels)
            loss = out["loss"]
        if loss is None:
            return 0.0, False
        loss_for_log = float(loss.detach().item())
        scaled = loss / accumulation_steps
    if scaler is not None:
        scaler.scale(scaled).backward()
    else:
        scaled.backward()
    return loss_for_log, True


def _maybe_optimizer_step(
    accum_count: int,
    accumulation_steps: int,
    is_last_batch: bool,
    scaler,
    optimizer,
    scheduler,
    grad_clip: float,
    model,
) -> bool:
    """Step the optimiser when accumulation is complete; return True on step."""
    if accum_count != accumulation_steps and not is_last_batch:
        return False
    if scaler is not None:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return True


def _save_checkpoint(
    model,
    optimizer,
    scheduler,
    step_global: int,
    epoch: int,
    save_dir: Path,
    config,
    wandb_run,
) -> None:
    if not is_main_process():
        return
    ckpt = CarinaCheckpoint(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        step=step_global,
        epoch=epoch,
        wandb_run_id=getattr(wandb_run, "id", None),
        config_json={"vocab_size": config.vocab_size},
    )
    ckpt.save(save_dir / "checkpoint")


def _log_progress(
    epoch: int,
    step: int,
    total_micro: int,
    accumulate_step_global: int,
    avg_loss: float,
    optimizer,
    epoch_start: float,
    loader_len: int,
    wandb_run,
) -> None:
    cur_lr = optimizer.param_groups[0]["lr"]
    elapsed = (time.time() - epoch_start) / 60.0
    eta = elapsed * (loader_len - step) / max(1, step)
    log(
        f"epoch={epoch + 1} step={step}/{total_micro} "
        f"global={accumulate_step_global} loss={avg_loss:.4f} "
        f"lr={cur_lr:.2e} eta={eta:.1f}m"
    )
    log_metrics(
        wandb_run,
        {"train/loss": avg_loss, "train/lr": cur_lr},
        step=accumulate_step_global,
    )


def _post_step_actions(
    epoch: int,
    args,
    optimizer,
    scheduler,
    scaler,
    model,
    wandb_run,
    save_dir: Path,
    config,
    step_global: int,
    step: int,
    total_micro: int,
    accum_count: int,
    loss_accum: float,
    loader_len: int,
    epoch_start: float,
) -> tuple[int, float, int]:
    """Handle accumulator finalisation, logging, and checkpointing."""
    accum_count = 0
    avg_loss = loss_accum / max(1, args.accumulation_steps)
    loss_accum = 0.0
    if step_global % args.log_interval == 0 or step == total_micro:
        _log_progress(
            epoch,
            step,
            total_micro,
            step_global,
            avg_loss,
            optimizer,
            epoch_start,
            loader_len,
            wandb_run,
        )
    if step % args.save_interval == 0 or step == total_micro:
        _save_checkpoint(
            model,
            optimizer,
            scheduler,
            step_global,
            epoch,
            save_dir,
            config,
            wandb_run,
        )
    return step_global, loss_accum, accum_count


def _epoch_inner_loop(
    epoch: int,
    args,
    model,
    loader,
    total_micro: int,
    optimizer,
    scheduler,
    autocast_ctx,
    scaler,
    device,
    wandb_run,
    save_dir: Path,
    config,
    step_global: int,
    epoch_start: float,
) -> int:
    accum_count = 0
    loss_accum = 0.0
    for step, batch in enumerate(loader, start=1):
        loss_for_log, contributed = _step_batch(
            model, batch, device, autocast_ctx, scaler, args.accumulation_steps,
            getattr(args, "label_smoothing", 0.0),
        )
        if contributed:
            loss_accum += loss_for_log
            accum_count += 1
            step_global += 1
        is_last = step == total_micro
        stepped = _maybe_optimizer_step(
            accum_count,
            args.accumulation_steps,
            is_last,
            scaler,
            optimizer,
            scheduler,
            args.grad_clip,
            model,
        )
        if stepped:
            step_global, loss_accum, accum_count = _post_step_actions(
                epoch,
                args,
                optimizer,
                scheduler,
                scaler,
                model,
                wandb_run,
                save_dir,
                config,
                step_global,
                step,
                total_micro,
                accum_count,
                loss_accum,
                len(loader),
                epoch_start,
            )
        if args.max_steps and step_global >= args.max_steps:
            break
    return step_global


def run_epoch(
    epoch: int,
    args,
    model,
    train_ds,
    train_sampler,
    optimizer,
    scheduler,
    autocast_ctx,
    scaler,
    device,
    start_step: int,
    wandb_run,
    save_dir: Path,
    config,
    step_global: int,
) -> int:
    """Train for one epoch; return the updated ``step_global``."""
    if torch.distributed.is_initialized():
        train_sampler.set_epoch(epoch)
    skip = start_step if (epoch == args.start_epoch and start_step > 0) else 0
    batch_sampler = SkipBatchSampler(train_sampler, args.batch_size, skip)
    loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    log(
        f"Epoch {epoch + 1}/{args.epochs}: {len(loader)} micro-batches "
        f"(skip={skip}, accum={args.accumulation_steps})"
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_micro = len(loader) + skip
    epoch_start = time.time()
    return _epoch_inner_loop(
        epoch,
        args,
        model,
        loader,
        total_micro,
        optimizer,
        scheduler,
        autocast_ctx,
        scaler,
        device,
        wandb_run,
        save_dir,
        config,
        step_global,
        epoch_start,
    )


__all__ = ["run_epoch"]
