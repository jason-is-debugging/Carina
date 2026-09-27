"""Supervised fine-tuning for Carina (or resume from checkpoint).

Edit ``configs/sft.py`` to change hyperparameters.  No ``--set``
overrides are needed; run with::

    python scripts/train_sft.py                # resume if checkpoint exists
    python scripts/train_sft.py --fresh       # start from pretrain final
    python scripts/train_sft.py --fresh --from-pretrain path  # from specific ckpt

Interactive commands (via FIFO ``/tmp/carina_train_cmd_<pid>.fifo`` or stdin):
    sane  — stop at the next epoch boundary
    si    — stop immediately
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

import configs.sft as _cfg_module
from configs.sft import SFTConfig

from carina.config import CarinaConfig
from carina.model import CarinaForCausalLM
from carina.train.builder import build_amp, build_tokenizer_and_config
from carina.train.checkpoint import CarinaCheckpoint
from carina.train.dataset import DummySFTDataset, load_jsonl_dataset
from carina.train.evaluation import evaluate
from carina.train.logging import (
    finish,
    log,
    log_metrics,
    setup_logging,
    setup_wandb,
    shutdown_logging,
)
from carina.train.optimizer import build_optimizer, build_scheduler
from carina.train.trainer_utils import init_distributed_mode, SkipBatchSampler
from carina.utils import setup_seed
from carina.utils import is_main_process

from command_handler import (
    TrainingController,
    install_signal_handler,
    start,
    stop,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_CHECKPOINT_DIR = Path("checkpoints/carina_sft")
_CHECKPOINT_FILE = _CHECKPOINT_DIR / "checkpoint"
_FINAL_DIR = _CHECKPOINT_DIR / "final"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _auto_resume() -> tuple[Path | None, int, int]:
    if not _CHECKPOINT_FILE.is_dir():
        return None, 0, 0
    meta = _CHECKPOINT_FILE / "checkpoint.json"
    if not meta.is_file():
        return None, 0, 0
    data = json.loads(meta.read_text())
    return (
        _CHECKPOINT_FILE,
        int(data.get("epoch", 0)),
        int(data.get("step", 0)),
    )


def _find_pretrain_ckpt(from_pretrain: Path | None) -> Path | None:
    """Locate the pretrain final checkpoint to initialise from."""
    candidates = [
        from_pretrain,
        Path("checkpoints/carina_pretrain/final"),
        Path("checkpoints/carina_pretrain/checkpoint"),
    ]
    for p in candidates:
        if p is not None and (p / "checkpoint.pt").is_file():
            return p
        if p is not None and (p / "model.pt").is_file():
            return p
    return None


def _load_model_weights(model: CarinaForCausalLM, ckpt_path: Path, device) -> None:
    """Load model weights from a checkpoint directory or .pt file."""
    pt_file = ckpt_path / "checkpoint.pt"
    if not pt_file.is_file():
        pt_file = ckpt_path / "model.pt"
    raw = torch.load(str(pt_file), map_location=device, weights_only=False)
    state = raw.get("model", raw) if isinstance(raw, dict) else raw
    model.load_state_dict(state, strict=False)
    log(f"[sft] loaded weights from {ckpt_path}")


def _save_checkpoint(
    model,
    optimizer,
    scheduler,
    step_global: int,
    epoch: int,
    config,
    tokenizer,
    wandb_run,
    controller: TrainingController,
) -> None:
    if not is_main_process():
        return
    _CHECKPOINT_FILE.mkdir(parents=True, exist_ok=True)
    ckpt = CarinaCheckpoint(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        step=step_global,
        epoch=epoch,
        wandb_run_id=getattr(wandb_run, "id", None),
        config_json={"vocab_size": config.vocab_size},
        extra={"resume_from": controller.resume_from or ""},
    )
    ckpt.save(_CHECKPOINT_FILE)
    controller.update_resume(str(_CHECKPOINT_FILE), f"epoch {epoch}, step {step_global}")


def _save_final(
    model,
    optimizer,
    scheduler,
    step_global: int,
    epochs: int,
    config,
    tokenizer,
    wandb_run,
) -> None:
    if not is_main_process():
        return
    _FINAL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CarinaCheckpoint(
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        scheduler_state=scheduler.state_dict(),
        step=step_global,
        epoch=epochs,
        wandb_run_id=getattr(wandb_run, "id", None),
        config_json={"vocab_size": config.vocab_size},
    )
    ckpt.save(_FINAL_DIR)
    torch.save(
        {k: v.detach().cpu() for k, v in model.state_dict().items()},
        _FINAL_DIR / "model.pt",
    )
    config.save(_FINAL_DIR / "carina_config.json")
    tokenizer.save(str(_FINAL_DIR / "tokenizer"))
    log(f"Final checkpoint saved to {_FINAL_DIR}")


# ── Training step ─────────────────────────────────────────────────────────────

def _step_batch(
    model,
    batch,
    device,
    autocast_ctx,
    scaler,
    grad_scale: float,
) -> float:
    input_ids, labels = batch
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    with autocast_ctx:
        out = model(input_ids=input_ids, labels=labels)
        loss = out["loss"]
        if loss is None:
            return 0.0
        scaled = loss / grad_scale
    if scaler is not None:
        scaler.scale(scaled).backward()
    else:
        scaled.backward()
    return float(loss.item())


def _maybe_optimizer_step(
    accum_count: int,
    accumulation_steps: int,
    scaler,
    optimizer,
    scheduler,
    grad_clip: float,
    model,
    is_last_batch: bool,
) -> bool:
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


# ── Epoch loop ────────────────────────────────────────────────────────────────

def _run_epoch(
    epoch: int,
    args: SFTConfig,
    config: CarinaConfig,
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
    controller: TrainingController,
    step_global_start: int,
) -> int:
    if torch.distributed.is_initialized():
        train_sampler.set_epoch(epoch)
    skip = start_step if (epoch == 0 and start_step > 0) else 0
    batch_sampler = SkipBatchSampler(train_sampler, args.batch_size, skip)
    loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    accum_count = 0
    loss_accum = 0.0
    step_global = step_global_start
    grad_scale = float(args.accumulation_steps)
    epoch_start = time.time()
    total_micro = len(loader) + skip
    log(
        f"[sft] epoch={epoch + 1}/{args.epochs} | "
        f"total_micro={total_micro} | skip={skip} | accum={args.accumulation_steps}"
    )

    for step, batch in enumerate(loader, start=1):
        controller.poll_commands()
        if controller.stop_now:
            log("[sft] si received — saving emergency checkpoint and exiting")
            _save_checkpoint(
                model, optimizer, scheduler, step_global,
                epoch, config, None, wandb_run, controller,
            )
            return step_global

        loss_val = _step_batch(
            model, batch, device, autocast_ctx, scaler, grad_scale
        )
        loss_accum += loss_val
        accum_count += 1
        stepped = _maybe_optimizer_step(
            accum_count, args.accumulation_steps,
            scaler, optimizer, scheduler, args.grad_clip,
            model, step == len(loader),
        )
        if stepped:
            accum_count = 0
            step_global += 1
            avg_loss = loss_accum / max(1, args.accumulation_steps)
            loss_accum = 0.0
            cur_lr = optimizer.param_groups[0]["lr"]
            elapsed = (time.time() - epoch_start) / 60.0
            if step_global % args.log_interval == 0:
                spent = elapsed * (total_micro - step) / max(1, step)
                log(
                    f"[sft] epoch={epoch + 1} step={step}/{total_micro} "
                    f"global={step_global} loss={avg_loss:.4f} "
                    f"lr={cur_lr:.2e} eta={spent:.1f}m"
                )
                log_metrics(wandb_run, {"train/loss": avg_loss, "train/lr": cur_lr}, step=step_global)
            if step_global % args.save_interval == 0:
                _save_checkpoint(
                    model, optimizer, scheduler, step_global,
                    epoch, config, None, wandb_run, controller,
                )
                log(f"[sft] checkpoint saved at step {step_global}")

    return step_global


# ── Main ──────────────────────────────────────────────────────────────────────

def train(
    args: SFTConfig,
    fresh: bool = False,
    from_pretrain: Path | None = None,
) -> dict:
    init_distributed_mode()
    device = torch.device(args.resolve_device())
    setup_seed(args.seed)

    # ── Logging ────────────────────────────────────────────────────────────
    run_name = args.run_name or "carina-sft"
    setup_logging(args.log_dir, run_name, level=20)
    log(f"════════════════════════════════════════════════════════")
    log(f"[{datetime.now():%H:%M:%S}] ★ SFT STARTING")
    log(f"  model   : {args.num_hidden_layers}L × {args.hidden_size} hidden, "
        f"{args.num_attention_heads} heads / {args.num_key_value_heads} KV heads")
    log(f"  batch   : {args.batch_size} × {args.accumulation_steps} = "
        f"{args.batch_size * args.accumulation_steps} effective")
    log(f"  seq_len : {args.max_seq_len}")
    log(f"  lr      : {args.learning_rate:.0e} (warmup={args.warmup_ratio:.0%})")
    log(f"  epochs  : {args.epochs}")
    log(f"  device  : {device}")
    log(f"  dtype   : {args.dtype}")
    log(f"  resume  : {'fresh' if fresh else 'auto-detect'}")
    log(f"════════════════════════════════════════════════════════")

    # ── Tokenizer + config ────────────────────────────────────────────────
    tokenizer, config = build_tokenizer_and_config(args)
    model = CarinaForCausalLM(config).to(device)
    log(f"[sft] model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ── Load from pretrain (fresh start) ──────────────────────────────────
    resume_dir, start_epoch, start_step = None, 0, 0
    resume_from_str = ""

    if not fresh:
        resume_dir, start_epoch, start_step = _auto_resume()
        if resume_dir:
            log(f"[sft] resuming from SFT checkpoint {resume_dir} "
                f"(epoch={start_epoch}, step={start_step})")
            ckpt = CarinaCheckpoint.load(resume_dir, model=model, map_location=device)
            resume_from_str = f"epoch {start_epoch}, step {start_step}"
            # Note: optimizer/scheduler state loaded after we build them below
        else:
            log("[sft] no SFT checkpoint found — checking for pretrain final")
            fresh = True

    if fresh:
        pretrain_ckpt = _find_pretrain_ckpt(from_pretrain)
        if pretrain_ckpt is None:
            raise FileNotFoundError(
                "No pretrain checkpoint found at checkpoints/carina_pretrain/final.\n"
                "Run pretrain first: python scripts/pretrain.py"
            )
        log(f"[sft] loading pretrain weights from {pretrain_ckpt}")
        _load_model_weights(model, pretrain_ckpt, device)
        resume_from_str = f"pretrain @ {pretrain_ckpt}"

    # ── Dataset ───────────────────────────────────────────────────────────
    if args.data_path.is_file():
        train_ds = load_jsonl_dataset(
            args.data_path,
            tokenizer,
            max_length=args.max_seq_len,
            system=args.system_prompt,
        )
    else:
        log(f"[sft] WARNING: {args.data_path} not found — using dummy data")
        train_ds = DummySFTDataset(tokenizer, num_samples=32, max_length=args.max_seq_len)
    log(f"[sft] dataset: {len(train_ds)} records")

    # Split val
    n_val = max(1, int(len(train_ds) * args.val_split))
    indices = list(range(len(train_ds)))
    val_ds = Subset(train_ds, indices[:n_val])
    train_ds = Subset(train_ds, indices[n_val:])
    log(f"[sft] train={len(train_ds)} val={len(val_ds)}")

    # ── Optimizer / scheduler ─────────────────────────────────────────────
    import math
    optimizer = build_optimizer(model, lr=args.learning_rate, weight_decay=args.weight_decay)
    total_micro = max(1, math.ceil(len(train_ds) / args.batch_size))
    total_steps = max(1, total_micro // max(1, args.accumulation_steps) * args.epochs)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    scheduler = build_scheduler(
        optimizer, warmup_steps, total_steps, min_lr_ratio=args.min_lr_ratio
    )
    autocast_ctx, scaler = build_amp(args.resolve_dtype(), device.type)

    # Restore optimizer/scheduler state if resuming SFT
    if resume_dir and not fresh:
        ckpt = CarinaCheckpoint.load(resume_dir, map_location=device)
        if ckpt.optimizer_state is not None:
            optimizer.load_state_dict(ckpt.optimizer_state)
        if ckpt.scheduler_state is not None:
            scheduler.load_state_dict(ckpt.scheduler_state)

    wandb_run = setup_wandb(
        project=args.wandb_project,
        name=args.wandb_name or args.run_name,
        config={"args": args.as_dict()},
        enabled=args.use_wandb,
    )

    # ── Sampler ────────────────────────────────────────────────────────────
    if torch.distributed.is_initialized():
        from torch.utils.data import DistributedSampler
        train_sampler = DistributedSampler(train_ds)
    else:
        train_sampler = torch.randperm(len(train_ds)).tolist()

    # ── Training loop ─────────────────────────────────────────────────────
    controller = TrainingController()
    install_signal_handler(controller)
    fifo_path = start(controller)
    if fifo_path:
        log(f"[sft] command FIFO: {fifo_path}")
        log(f"[sft] echo sane > {fifo_path}   # stop at next epoch")
        log(f"[sft] echo si   > {fifo_path}   # stop immediately")

    step_global = start_step
    controller.update_resume(str(_CHECKPOINT_FILE), resume_from_str)

    # Early stopping state
    best_val_loss = float("inf")
    no_improve = 0

    try:
        for epoch in range(start_epoch, args.epochs):
            step_global = _run_epoch(
                epoch, args, config, model, train_ds, train_sampler,
                optimizer, scheduler, autocast_ctx, scaler, device,
                start_step if epoch == start_epoch else 0,
                wandb_run, controller, step_global,
            )

            # ── Validation ──────────────────────────────────────────────
            val_loader = DataLoader(
                val_ds,
                batch_size=args.batch_size,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            val_loss = evaluate(
                model, val_loader, device, autocast_ctx, max_batches=50
            )
            log(
                f"[sft] ★ epoch={epoch + 1}/{args.epochs} | "
                f"val_loss={val_loss:.4f} | global_step={step_global}"
            )
            log_metrics(wandb_run, {"val/loss": val_loss}, step=step_global)

            # ── Checkpoint ──────────────────────────────────────────────
            _save_checkpoint(
                model, optimizer, scheduler, step_global,
                epoch, config, tokenizer, wandb_run, controller,
            )

            # ── Early stopping ───────────────────────────────────────────
            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                no_improve = 0
            else:
                no_improve += 1
                log(f"[sft] no improvement ×{no_improve}/{args.patience}")

            if no_improve >= args.patience:
                log(f"[sft] early stopping triggered (no improvement for {no_improve} epochs)")
                break

            # ── Interrupt check ─────────────────────────────────────────
            controller.poll_commands()
            if controller.stop_at_epoch:
                log("[sft] sane received — stopping at epoch boundary")
                break
            if controller.stop_now:
                log("[sft] si received — stopping immediately")
                break

    finally:
        stop(controller)
        log(f"[sft] commands received: {controller.command_log}")

    # ── Save final ─────────────────────────────────────────────────────────
    if is_main_process():
        _save_final(
            model, optimizer, scheduler, step_global,
            args.epochs, config, tokenizer, wandb_run,
        )
    finish(wandb_run)
    shutdown_logging()
    return {
        "stage": "sft",
        "final_step": step_global,
        "best_val_loss": best_val_loss,
        "epochs_run": args.epochs,
        "save_dir": str(_FINAL_DIR),
        "stopped_at_epoch": (controller.stop_at_epoch or controller.stop_now),
        "resume_from": controller.resume_from,
        "commands": controller.command_log,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Carina SFT (Python-file config).")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from pretrain final checkpoint (ignore SFT checkpoint).",
    )
    parser.add_argument(
        "--from-pretrain",
        type=Path,
        default=None,
        help="Path to pretrain checkpoint directory (default: checkpoints/carina_pretrain/final).",
    )
    args = parser.parse_args()

    cfg: SFTConfig = SFTConfig()
    summary = train(cfg, fresh=args.fresh, from_pretrain=args.from_pretrain)
    log(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
