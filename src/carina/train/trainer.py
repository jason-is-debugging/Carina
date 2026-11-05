"""Training logic for CarinaCausalModel (pretrain + SFT share it).

Single route: SGD + momentum, whole model in bf16, no AMP branches.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.utils as nn_utils
from torch import Tensor
from torch.utils.data import DataLoader

from carina.model.carina_casual_model import CarinaCausalModel


@dataclass
class TrainerOptions:
    """Hyperparams + paths for one training run."""

    lr: float = 1e-2
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    max_steps: int = 20000
    eval_interval: int = 1000
    save_dir: str = "saved/models"
    warmup_steps: int = 200
    min_lr_ratio: float = 0.05
    # fp32 master weight copy: compute still runs in bf16, but the optimizer
    # updates the fp32 copy which is then written back to the bf16 model.
    # Measured necessity: bf16 resolves only ~0.004 around 1.0, which is
    # coarser than a single update, so updates get rounded away (this is
    # exactly why the residual gains used to stay pinned at 1.0).
    fp32_master: bool = True


class Trainer:
    """Owns SGD optimizer + train/eval/fit for a causal model."""

    model: CarinaCausalModel
    options: TrainerOptions
    optimizer: torch.optim.Optimizer
    device: torch.device

    def __init__(self, model: CarinaCausalModel, options: TrainerOptions) -> None:
        self.model = model
        self.options = options
        self.master: dict[str, Tensor] = {}
        if options.fp32_master:
            # The optimizer holds the fp32 copies; the model parameters stay
            # bf16 and are only used for the forward/backward pass.
            self.master = {
                name: param.detach().float().clone().requires_grad_(True)
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            opt_params = list(self.master.values())
        else:
            opt_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            opt_params, lr=options.lr, momentum=0.9,
            weight_decay=options.weight_decay,
        )
        self.device = next(model.parameters()).device

    def train_step(self, inputs: Tensor, labels: Tensor) -> float:
        """Single SGD step, return loss value. Model is bf16 throughout."""
        self.model.train()
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        logits: Tensor = self.model(inputs)
        loss: Tensor = self.model.compute_loss(logits, labels)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.master:
            # Route the bf16 gradients into the fp32 master gradients and
            # clear the bf16-side accumulation.
            for name, param in self.model.named_parameters():
                if name in self.master:
                    self.master[name].grad = (
                        param.grad.float() if param.grad is not None else None)
                    param.grad = None
            nn_utils.clip_grad_norm_(
                list(self.master.values()), self.options.grad_clip)
        else:
            nn_utils.clip_grad_norm_(
                self.model.parameters(), self.options.grad_clip)
        self.optimizer.step()
        if self.master:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if name in self.master:
                        param.copy_(self.master[name].to(param.dtype))
                        param.grad = None
        return float(loss.item())

    def set_lr(self, step: int) -> float:
        """Set lr for step: linear warmup then cosine decay to min_lr_ratio*lr.

        Args:
            step: 1-based step index.

        Returns:
            The lr applied for this step.
        """
        opts = self.options
        warmup = min(opts.warmup_steps, max(opts.max_steps // 10, 1))
        if warmup > 0 and step <= warmup:
            lr = opts.lr * step / float(warmup)
        else:
            span = max(opts.max_steps - warmup, 1)
            progress = min((step - warmup) / span, 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            lr = opts.lr * (opts.min_lr_ratio + (1.0 - opts.min_lr_ratio) * cos)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, max_batches: int = 20) -> tuple[float, float]:
        """Mean loss + perplexity over a loader (capped at max_batches)."""
        self.model.eval()
        total: float = 0.0
        count: int = 0
        for inputs, labels in loader:
            if count >= max_batches:
                break
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            logits: Tensor = self.model(inputs)
            loss: Tensor = self.model.compute_loss(logits, labels)
            total += float(loss.item())
            count += 1
        mean: float = total / max(count, 1)
        return mean, float(torch.exp(torch.tensor(mean)).item())

    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            log_fn: Callable[[str], None] | None = None) -> dict:
        """Run max_steps, return history with losses/ppls/steps."""
        save_dir = Path(self.options.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        best: Path = save_dir / "best.pt"
        latest: Path = save_dir / "latest.pt"
        best_ppl: float = float("inf")
        losses: list[float] = []
        ppls: list[float] = []
        steps: list[int] = []
        emit: Callable[[str], None] = log_fn or (lambda _msg: None)

        data_iter = iter(train_loader)

        for step in range(1, self.options.max_steps + 1):
            try:
                inputs, labels = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                inputs, labels = next(data_iter)

            self.set_lr(step)
            loss_val: float = self.train_step(inputs, labels)
            if step % 50 == 0 or step == 1:
                emit(f"step={step} loss={loss_val:.4f} "
                     f"lr={self.optimizer.param_groups[0]['lr']:.4f}")

            if step % self.options.eval_interval == 0:
                _, ppl = self.evaluate(val_loader)
                losses.append(loss_val)
                ppls.append(ppl)
                steps.append(step)
                emit(f"step={step} loss={loss_val:.4f} val_ppl={ppl:.3f}")

                torch.save(self.model.state_dict(), latest)

                if ppl < best_ppl:
                    best_ppl = ppl
                    torch.save(self.model.state_dict(), best)

        if not steps:
            _, val_ppl = self.evaluate(val_loader)
            losses.append(loss_val)
            ppls.append(val_ppl)
            steps.append(self.options.max_steps)
            best_ppl = val_ppl
            torch.save(self.model.state_dict(), latest)
            torch.save(self.model.state_dict(), best)

        return {"steps": steps, "losses": losses, "ppls": ppls,
                "best_ppl": best_ppl, "final_train_loss": loss_val}
