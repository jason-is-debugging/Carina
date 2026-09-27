"""Checkpoint save / load helpers for Carina SFT.

The :class:`CarinaCheckpoint` dataclass bundles everything needed to
resume training: model + optimizer + scheduler state dicts, the global
step and epoch counters, and an optional wandb run id. Writes are
atomic (temp file → ``os.replace``) so a crash mid-save cannot leave a
truncated checkpoint on disk.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass
class CarinaCheckpoint:
    """Container for a single training checkpoint.

    Attributes:
        model_state: ``state_dict()`` from the underlying model.
        optimizer_state: Optional ``state_dict()`` from the optimiser.
        scheduler_state: Optional ``state_dict()`` from the scheduler.
        step: Global training step at save time.
        epoch: Epoch index at save time.
        wandb_run_id: wandb run id used for resume (or ``None``).
        config_json: Serialised :class:`CarinaConfig`.
        extra: Arbitrary user-supplied metadata.
    """

    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any] | None = None
    scheduler_state: dict[str, Any] | None = None
    step: int = 0
    epoch: int = 0
    wandb_run_id: str | None = None
    config_json: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def save(self, directory: str | Path) -> None:
        """Persist the checkpoint to ``directory`` atomically."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        cpu_state = {k: v.detach().cpu() for k, v in self.model_state.items()}
        payload: dict[str, Any] = {
            "model": cpu_state,
            "step": self.step,
            "epoch": self.epoch,
            "wandb_run_id": self.wandb_run_id,
            "config_json": self.config_json,
            "extra": self.extra,
        }
        if self.optimizer_state is not None:
            payload["optimizer"] = self.optimizer_state
        if self.scheduler_state is not None:
            payload["scheduler"] = self.scheduler_state
        with tempfile.NamedTemporaryFile(
            "wb", delete=False, dir=str(directory), suffix=".pt"
        ) as fh:
            tmp = fh.name
            torch.save(payload, tmp)
            os.replace(tmp, directory / "checkpoint.pt")
        meta_path = directory / "checkpoint.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "step": self.step,
                    "epoch": self.epoch,
                    "wandb_run_id": self.wandb_run_id,
                    "config_json": self.config_json,
                    "extra": self.extra,
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load(
        cls,
        directory: str | Path,
        model: nn.Module | None = None,
        map_location: str | torch.device = "cpu",
    ) -> CarinaCheckpoint:
        """Load a checkpoint and (optionally) re-inject it into ``model``."""
        directory = Path(directory)
        ckpt_path = directory / "checkpoint.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
        raw = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
        ckpt = cls(
            model_state=raw.get("model", {}),
            optimizer_state=raw.get("optimizer"),
            scheduler_state=raw.get("scheduler"),
            step=int(raw.get("step", 0)),
            epoch=int(raw.get("epoch", 0)),
            wandb_run_id=raw.get("wandb_run_id"),
            config_json=raw.get("config_json", {}),
            extra=raw.get("extra", {}),
        )
        if model is not None:
            model.load_state_dict(ckpt.model_state, strict=False)
        return ckpt


__all__ = ["CarinaCheckpoint"]
