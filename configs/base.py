"""Base configuration shared by every Carina training stage.

Concrete stage configs (``SFTConfig``, ``PretrainConfig``) inherit from
this class so common fields live in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BaseConfig:
    """Fields common to every Carina training stage.

    Attributes:
        seed: RNG seed; applied to Python, NumPy, and PyTorch.
        output_dir: Where logs and checkpoints are written.
        log_dir: Where per-run log files are appended.
        run_name: Optional human-readable name; auto-derived if empty.
        use_wandb: Whether to initialise a Weights & Biases run.
        wandb_project: Project name for ``wandb.init``.
        wandb_name: Optional run name for ``wandb.init``.
        wandb_run_id: Run id to resume (``None`` for new runs).
        num_workers: DataLoader worker count (``0`` → main process).
        device: ``"cuda"`` or ``"cpu"``; ``auto`` chooses at runtime.
        dtype: Compute dtype (``"bfloat16"``, ``"float16"``, ``"float32"``).
        log_interval: Steps between console metric prints.
        save_interval: Steps between checkpoint writes.
        notes: Free-form notes; surfaced in run summaries.
    """

    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("checkpoints/carina"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    run_name: str = "carina"
    use_wandb: bool = False
    wandb_project: str = "carina"
    wandb_name: str = ""
    wandb_run_id: str | None = None
    num_workers: int = 0
    device: str = "auto"
    dtype: str = "bfloat16"
    log_interval: int = 20
    save_interval: int = 500
    notes: str = ""

    def resolve_device(self) -> str:
        """Pick a concrete device string.

        Returns ``"cuda"`` when ``device == "auto"`` and a CUDA GPU is
        available, otherwise ``"cpu"``.
        """
        if self.device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    def resolve_dtype(self) -> str:
        """Return ``float32`` on CPU, the configured dtype on GPU."""
        return "float32" if self.resolve_device() == "cpu" else self.dtype

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow dict copy with ``Path`` fields stringified."""
        from dataclasses import asdict

        out = asdict(self)
        for k, v in list(out.items()):
            if isinstance(v, Path):
                out[k] = str(v)
        return out


__all__ = ["BaseConfig"]
