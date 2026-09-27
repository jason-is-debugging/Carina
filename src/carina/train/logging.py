"""Logging helpers — wandb if available, console / file otherwise.

The module exposes two layers:

* :func:`setup_logging` configures a Python ``logging`` handler that
  writes to both stdout and a per-run file under ``log_dir``.
* :func:`setup_wandb` / :func:`log_metrics` / :func:`finish` are thin
  wrappers that degrade gracefully when wandb is missing or offline.

Status messages emitted through :func:`log` (or any standard
``logging`` call below the root logger) reach both the per-run
``train.log`` file *and* the console as soon as :func:`setup_logging`
has been called. Always call :func:`shutdown_logging` before the
trainer exits so the final lines are flushed to disk.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils import format_metrics, is_main_process

try:  # pragma: no cover - wandb is optional at test time
    import wandb as _wandb
except ImportError:  # pragma: no cover
    _wandb = None

_LOGGER = logging.getLogger("carina.logging")


class _FlushFileHandler(logging.FileHandler):
    """``FileHandler`` that flushes after every record.

    Standard ``FileHandler`` only flushes when the buffer is full or
    on ``close()``. That means a crash can lose the last few KB of
    training logs — exactly the lines you most want to keep. Flushing
    after each ``emit`` is cheap on local disks and makes ``tail -f``
    of ``train.log`` reflect the live run.
    """

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401 - std API
        try:
            super().emit(record)
            self.flush()
        except RecursionError:  # pragma: no cover - defensive
            raise
        except Exception:  # pragma: no cover - defensive
            self.handleError(record)


def setup_logging(
    log_dir: str | Path,
    run_name: str,
    level: int = logging.INFO,
    mirror_stdout: bool = True,
) -> Path:
    """Configure console + file logging.

    Args:
        log_dir: Root directory; per-run subdirs are created inside.
        run_name: Human-readable run name (sanitised).
        level: Logging level for both handlers.
        mirror_stdout: When ``False`` only the file handler is added.

    Returns:
        The path to the per-run log file.
    """
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_name)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_dir) / f"{safe_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "train.log"
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = _FlushFileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if mirror_stdout:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(fmt)
        root.addHandler(ch)
    return log_file


def shutdown_logging() -> None:
    """Flush + close every handler attached to the root logger.

    Trainers should call this once before returning so the final lines
    land in ``train.log`` even if the interpreter exits without going
    through a normal ``atexit`` handler.
    """
    logging.shutdown()


def log(msg: str, *, level: int = logging.INFO) -> None:
    """Emit ``msg`` through the ``carina.logging`` logger (rank-0 only).

    This is the single entry point training code should use for
    human-readable status output: once :func:`setup_logging` has been
    called, the message lands in both the per-run ``train.log`` and
    stdout.
    """
    if is_main_process():
        _LOGGER.log(level, msg)


def setup_wandb(
    project: str,
    name: str | None = None,
    config: dict[str, Any] | None = None,
    run_id: str | None = None,
    enabled: bool = True,
):
    """Initialise a wandb run, falling back to a no-op offline mode.

    Returns:
        The wandb ``Run`` object on success, otherwise a small stub that
        mimics the ``log`` / ``finish`` interface.
    """
    if not enabled or os.environ.get("WANDB_DISABLED") == "1":
        return _NoOpRun()
    if _wandb is None:  # pragma: no cover
        return _NoOpRun()
    mode = os.environ.get("WANDB_MODE")
    if mode not in {"online", "offline", "disabled"}:
        os.environ["WANDB_MODE"] = "offline"
    try:
        return _wandb.init(
            project=project,
            name=name,
            config=config or {},
            id=run_id,
            resume="allow" if run_id else None,
        )
    except (OSError, ValueError, RuntimeError):  # pragma: no cover
        _LOGGER.debug("wandb init failed; using no-op run", exc_info=True)
        return _NoOpRun()


def log_metrics(
    run: Any,
    metrics: dict[str, Any],
    step: int | None = None,
    extra_keys: Sequence[str] | None = None,
) -> None:
    """Forward ``metrics`` to wandb (or print them on rank 0)."""
    if run is None:
        return
    if hasattr(run, "log"):
        try:
            run.log(metrics, step=step)
            return
        except (OSError, ValueError, RuntimeError):  # pragma: no cover
            _LOGGER.debug("wandb log failed; falling back to console", exc_info=True)
    if is_main_process():
        msg = format_metrics(metrics)
        if extra_keys:
            for key in extra_keys:
                msg += f" | {key}={metrics.get(key, 'n/a')}"
        _LOGGER.info(f"[step {step}] {msg}")


def finish(run: Any) -> None:
    """Tear down a wandb run if it is real."""
    if run is None:
        return
    if hasattr(run, "finish"):
        try:
            run.finish()
        except (OSError, ValueError, RuntimeError):  # pragma: no cover
            _LOGGER.debug("wandb finish failed", exc_info=True)


class _NoOpRun:
    """Drop-in replacement used when wandb is unavailable."""

    def log(self, *args: object, **kwargs: object) -> None:
        return None

    def finish(self) -> None:
        return None


__all__ = ["finish", "log", "log_metrics", "setup_logging", "setup_wandb", "shutdown_logging"]
