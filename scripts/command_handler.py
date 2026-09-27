"""Interactive command handler for training scripts.

Provides two ways to interrupt training from another terminal or stdin:

* **FIFO mode** (default, recommended): creates a named pipe at
  ``/tmp/carina_train_cmd_<pid>.fifo``.  Writing ``sane`` or ``si`` to
  that file from another terminal triggers the corresponding interrupt.
* **stdin mode** (fallback): if stdin is a TTY the module spawns a
  background reader thread that pulls commands from stdin itself.

Commands
--------
sane  — stop at the next epoch boundary (graceful)
si    — stop immediately (lossy — current step may not be checkpointed)
"""

from __future__ import annotations

import os
import select
import signal
import sys
import threading
from queue import Empty, Queue
from typing import Optional

# ── Public API ─────────────────────────────────────────────────────────────────


class TrainingController:
    """Thread-safe interrupt controller for long-running training loops.

    Attributes
    ----------
    stop_at_epoch : bool
        Set to ``True`` by the ``sane`` command.  Training code should
        check this at the end of each epoch and return if it is ``True``.
    stop_now : bool
        Set to ``True`` by the ``si`` command.  Training code should
        check this at the end of every step and abort immediately if
        it is ``True``.
    resume_checkpoint_path : Optional[str]
        Path to the latest checkpoint.  Updated after every save so that
        a crashed / interrupted run can be resumed without searching.
    resume_from : Optional[str]
        Human-readable string describing the resume source, e.g.
        ``"epoch 3, step 1234"``.
    command_log : list[str]
        Append-only list of received commands with timestamps (seconds
        since controller creation).
    """

    # Commands recognized by the handler
    CMD_STOP_AT_EPOCH = "sane"
    CMD_STOP_IMMEDIATE = "si"

    def __init__(self) -> None:
        self.stop_at_epoch = False
        self.stop_now = False
        self.resume_checkpoint_path: Optional[str] = None
        self.resume_from: Optional[str] = None
        self.command_log: list[str] = []
        self._lock = threading.Lock()
        self._queue: Queue[str] = Queue()
        self._fifo_path: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ── public helpers ────────────────────────────────────────────────────

    def update_resume(self, path: str, resume_from: str) -> None:
        """Call after loading a checkpoint so resume info is accurate."""
        with self._lock:
            self.resume_checkpoint_path = path
            self.resume_from = resume_from

    def poll_commands(self) -> None:
        """Call this at the end of every step.

        Pumps the command queue and sets the corresponding flags.
        ``stop_now`` takes precedence over ``stop_at_epoch``.
        """
        drained: list[str] = []
        while True:
            try:
                drained.append(self._queue.get_nowait())
            except Empty:
                break
        for cmd in drained:
            with self._lock:
                ts = f"{len(self.command_log):03d}"
                if cmd == self.CMD_STOP_AT_EPOCH:
                    self.stop_at_epoch = True
                    self.command_log.append(f"[{ts}] sane (stop at next epoch)")
                elif cmd == self.CMD_STOP_IMMEDIATE:
                    self.stop_now = True
                    self.command_log.append(f"[{ts}] si (stop immediately)")
                else:
                    self.command_log.append(f"[{ts}] unknown: {cmd!r}")

    def is_interrupted(self) -> bool:
        """Returns ``True`` if either interrupt flag is active."""
        with self._lock:
            return self.stop_at_epoch or self.stop_now

    def __repr__(self) -> str:
        return (
            f"TrainingController(stop_at_epoch={self.stop_at_epoch}, "
            f"stop_now={self.stop_now}, "
            f"resume_from={self.resume_from!r})"
        )


# ── FIFO mode ───────────────────────────────────────────────────────────────────

def _make_fifo(path: str) -> None:
    """Create a named pipe, removing an existing file first."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    os.mkfifo(path, 0o600)


def _fifo_reader(controller: TrainingController, fifo_path: str) -> None:
    """Background thread: drain the FIFO and push commands onto the queue."""
    fd = None
    try:
        fd = os.open(fifo_path, os.O_RDONLY | os.O_NONBLOCK)
        while controller._running:
            # Use select() so we can wake up even when no data is available
            ready, _, _ = select.select([fd], [], [], 0.5)
            if not ready:
                continue
            try:
                data = os.read(fd, 4096).decode("utf-8", errors="ignore")
            except OSError:
                # Non-blocking read had nothing
                continue
            for line in data.splitlines():
                line = line.strip()
                if line:
                    controller._queue.put(line)
    except Exception:
        pass
    finally:
        if fd is not None:
            os.close(fd)


def start_fifo_mode(controller: TrainingController) -> str:
    """Create the FIFO and launch the background reader thread.

    Returns the FIFO path (printed to the user so they know where to write).
    """
    pid = os.getpid()
    fifo_path = f"/tmp/carina_train_cmd_{pid}.fifo"
    _make_fifo(fifo_path)
    controller._fifo_path = fifo_path
    controller._running = True
    t = threading.Thread(target=_fifo_reader, args=(controller, fifo_path), daemon=True)
    t.start()
    controller._thread = t
    return fifo_path


def stop_fifo_mode(controller: TrainingController) -> None:
    """Stop the background reader and remove the FIFO."""
    controller._running = False
    if controller._thread is not None:
        controller._thread.join(timeout=2.0)
        controller._thread = None
    if controller._fifo_path:
        try:
            os.unlink(controller._fifo_path)
        except FileNotFoundError:
            pass
        controller._fifo_path = None


# ── stdin mode (TTY fallback) ───────────────────────────────────────────────────

def _stdin_reader(controller: TrainingController) -> None:
    """Background thread: read commands from stdin (for use inside a TTY)."""
    try:
        while controller._running:
            if select.select([sys.stdin], [], [], 0.5)[0]:
                try:
                    line = sys.stdin.readline()
                except (OSError, IOError):
                    break
                if not line:
                    break
                line = line.strip()
                if line:
                    controller._queue.put(line)
    except Exception:
        pass


def start_stdin_mode(controller: TrainingController) -> None:
    """Launch the stdin reader thread (call only when stdin is a TTY)."""
    controller._running = True
    t = threading.Thread(target=_stdin_reader, args=(controller,), daemon=True)
    t.start()
    controller._thread = t


# ── Auto-detection entry point ─────────────────────────────────────────────────

def start(controller: TrainingController) -> Optional[str]:
    """Start the appropriate reader mode and return the FIFO path (or None).

    If stdin is a TTY → stdin mode (no FIFO, commands from the same terminal).
    Otherwise → FIFO mode.
    """
    if sys.stdin.isatty():
        start_stdin_mode(controller)
        return None
    return start_fifo_mode(controller)


def stop(controller: TrainingController) -> None:
    """Stop all background readers."""
    controller._running = False
    if controller._thread is not None:
        controller._thread.join(timeout=2.0)
        controller._thread = None
    stop_fifo_mode(controller)


# ── Signal-based immediate stop (Unix) ────────────────────────────────────────

_SI_CONTROLLER: Optional[TrainingController] = None


def _handle_sigint(signum: int, frame) -> None:
    """Convert SIGINT (Ctrl+C) and SIGTERM into a ``si`` command."""
    if _SI_CONTROLLER is not None:
        _SI_CONTROLLER.stop_now = True


def install_signal_handler(controller: TrainingController) -> None:
    """Install SIGINT/SIGTERM handlers that trigger ``si`` behaviour."""
    global _SI_CONTROLLER
    _SI_CONTROLLER = controller
    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)


__all__ = [
    "TrainingController",
    "start",
    "stop",
    "start_fifo_mode",
    "stop_fifo_mode",
    "start_stdin_mode",
    "install_signal_handler",
]
