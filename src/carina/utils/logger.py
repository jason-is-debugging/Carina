"""Minimal stdlib logger: console + file, four levels."""
from __future__ import annotations

import logging
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

_LEVELS: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def make_logger(
    name: str,
    log_file: str | None = None,
    level: LogLevel = "INFO",
) -> logging.Logger:
    """Create a logger writing to console and optionally a file.

    Args:
        name: Logger channel name (usually module name, non-empty).
        log_file: Optional file path for persistent logs.
        level: One of DEBUG/INFO/WARNING/ERROR.

    Returns:
        A configured ``logging.Logger`` instance.

    Raises:
        ValueError: If ``level`` is not a known level name.
    """
    assert isinstance(name, str) and len(name) > 0, (
        f"[logger] expected non-empty str name, got {name!r}")
    assert log_file is None or isinstance(log_file, str), (
        f"[logger] expected str|None log_file, got {type(log_file)}")
    if level not in _LEVELS:
        raise ValueError(f"[logger] unknown level, expected one of "
                         f"{list(_LEVELS)}, got {level!r}")
    logger: logging.Logger = logging.getLogger(name)
    logger.setLevel(_LEVELS[level])
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt: logging.Formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    console: logging.StreamHandler[logging.LogRecord] = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file is not None:
        file_handler: logging.FileHandler = logging.FileHandler(log_file)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger
