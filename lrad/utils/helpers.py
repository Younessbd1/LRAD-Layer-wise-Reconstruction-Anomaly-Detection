"""Utility functions: device selection, seeding, logging."""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def get_device() -> torch.device:
    """Select best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class _FlushingStreamHandler(logging.StreamHandler):
    """Stream handler that flushes after every record so OAR's `tail -f`
    on the captured stdout/stderr shows progress live instead of in
    multi-MB bursts when Python's stdio buffer fills."""

    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


def setup_logging(
    log_dir: str = "logs",
    level: int = logging.INFO,
    run_tag: Optional[str] = None,
) -> logging.Logger:
    """Configure project-wide logger.

    Writes to both stdout and a file under ``log_dir``. The file name
    includes a timestamp (and optional ``run_tag``) so successive OAR
    submissions don't clobber each other's logs.

    Also installs a ``sys.excepthook`` that routes uncaught exceptions to
    the same logger — so a crash on the GPU node still leaves a trace in
    the log file even when stderr capture is unreliable.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lrad")
    logger.setLevel(level)

    if not logger.handlers:
        ch = _FlushingStreamHandler(stream=sys.stdout)
        ch.setFormatter(logging.Formatter(
            "[%(asctime)s %(levelname)s] %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(ch)

        suffix = f"_{run_tag}" if run_tag else ""
        log_path = Path(log_dir) / f"lrad{suffix}.log"
        fh = logging.FileHandler(log_path, mode="w")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        logger.addHandler(fh)

        def _log_uncaught(exc_type, exc_value, exc_tb):
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_tb)
                return
            logger.critical("Uncaught exception",
                            exc_info=(exc_type, exc_value, exc_tb))
            for h in logger.handlers:
                try:
                    h.flush()
                except Exception:
                    pass
        sys.excepthook = _log_uncaught

    return logger


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
