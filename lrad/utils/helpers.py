"""Utility functions: device selection, seeding, logging."""

import torch
import random
import numpy as np
import logging
from pathlib import Path


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


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """Configure project-wide logger."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lrad")
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(
            "[%(asctime)s %(levelname)s] %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(Path(log_dir) / "lrad.log")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ))
        logger.addHandler(fh)

    return logger


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
