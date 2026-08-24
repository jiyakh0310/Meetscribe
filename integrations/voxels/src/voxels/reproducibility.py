"""Utilities for reproducible local runs."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedReport:
    """Summary of libraries touched by seed setup."""

    seed: int
    python_seeded: bool
    numpy_seeded: bool
    torch_seeded: bool
    torch_available: bool


def set_seed(seed: int, deterministic_torch: bool = True) -> SeedReport:
    """Seed Python, NumPy, and PyTorch when PyTorch is installed."""
    if seed < 0:
        raise ValueError("Seed must be non-negative.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_available = False
    torch_seeded = False
    try:
        import torch
    except ImportError:
        pass
    else:
        torch_available = True
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
        torch_seeded = True

    return SeedReport(
        seed=seed,
        python_seeded=True,
        numpy_seeded=True,
        torch_seeded=torch_seeded,
        torch_available=torch_available,
    )

