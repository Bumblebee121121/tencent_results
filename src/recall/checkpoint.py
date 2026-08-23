"""Reproducible PyTorch checkpoint helpers for Stage 5."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def random_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_random_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Mapping[str, Any],
    protocols: Mapping[str, Any],
    validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "config": dict(config),
            "protocols": dict(protocols),
            "validation_loss": float(validation_loss),
            "random_state": random_state(),
        },
        path,
    )


def load_checkpoint(path: Path, model=None, optimizer=None, map_location="cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
