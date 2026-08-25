"""Reproducible PyTorch checkpoint helpers for Stage 5."""

from __future__ import annotations

import gc
import os
import random
import uuid
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
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: Mapping[str, Any],
    protocols: Mapping[str, Any],
    validation_loss: float,
    training_state: Mapping[str, Any] | None = None,
) -> None:
    """Atomically replace ``path`` only after a complete checkpoint is written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload: dict[str, Any] = {
            "model": model.state_dict(),
            "epoch": int(epoch),
            "config": dict(config),
            "protocols": dict(protocols),
            "validation_loss": float(validation_loss),
            "random_state": random_state(),
            "training_state": dict(training_state or {}),
        }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    try:
        torch.save(payload, temporary_path)
        # torch.save(path) closes the completed temp file before the atomic replacement.
        # Avoid fsync here: it serializes multi-GB writes with training and is unnecessary
        # for protection against process/OOM failure (the old path remains until replace).
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
        del payload
        gc.collect()


def load_checkpoint(path: Path, model=None, optimizer=None, map_location="cpu") -> dict[str, Any]:
    # CPU mmap prevents a multi-GB resume checkpoint from being eagerly copied into RAM.
    checkpoint = torch.load(
        path, map_location=map_location, weights_only=False,
        mmap=str(map_location) == "cpu",
    )
    if model is not None:
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        if "optimizer" not in checkpoint:
            raise ValueError(f"checkpoint does not contain optimizer state: {path.resolve()}")
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def load_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location="cpu",
) -> dict[str, Any]:
    """Restore training state and release the large deserialized CPU tensor dictionary."""

    checkpoint = load_checkpoint(path, model, optimizer, map_location=map_location)
    metadata = {
        "epoch": int(checkpoint["epoch"]),
        "config": dict(checkpoint.get("config") or {}),
        "protocols": dict(checkpoint.get("protocols") or {}),
        "validation_loss": float(checkpoint["validation_loss"]),
        "random_state": checkpoint["random_state"],
        "training_state": dict(checkpoint.get("training_state") or {}),
    }
    del checkpoint
    gc.collect()
    return metadata


def load_model_checkpoint(
    path: Path,
    model: torch.nn.Module,
    map_location="cpu",
) -> dict[str, Any]:
    """Restore model weights and discard optimizer/model tensors deserialized on the CPU."""

    checkpoint = load_checkpoint(path, model=model, map_location=map_location)
    metadata = {
        "epoch": int(checkpoint["epoch"]),
        "config": dict(checkpoint.get("config") or {}),
        "protocols": dict(checkpoint.get("protocols") or {}),
        "validation_loss": float(checkpoint["validation_loss"]),
        "training_state": dict(checkpoint.get("training_state") or {}),
    }
    del checkpoint
    gc.collect()
    return metadata
