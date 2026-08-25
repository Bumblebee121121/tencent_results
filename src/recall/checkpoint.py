"""Reproducible PyTorch checkpoint helpers for Stage 5."""

from __future__ import annotations

import gc
import os
import random
import uuid
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


class CpuCheckpointBuffer:
    """Reusable CPU tensors that prevent CUDA checkpoint staging from growing each save."""

    def __init__(self) -> None:
        self._tensor_buffers: dict[tuple[str, ...], torch.Tensor] = {}

    def _copy_tree(self, value: Any, path: tuple[str, ...]) -> Any:
        if isinstance(value, torch.Tensor):
            source = value.detach()
            target = self._tensor_buffers.get(path)
            if (
                target is None
                or target.shape != source.shape
                or target.dtype != source.dtype
            ):
                target = torch.empty_like(source, device="cpu", pin_memory=False)
                self._tensor_buffers[path] = target
            target.copy_(source, non_blocking=False)
            return target
        if isinstance(value, OrderedDict):
            copied = OrderedDict(
                (key, self._copy_tree(item, path + (str(key),)))
                for key, item in value.items()
            )
            if hasattr(value, "_metadata"):
                copied._metadata = deepcopy(value._metadata)
            return copied
        if isinstance(value, dict):
            return {
                key: self._copy_tree(item, path + (str(key),))
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._copy_tree(item, path + (str(index),)) for index, item in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(self._copy_tree(item, path + (str(index),)) for index, item in enumerate(value))
        return deepcopy(value)

    def model_state_dict(self, model: torch.nn.Module) -> Mapping[str, Any]:
        return self._copy_tree(model.state_dict(), ("model",))

    def optimizer_state_dict(self, optimizer: torch.optim.Optimizer) -> Mapping[str, Any]:
        return self._copy_tree(optimizer.state_dict(), ("optimizer",))

    @property
    def allocated_bytes(self) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in self._tensor_buffers.values())

    @property
    def tensor_count(self) -> int:
        return len(self._tensor_buffers)


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
    cpu_buffer: CpuCheckpointBuffer | None = None,
) -> None:
    """Atomically replace ``path`` only after a complete checkpoint is written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload: dict[str, Any] = {
            "model": (
                cpu_buffer.model_state_dict(model)
                if cpu_buffer is not None
                else model.state_dict()
            ),
            "epoch": int(epoch),
            "config": dict(config),
            "protocols": dict(protocols),
            "validation_loss": float(validation_loss),
            "random_state": random_state(),
            "training_state": dict(training_state or {}),
        }
    if optimizer is not None:
        payload["optimizer"] = (
            cpu_buffer.optimizer_state_dict(optimizer)
            if cpu_buffer is not None
            else optimizer.state_dict()
        )
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
