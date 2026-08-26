"""Shared model construction, optimizer partitioning and epoch execution."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch import nn

from src.models.enhanced_two_tower import EnhancedTwoTower

from .stage6_data import move_tensor_tree


def build_model(variant: str, config: Mapping[str, object], num_item_tokens: int, side_vocab_sizes: Sequence[int]) -> EnhancedTwoTower:
    model = config["model"]; user = config["user_tower"]; item = config["item_tower"]
    return EnhancedTwoTower(
        variant, num_item_tokens, side_vocab_sizes, int(model["embedding_dim"]),
        int(user["attention"]["num_heads"]), float(user["attention"]["dropout"]),
        int(item["side_embedding_dim"]), int(item["mm_input_dim"]),
        int(user["time_feature_dim"]), int(user["time_hidden_dim"]),
    )


def partition_sparse_dense_parameters(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    sparse_ids = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding) and bool(module.sparse):
            sparse_ids.add(id(module.weight))
    sparse, dense = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (sparse if id(parameter) in sparse_ids else dense).append(parameter)
    if set(map(id, sparse)) & set(map(id, dense)):
        raise AssertionError("sparse/dense optimizer parameter sets overlap")
    return sparse, dense


def make_optimizers(model: nn.Module, config: Mapping[str, object]):
    section = config["training"]
    sparse, dense = partition_sparse_dense_parameters(model)
    sparse_optimizer = torch.optim.SparseAdam(sparse, lr=float(section["sparse_learning_rate"])) if sparse else None
    dense_optimizer = torch.optim.AdamW(dense, lr=float(section["dense_learning_rate"]), weight_decay=float(section.get("weight_decay", 0.0))) if dense else None
    return sparse_optimizer, dense_optimizer


def run_epoch(model, loader, device, optimizers=()) -> tuple[float, int]:
    training = bool(optimizers)
    model.train(training)
    total, count = 0.0, 0
    for batch in loader:
        if not batch.get("rows"):
            continue
        user = move_tensor_tree(batch["user"], device)
        positive = move_tensor_tree(batch["positive"], device)
        negative = move_tensor_tree(batch["negative"], device)
        for optimizer in optimizers:
            if optimizer is not None: optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            loss = model.sampled_softmax_loss(user, positive, negative)
        if training:
            loss.backward()
            for optimizer in optimizers:
                if optimizer is not None: optimizer.step()
        size = len(batch["rows"]); total += float(loss.detach()) * size; count += size
    return (total / count if count else 0.0), count


def atomic_save_checkpoint(path: Path, model, sparse_optimizer, dense_optimizer, metadata: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = {"model": model.state_dict(), **dict(metadata)}
    if sparse_optimizer is not None: payload["sparse_optimizer"] = sparse_optimizer.state_dict()
    if dense_optimizer is not None: payload["dense_optimizer"] = dense_optimizer.state_dict()
    try:
        torch.save(payload, temporary); os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_model_weights(path: Path, model: nn.Module, strict: bool = True) -> dict[str, object]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(checkpoint["model"], strict=strict)
    return {key: value for key, value in checkpoint.items() if key not in {"model", "sparse_optimizer", "dense_optimizer"}}

