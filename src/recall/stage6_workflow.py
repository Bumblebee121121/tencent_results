"""Executable training workflow shared by Stage 6 variant entry points."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.features.feature_store import FeatureStore
from src.recall.data import UniformNegativeSampler
from src.recall.early_stopping import ValidationLossEarlyStopping

from .stage6_data import Stage6Collator, Stage6ItemStore, Stage6ParquetDataset, TimeNormalization
from .stage6_runtime import configured_session, guard_outputs, load_json, save_csv, save_json, seed_everything, select_device
from .stage6_training import atomic_save_checkpoint, build_model, make_optimizers, run_epoch


PARENT = {"U2": "U1", "U3": "U2", "I1": "U3", "I2": "U3", "I3": "U3", "E1": "I3"}


def _load_compatible_parent(model, path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    current = model.state_dict()
    compatible = {key: value for key, value in checkpoint["model"].items() if key in current and current[key].shape == value.shape}
    model.load_state_dict(compatible, strict=False)
    return len(compatible)


def _freeze_user_for_item_ablation(model) -> None:
    for parameter in model.user_tower.parameters():
        parameter.requires_grad = False
    model.item_embedding.weight.requires_grad = False


def _selected_parent_checkpoint(root: Path, parent: str, debug: bool) -> Path:
    selection_path = root / "manifests" / f"{parent.lower()}_checkpoint_selection.json"
    if selection_path.exists():
        label = str(load_json(selection_path)["selected_checkpoint_label"])
        return root / "checkpoints" / parent / f"{label}.pt"
    if debug:
        return root / "checkpoints" / parent / "best_loss.pt"
    if parent in {"U3", "I3"}:
        raise FileNotFoundError(f"Validation Recall@100 checkpoint selection is required before training children of {parent}")
    return root / "checkpoints" / parent / "best_loss.pt"


def train_variant(
    variant: str,
    config: Mapping[str, object],
    paths: Mapping[str, Path],
    debug: bool,
    overwrite: bool,
    device_name: str | None = None,
) -> dict[str, object]:
    started = time.perf_counter()
    seed = int(config["seed"]); seed_everything(seed)
    root = paths["output_root"]
    checkpoint_dir = root / "checkpoints" / variant
    best_path, final_path = checkpoint_dir / "best_loss.pt", checkpoint_dir / "final.pt"
    manifest_path = root / "manifests" / f"{variant.lower()}_training_manifest.json"
    history_path = root / "reports" / f"{variant.lower()}_training_history.csv"
    guard_outputs([best_path, final_path, manifest_path, history_path], overwrite)
    time_stats = TimeNormalization.from_json(load_json(root / "audits" / "time_normalization.json"))
    store = FeatureStore(paths["stage4_root"])
    item_store = Stage6ItemStore(paths["stage4_root"], config["item_tower"]["side_fields"])
    model = build_model(variant, config, int(store.rid_to_token.size + 1), item_store.side_vocab_sizes)
    parent = PARENT.get(variant); parent_loaded = 0
    if parent:
        parent_path = _selected_parent_checkpoint(root, parent, debug)
        if not parent_path.exists():
            raise FileNotFoundError(f"train {parent} before {variant}: {parent_path}")
        parent_loaded = _load_compatible_parent(model, parent_path)
    if variant in {"I1", "I2", "I3", "E1"}:
        _freeze_user_for_item_ablation(model)
    device = select_device(device_name); model.to(device)
    sparse_optimizer, dense_optimizer = make_optimizers(model, config)
    tokens = np.asarray(item_store.rid_to_token[np.flatnonzero(np.asarray(item_store.count_by_rid) > 0)], dtype=np.int64)
    gap, short_max, long_max = configured_session(config, debug)
    section = config["training"]
    epochs = int(config["debug"]["epochs"] if debug else section["max_epochs"])
    max_train = int(config["debug"]["max_train_samples"]) if debug else None
    max_val = int(config["debug"]["max_eval_samples"]) if debug else section.get("validation_max_samples")

    def loader(filename, maximum, validation, epoch_seed):
        dataset = Stage6ParquetDataset(paths["stage3_root"] / "samples" / filename, store, maximum, int(config["scan_batch_size"]))
        sampler = UniformNegativeSampler(tokens, int(section["random_negatives"]), epoch_seed)
        collator = Stage6Collator(gap, item_store, time_stats, sampler, short_max, long_max, True)
        return DataLoader(dataset, batch_size=int(section["batch_size"]), collate_fn=collator,
                          num_workers=int(section["num_workers"]), pin_memory=bool(section["pin_memory"]))

    stopper = ValidationLossEarlyStopping(int(section["early_stopping_patience"]), float(section["early_stopping_min_delta"]))
    rows = []; completed = 0
    protocols = {"stage3": config["stage3_protocol_version"], "stage4": config["stage4_protocol_version"], "stage6": config["stage6_protocol_version"]}
    for epoch in range(1, epochs + 1):
        train_loss, train_count = run_epoch(model, loader("train_samples.parquet", max_train, False, seed + epoch), device, (sparse_optimizer, dense_optimizer))
        val_loss, val_count = run_epoch(model, loader("val_primary.parquet", max_val, True, seed + 100000), device)
        decision = stopper.update(val_loss, epoch); completed = epoch
        metadata = {"epoch": epoch, "variant": variant, "validation_loss": val_loss, "protocols": protocols, "debug": debug}
        if decision.is_best:
            atomic_save_checkpoint(best_path, model, None, None, metadata)
        rows.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                     "train_samples": train_count, "validation_samples": val_count, "is_best_loss": int(decision.is_best)})
        if decision.should_stop: break
    atomic_save_checkpoint(final_path, model, sparse_optimizer, dense_optimizer,
                           {"epoch": completed, "variant": variant, "validation_loss": rows[-1]["validation_loss"], "protocols": protocols, "debug": debug})
    save_csv(rows, ["epoch", "train_loss", "validation_loss", "train_samples", "validation_samples", "is_best_loss"], history_path, True)
    manifest = {
        "stage": "6.training", "variant": variant, "protocol_version": config["stage6_protocol_version"],
        "debug": debug, "debug_results_must_not_be_used_for_conclusions": debug,
        "parent_variant": parent, "compatible_parent_tensors_loaded": parent_loaded,
        "user_tower_frozen": variant in {"I1", "I2", "I3", "E1"},
        "negative_pool": "train_item_count > 0 only", "random_negatives": int(section["random_negatives"]),
        "best_loss_checkpoint": str(best_path), "final_checkpoint": str(final_path),
        "final_checkpoint_selection_pending_validation_recall100": True,
        "completed_epochs": completed, "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    save_json(manifest, manifest_path, True)
    return manifest
