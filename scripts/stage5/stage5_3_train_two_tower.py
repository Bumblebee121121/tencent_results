"""Stage 5.3: train the pure-ID PyTorch Vanilla Two-Tower."""

from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vanilla_two_tower import VanillaTwoTower
from src.recall.checkpoint import (
    CpuCheckpointBuffer, load_training_checkpoint, restore_random_state, save_checkpoint,
)
from src.recall.data import (
    ParquetSampleIterableDataset, Stage5SequenceStore, TwoTowerCollator,
    UniformNegativeSampler, train_seen_item_tokens,
)
from src.recall.early_stopping import ValidationLossEarlyStopping
from src.recall.runtime import (
    Timer, add_common_arguments, configure_logging, guard_outputs, load_config,
    require_contracts, save_csv, save_json, seed_everything, select_device, stage5_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--device", help="e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument(
        "--resume", action="store_true",
        help="continue from resume.pt, or a legacy full best.pt, without --overwrite",
    )
    return parser.parse_args()


def run_epoch(model, loader, device, optimizer=None) -> tuple[float, int]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    sample_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            if not batch["rows"]:
                continue
            history = batch["history_tokens"].to(device, non_blocking=True)
            history_offsets = batch["history_offsets"].to(device, non_blocking=True)
            targets = batch["target_tokens"].to(device, non_blocking=True)
            negatives = batch["negative_tokens"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = model.sampled_softmax_loss(history, targets, negatives, history_offsets)
            if training:
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    model.item_embedding.weight[model.padding_idx].zero_()
            count = targets.shape[0]
            loss_sum += float(loss.detach().cpu()) * count
            sample_count += count
    return (loss_sum / sample_count if sample_count else float("inf")), sample_count


def make_loader(path, store, candidate_tokens, section, seed, max_rows, validation=False):
    sampler = UniformNegativeSampler(candidate_tokens, int(section["random_negatives"]), seed)
    max_history = section.get("max_history_length")
    collator = TwoTowerCollator(
        store, sampler, None if max_history is None else int(max_history),
        require_seen_target=True,
    )
    dataset = ParquetSampleIterableDataset(path, max_rows=max_rows)
    return DataLoader(
        dataset, batch_size=int(section["batch_size"]), collate_fn=collator,
        num_workers=int(section.get("num_workers", 0)),
        pin_memory=bool(section.get("pin_memory", False)) and torch.cuda.is_available(),
    )


def load_existing_history(path: Path, through_epoch: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if int(row["epoch"]) <= through_epoch]


def replace_resume_baseline(
    rows: list[dict], epoch: int, validation_loss: float, validation_count: int,
    validation_scope: str,
) -> None:
    for row in rows:
        row.setdefault("validation_scope", "legacy_limited_train_seen")
        if int(row["epoch"]) == epoch:
            row.update({
                "validation_loss": validation_loss,
                "validation_samples": validation_count,
                "is_best": 1,
                "significant_improvement": 1,
                "early_stopping_bad_epochs": 0,
                "validation_scope": validation_scope,
            })


HISTORY_FIELDS = [
    "epoch", "train_loss", "validation_loss", "train_samples", "validation_samples", "is_best",
    "significant_improvement", "early_stopping_bad_epochs", "validation_scope",
    "train_seconds", "validation_seconds", "checkpoint_seconds", "process_rss_mb",
    "process_private_mb", "system_available_mb", "system_commit_available_mb",
    "checkpoint_cpu_buffer_mb",
]


def memory_snapshot() -> dict[str, float | None]:
    try:
        import psutil

        process_info = psutil.Process().memory_info()
        virtual = psutil.virtual_memory()
        swap = psutil.swap_memory()
        commit_limit = int(virtual.total + swap.total)
        committed = int((virtual.total - virtual.available) + swap.used)
        return {
            "process_rss_mb": round(process_info.rss / (1024 ** 2), 1),
            "process_private_mb": round(getattr(process_info, "private", process_info.vms) / (1024 ** 2), 1),
            "system_available_mb": round(virtual.available / (1024 ** 2), 1),
            "system_commit_available_mb": round((commit_limit - committed) / (1024 ** 2), 1),
        }
    except (ImportError, OSError):
        return {
            "process_rss_mb": None, "process_private_mb": None,
            "system_available_mb": None, "system_commit_available_mb": None,
        }


def build_training_manifest(
    *, config, device, num_tokens, negative_pool_count, max_validation,
    validation_count, patience, min_delta, max_epochs, stopped_early, completed_epoch,
    resumed, resume_source_path, checkpoint_path, resume_checkpoint_path,
    resume_checkpoint_interval, stopper, debug, elapsed_seconds, status, phase_metrics,
) -> dict:
    return {
        "stage": "5.3", "schema_version": 6,
        "recall_protocol_version": config["recall_protocol_version"],
        "framework": "pytorch", "model": "VanillaTwoTower", "shared_item_embedding": True,
        "inputs": ["history_item_token", "target_or_negative_item_token"],
        "excluded_inputs": ["action", "timestamp", "side", "multimodal", "strength"],
        "history_pooling_ignored_tokens": {"PAD": 0, "UNK": 1},
        "device": str(device), "num_item_tokens": num_tokens,
        "negative_pool_scope": "all items with train_item_count > 0; independent of eval candidate membership",
        "train_seen_negative_pool_count": int(negative_pool_count),
        "validation_loss_target_scope": "Train-Seen targets only (model_item_token > UNK); Train-Unseen validation targets are excluded",
        "validation_uses_complete_split": max_validation is None,
        "validation_max_samples": max_validation,
        "validation_train_seen_sample_count": validation_count,
        "early_stopping": {
            "metric": "validation_loss", "mode": "min", "patience": patience,
            "min_delta": min_delta, "max_epochs": max_epochs,
            "comparison": "adjacent_epoch",
            "improvement": "previous_validation_loss - current_validation_loss",
            "stopped_early": stopped_early, "completed_epoch": completed_epoch,
        },
        "checkpoint_write_mode": "atomic_temp_then_replace",
        "checkpoint_cpu_staging": "reusable_cpu_tensor_buffer",
        "checkpoint_policy": {
            "best": "model-only, saved on every lower validation loss",
            "resume": "model+SparseAdam, saved periodically and at early-stop/max-epoch",
            "resume_interval_epochs": resume_checkpoint_interval,
            "best_path": str(checkpoint_path.resolve()),
            "resume_path": str(resume_checkpoint_path.resolve()),
        },
        "history_representation": "ragged_offsets_no_padding",
        "negative_sampling_implementation": "vectorized_batch_proposals_with_exact_exclusion",
        "last_epoch_phase_metrics": dict(phase_metrics),
        "status": status,
        "resumed": bool(resumed),
        "resumed_from": str(resume_source_path.resolve()) if resumed else None,
        "best_epoch": stopper.best_epoch, "best_validation_loss": stopper.best_loss,
        "debug": bool(debug),
        "elapsed_seconds": elapsed_seconds,
    }


def save_training_progress(history_rows, history_path, manifest, manifest_path) -> None:
    # These files are owned by this running invocation after the initial output guard.
    save_csv(history_rows, HISTORY_FIELDS, history_path, overwrite=True)
    save_json(manifest, manifest_path, overwrite=True)


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    config = load_config(args.config)
    _, stage3_root, stage4_root, output_root, log_root = stage5_paths(config, args.debug)
    logger = configure_logging(log_root, "stage5_3_train_two_tower")
    contracts = require_contracts(stage3_root, stage4_root, config)
    timer = Timer()
    seed = int(config["seed"])
    seed_everything(seed)
    device = select_device(args.device)
    section = dict(config["two_tower"])
    max_epochs = int(section.get("max_epochs", section.get("epochs", 3)))
    patience = int(section["early_stopping_patience"])
    min_delta = float(section["early_stopping_min_delta"])
    max_train = args.max_train_samples
    configured_validation_max = section.get("validation_max_samples")
    max_validation = (
        args.max_validation_samples
        if args.max_validation_samples is not None
        else (None if configured_validation_max is None else int(configured_validation_max))
    )
    validation_scope = "complete_train_seen" if max_validation is None else "limited_train_seen"
    if args.debug:
        max_epochs = int(config["debug"]["epochs"])
        max_train = max_train if max_train is not None else int(config["debug"]["max_train_samples"])
        max_validation = (
            args.max_validation_samples
            if args.max_validation_samples is not None
            else int(config["debug"]["max_eval_samples"])
        )
        validation_scope = "limited_train_seen"

    root = output_root / "two_tower"
    checkpoint_path = root / "checkpoints" / "best.pt"
    resume_checkpoint_path = root / "checkpoints" / "resume.pt"
    history_path = root / "training_history.csv"
    manifest_path = root / "training_manifest.json"
    if args.resume:
        resume_source_path = resume_checkpoint_path if resume_checkpoint_path.exists() else checkpoint_path
        if not resume_source_path.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_source_path.resolve()}")
    else:
        resume_source_path = checkpoint_path
        guard_outputs([checkpoint_path, resume_checkpoint_path, history_path, manifest_path], args.overwrite)
    store = Stage5SequenceStore(stage4_root)
    negative_pool_tokens = train_seen_item_tokens(store)
    num_tokens = int(store.rid_to_token.size + 1)
    model = VanillaTwoTower(num_tokens, int(section["embedding_dim"])).to(device)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=float(section["learning_rate"]))
    checkpoint_buffer = CpuCheckpointBuffer()
    protocols = {
        "recall": config["recall_protocol_version"],
        "stage3": config["stage3_protocol_version"],
        "stage4": config["stage4_protocol_version"],
        "train_cutoff_exclusive": int(contracts["splits"]["train_raw_event_cutoff_exclusive"]),
    }
    start_epoch = 1
    resume_interval = int(section.get("resume_checkpoint_interval", 3))
    if resume_interval < 1:
        raise ValueError("resume_checkpoint_interval must be at least 1")
    last_resume_epoch = 0
    history_rows: list[dict] = []
    stopper = ValidationLossEarlyStopping(patience, min_delta)
    if args.resume:
        checkpoint = load_training_checkpoint(resume_source_path, model, optimizer, map_location="cpu")
        if checkpoint.get("protocols") != protocols:
            raise ValueError("resume checkpoint protocol does not match the current Stage 3/4 contract")
        restore_random_state(checkpoint["random_state"])
        for group in optimizer.param_groups:
            group["lr"] = float(section["learning_rate"])
        checkpoint_epoch = int(checkpoint["epoch"])
        last_resume_epoch = checkpoint_epoch if resume_checkpoint_path.exists() else 0
        start_epoch = checkpoint_epoch + 1
        history_rows = load_existing_history(history_path, checkpoint_epoch)
        checkpoint_training_state = checkpoint.get("training_state") or {}
        same_validation_scope = (
            checkpoint_training_state.get("validation_scope") == validation_scope
            and checkpoint_training_state.get("validation_max_samples") == max_validation
        )
        if same_validation_scope:
            stopper = ValidationLossEarlyStopping.resume(
                checkpoint_training_state, patience=patience, min_delta=min_delta,
                checkpoint_loss=float(checkpoint["validation_loss"]), checkpoint_epoch=checkpoint_epoch,
            )
        else:
            logger.info(
                "recomputing epoch=%d validation baseline with scope=%s before continuation",
                checkpoint_epoch, validation_scope,
            )
            baseline_loader = make_loader(
                stage3_root / "samples" / "val_primary.parquet", store, negative_pool_tokens,
                section, seed + 100000, max_validation, validation=True,
            )
            baseline_loss, baseline_count = run_epoch(model, baseline_loader, device)
            stopper = ValidationLossEarlyStopping(
                patience, min_delta, best_loss=baseline_loss, best_epoch=checkpoint_epoch,
                previous_loss=baseline_loss,
            )
            replace_resume_baseline(
                history_rows, checkpoint_epoch, baseline_loss, baseline_count, validation_scope,
            )
            save_checkpoint(
                resume_checkpoint_path, model, optimizer, checkpoint_epoch, config, protocols, baseline_loss,
                training_state={
                    **stopper.state_dict(), "validation_scope": validation_scope,
                    "validation_max_samples": max_validation,
                }, cpu_buffer=checkpoint_buffer,
            )
            save_checkpoint(
                checkpoint_path, model, None, checkpoint_epoch, config, protocols, baseline_loss,
                training_state={
                    **stopper.state_dict(), "validation_scope": validation_scope,
                    "validation_max_samples": max_validation,
                }, cpu_buffer=checkpoint_buffer,
            )
            last_resume_epoch = checkpoint_epoch
            logger.info(
                "recomputed validation baseline loss=%.6f train_seen_samples=%d",
                baseline_loss, baseline_count,
            )
        logger.info(
            "resuming checkpoint=%s completed_epoch=%d best_validation_loss=%.6f",
            resume_source_path, checkpoint_epoch, stopper.best_loss,
        )
    if start_epoch > max_epochs:
        raise ValueError(
            f"checkpoint epoch {start_epoch - 1} already reached configured max_epochs={max_epochs}"
        )
    logger.info(
        "device=%s item_tokens=%d train_seen_negative_pool=%d epochs=%d..%d "
        "validation_max_samples=%s early_stopping_patience=%d min_delta=%.6f",
        device, num_tokens, negative_pool_tokens.size, start_epoch, max_epochs,
        max_validation, patience, min_delta,
    )
    stopped_early = False
    completed_epoch = start_epoch - 1
    validation_count = 0
    phase_metrics: dict[str, float | None] = {}
    for epoch in range(start_epoch, max_epochs + 1):
        train_started = time.perf_counter()
        train_loader = make_loader(
            stage3_root / "samples" / "train_samples.parquet", store, negative_pool_tokens,
            section, seed + epoch, max_train,
        )
        train_loss, train_count = run_epoch(model, train_loader, device, optimizer)
        train_seconds = time.perf_counter() - train_started
        # Recreate both sampler and loader with the same seed every epoch: validation loss is fixed.
        validation_started = time.perf_counter()
        validation_loader = make_loader(
            stage3_root / "samples" / "val_primary.parquet", store, negative_pool_tokens,
            section, seed + 100000, max_validation, validation=True,
        )
        validation_loss, validation_count = run_epoch(model, validation_loader, device)
        validation_seconds = time.perf_counter() - validation_started
        decision = stopper.update(validation_loss, epoch)
        checkpoint_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        training_state = {
            **stopper.state_dict(), "validation_scope": validation_scope,
            "validation_max_samples": max_validation,
        }
        save_resume = (
            (not args.resume and epoch == 1)
            or not resume_checkpoint_path.exists()
            or epoch - last_resume_epoch >= resume_interval
            or decision.should_stop
            or epoch == max_epochs
        )
        if save_resume:
            save_checkpoint(
                resume_checkpoint_path, model, optimizer, epoch, config, protocols, validation_loss,
                training_state=training_state, cpu_buffer=checkpoint_buffer,
            )
            last_resume_epoch = epoch
        if decision.is_best:
            save_checkpoint(
                checkpoint_path, model, None, epoch, config, protocols, validation_loss,
                training_state=training_state, cpu_buffer=checkpoint_buffer,
            )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        memory = memory_snapshot()
        phase_metrics = {
            "train_seconds": round(train_seconds, 3),
            "validation_seconds": round(validation_seconds, 3),
            "checkpoint_seconds": round(checkpoint_seconds, 3),
            "checkpoint_cpu_buffer_mb": round(checkpoint_buffer.allocated_bytes / (1024 ** 2), 1),
            **memory,
        }
        history_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
             "train_samples": train_count, "validation_samples": validation_count,
             "is_best": int(decision.is_best),
             "significant_improvement": int(decision.significant_improvement),
             "early_stopping_bad_epochs": stopper.bad_epochs,
             "validation_scope": validation_scope,
             **phase_metrics}
        )
        completed_epoch = epoch
        logger.info(
            "epoch=%d train_loss=%.6f validation_loss=%.6f train_samples=%d validation_samples=%d "
            "best=%s significant=%s bad_epochs=%d/%d train_seconds=%.2f validation_seconds=%.2f "
            "checkpoint_seconds=%.2f rss_mb=%s private_mb=%s commit_available_mb=%s",
            epoch, train_loss, validation_loss, train_count, validation_count, decision.is_best,
            decision.significant_improvement, stopper.bad_epochs, patience,
            train_seconds, validation_seconds, checkpoint_seconds,
            memory["process_rss_mb"], memory["process_private_mb"],
            memory["system_commit_available_mb"],
        )
        epoch_stopped_early = decision.should_stop
        progress_manifest = build_training_manifest(
            config=config, device=device, num_tokens=num_tokens,
            negative_pool_count=negative_pool_tokens.size, max_validation=max_validation,
            validation_count=validation_count, patience=patience, min_delta=min_delta,
            max_epochs=max_epochs, stopped_early=epoch_stopped_early,
            completed_epoch=completed_epoch, resumed=args.resume,
            resume_source_path=resume_source_path, checkpoint_path=checkpoint_path,
            resume_checkpoint_path=resume_checkpoint_path, resume_checkpoint_interval=resume_interval,
            stopper=stopper, debug=args.debug, elapsed_seconds=timer.elapsed_seconds,
            status="early_stopped" if epoch_stopped_early else "running",
            phase_metrics=phase_metrics,
        )
        save_training_progress(history_rows, history_path, progress_manifest, manifest_path)
        if decision.should_stop:
            stopped_early = True
            logger.info(
                "early stopping at epoch=%d: adjacent-epoch validation loss did not improve "
                "by at least %.6f for %d consecutive epochs",
                epoch, min_delta, patience,
            )
            break
    final_manifest = build_training_manifest(
        config=config, device=device, num_tokens=num_tokens,
        negative_pool_count=negative_pool_tokens.size, max_validation=max_validation,
        validation_count=validation_count, patience=patience, min_delta=min_delta,
        max_epochs=max_epochs, stopped_early=stopped_early, completed_epoch=completed_epoch,
        resumed=args.resume, resume_source_path=resume_source_path, checkpoint_path=checkpoint_path,
        resume_checkpoint_path=resume_checkpoint_path, resume_checkpoint_interval=resume_interval,
        stopper=stopper, debug=args.debug,
        elapsed_seconds=timer.elapsed_seconds,
        status="early_stopped" if stopped_early else "max_epochs_reached",
        phase_metrics=phase_metrics,
    )
    save_training_progress(history_rows, history_path, final_manifest, manifest_path)
    logger.info(
        "training complete completed_epoch=%d best_epoch=%d best_validation_loss=%.6f elapsed_seconds=%.2f",
        completed_epoch, stopper.best_epoch, stopper.best_loss, timer.elapsed_seconds,
    )


if __name__ == "__main__":
    main()
