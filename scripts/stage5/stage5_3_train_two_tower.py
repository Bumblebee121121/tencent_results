"""Stage 5.3: train the pure-ID PyTorch Vanilla Two-Tower."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vanilla_two_tower import VanillaTwoTower
from src.recall.checkpoint import save_checkpoint
from src.recall.data import (
    ParquetSampleIterableDataset, Stage5SequenceStore, TwoTowerCollator,
    UniformNegativeSampler, train_seen_item_tokens,
)
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
            targets = batch["target_tokens"].to(device, non_blocking=True)
            negatives = batch["negative_tokens"].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = model.sampled_softmax_loss(history, targets, negatives)
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
        num_workers=int(section.get("num_workers", 0)), pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, stage3_root, stage4_root, output_root, log_root = stage5_paths(config, args.debug)
    logger = configure_logging(log_root, "stage5_3_train_two_tower")
    contracts = require_contracts(stage3_root, stage4_root, config)
    timer = Timer()
    seed = int(config["seed"])
    seed_everything(seed)
    device = select_device(args.device)
    section = dict(config["two_tower"])
    epochs = int(section["epochs"])
    max_train = args.max_train_samples
    max_validation = args.max_validation_samples or int(section["validation_max_samples"])
    if args.debug:
        epochs = int(config["debug"]["epochs"])
        max_train = max_train or int(config["debug"]["max_train_samples"])
        max_validation = args.max_validation_samples or int(config["debug"]["max_eval_samples"])

    root = output_root / "two_tower"
    checkpoint_path = root / "checkpoints" / "best.pt"
    history_path = root / "training_history.csv"
    manifest_path = root / "training_manifest.json"
    guard_outputs([checkpoint_path, history_path, manifest_path], args.overwrite)
    store = Stage5SequenceStore(stage4_root)
    negative_pool_tokens = train_seen_item_tokens(store)
    num_tokens = int(store.rid_to_token.size + 1)
    model = VanillaTwoTower(num_tokens, int(section["embedding_dim"])).to(device)
    optimizer = torch.optim.SparseAdam(model.parameters(), lr=float(section["learning_rate"]))
    logger.info(
        "device=%s item_tokens=%d train_seen_negative_pool=%d epochs=%d",
        device, num_tokens, negative_pool_tokens.size, epochs,
    )
    history_rows = []
    best_loss = float("inf")
    best_epoch = 0
    protocols = {
        "recall": config["recall_protocol_version"],
        "stage3": config["stage3_protocol_version"],
        "stage4": config["stage4_protocol_version"],
        "train_cutoff_exclusive": int(contracts["splits"]["train_raw_event_cutoff_exclusive"]),
    }
    for epoch in range(1, epochs + 1):
        train_loader = make_loader(
            stage3_root / "samples" / "train_samples.parquet", store, negative_pool_tokens,
            section, seed + epoch, max_train,
        )
        train_loss, train_count = run_epoch(model, train_loader, device, optimizer)
        # Recreate both sampler and loader with the same seed every epoch: validation loss is fixed.
        validation_loader = make_loader(
            stage3_root / "samples" / "val_primary.parquet", store, negative_pool_tokens,
            section, seed + 100000, max_validation, validation=True,
        )
        validation_loss, validation_count = run_epoch(model, validation_loader, device)
        is_best = validation_loss < best_loss
        if is_best:
            best_loss, best_epoch = validation_loss, epoch
            save_checkpoint(checkpoint_path, model, optimizer, epoch, config, protocols, validation_loss)
        history_rows.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
             "train_samples": train_count, "validation_samples": validation_count, "is_best": int(is_best)}
        )
        logger.info(
            "epoch=%d train_loss=%.6f validation_loss=%.6f train_samples=%d validation_samples=%d best=%s",
            epoch, train_loss, validation_loss, train_count, validation_count, is_best,
        )
    save_csv(
        history_rows,
        ["epoch", "train_loss", "validation_loss", "train_samples", "validation_samples", "is_best"],
        history_path, args.overwrite,
    )
    save_json(
        {
            "stage": "5.3", "schema_version": 1, "recall_protocol_version": config["recall_protocol_version"],
            "framework": "pytorch", "model": "VanillaTwoTower", "shared_item_embedding": True,
            "inputs": ["history_item_token", "target_or_negative_item_token"],
            "excluded_inputs": ["action", "timestamp", "side", "multimodal", "strength"],
            "history_pooling_ignored_tokens": {"PAD": 0, "UNK": 1},
            "device": str(device), "num_item_tokens": num_tokens,
            "negative_pool_scope": "all items with train_item_count > 0; independent of eval candidate membership",
            "train_seen_negative_pool_count": int(negative_pool_tokens.size),
            "validation_loss_target_scope": "Train-Seen targets only (model_item_token > UNK); Train-Unseen validation targets are excluded",
            "best_epoch": best_epoch, "best_validation_loss": best_loss, "debug": bool(args.debug),
            "elapsed_seconds": timer.elapsed_seconds,
        },
        manifest_path, args.overwrite,
    )
    logger.info("training complete best_epoch=%d elapsed_seconds=%.2f", best_epoch, timer.elapsed_seconds)


if __name__ == "__main__":
    main()
