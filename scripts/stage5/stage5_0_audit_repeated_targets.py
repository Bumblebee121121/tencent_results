"""Stage 5.0: audit whether each target RID already occurs in its strict history prefix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recall.audits import repeated_target_stats
from src.recall.data import Stage5SequenceStore
from src.recall.runtime import (
    Timer, add_common_arguments, configure_logging, guard_outputs, load_config,
    require_contracts, save_json, stage5_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-samples-per-split", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, stage3_root, stage4_root, output_root, log_root = stage5_paths(config, args.debug)
    logger = configure_logging(log_root, "stage5_0_audit_repeated_targets")
    require_contracts(stage3_root, stage4_root, config)
    timer = Timer()
    output_path = output_root / "audits" / "repeated_target_audit.json"
    guard_outputs([output_path], args.overwrite)
    max_samples = args.max_samples_per_split
    if args.debug and max_samples is None:
        max_samples = int(config["debug"]["repeated_target_max_samples"])
    store = Stage5SequenceStore(stage4_root)
    split_paths = {
        "train": stage3_root / "samples" / "train_samples.parquet",
        "validation_primary": stage3_root / "samples" / "val_primary.parquet",
        "test_primary": stage3_root / "samples" / "test_primary.parquet",
    }
    results = {}
    for split, path in split_paths.items():
        logger.info("auditing split=%s max_samples=%s", split, max_samples)
        results[split] = repeated_target_stats(
            path, store, batch_size=int(config["scan_batch_size"]), max_samples=max_samples
        )
    save_json(
        {
            "stage": "5.0", "schema_version": 1,
            "recall_protocol_version": config["recall_protocol_version"],
            "stage3_protocol_version": config["stage3_protocol_version"],
            "debug": bool(args.debug),
            "definition": "target_item_rid occurs at least once in seq[:history_end_position]",
            "retrieval_policy": {
                "exclude_history_items": bool(config["retrieval"]["exclude_history_items"]),
                "reason": "Stage 3 does not declare repeated historical items invalid targets",
            },
            "splits": results,
            "elapsed_seconds": timer.elapsed_seconds,
        },
        output_path, args.overwrite,
    )
    logger.info("repeated-target audit complete elapsed_seconds=%.2f", timer.elapsed_seconds)


if __name__ == "__main__":
    main()
