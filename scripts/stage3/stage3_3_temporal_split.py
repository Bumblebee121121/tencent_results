"""Stage 3.3: split compact samples by global target-exposure time."""

from __future__ import annotations

import argparse
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sample_builder import SAMPLE_SCHEMA
from src.data.stage3_runtime import (
    ParquetSink,
    common_parser_arguments,
    guard_outputs,
    load_config,
    require_paths,
    runtime_paths,
    save_json,
)
from src.data.temporal_split import (
    assign_splits,
    choose_temporal_cutoffs,
    contains_sorted,
    earliest_sample_ids,
    first_n_sample_ids_per_user,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common_parser_arguments(parser)
    return parser.parse_args()


def bounds(values: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    selected = values[mask]
    return {"min": int(selected.min()), "max": int(selected.max())}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, output_root = runtime_paths(config, args.data_root, args.output_root, args.debug)
    input_path = output_root / "samples" / "all_samples.parquet"
    require_paths([input_path])

    paths = {
        "train": output_root / "samples" / "train_samples.parquet",
        "val_all": output_root / "samples" / "val_all_targets.parquet",
        "test_all": output_root / "samples" / "test_all_targets.parquet",
        "val_primary": output_root / "samples" / "val_primary.parquet",
        "test_primary": output_root / "samples" / "test_primary.parquet",
        "manifest": output_root / "splits" / "split_manifest.json",
    }
    guard_outputs(paths.values(), args.overwrite)

    dataset = ds.dataset(input_path, format="parquet")
    metadata_columns = ["sample_id", "user_id", "target_exposure_timestamp"]
    required = set(metadata_columns)
    if not required.issubset(dataset.schema.names):
        raise ValueError(f"sample index missing columns: {sorted(required - set(dataset.schema.names))}")
    id_parts, user_parts, timestamp_parts = [], [], []
    scanner = dataset.scanner(columns=metadata_columns, batch_size=65536)
    for batch_number, batch in enumerate(scanner.to_batches(), start=1):
        columns = {name: batch.column(index) for index, name in enumerate(metadata_columns)}
        id_parts.append(np.asarray(columns["sample_id"].to_numpy(), dtype=np.int64))
        user_parts.append(np.asarray(columns["user_id"].to_numpy(), dtype=np.int64))
        timestamp_parts.append(
            np.asarray(columns["target_exposure_timestamp"].to_numpy(), dtype=np.int64)
        )
        if batch_number % 20 == 0:
            print(f"read metadata batches={batch_number}")
    if not timestamp_parts:
        raise ValueError("sample index is empty")
    sample_ids = np.concatenate(id_parts)
    user_ids = np.concatenate(user_parts)
    timestamps = np.concatenate(timestamp_parts)

    train_ratio = float(config.get("train_ratio", 0.8))
    val_ratio = float(config.get("val_ratio", 0.1))
    test_ratio = float(config.get("test_ratio", 0.1))
    cutoffs = choose_temporal_cutoffs(timestamps, train_ratio, val_ratio, test_ratio)
    split_names = assign_splits(timestamps, cutoffs)
    train_mask = split_names == "train"
    val_mask = split_names == "val"
    test_mask = split_names == "test"

    val_primary_ids = earliest_sample_ids(
        sample_ids[val_mask], user_ids[val_mask], timestamps[val_mask]
    )
    test_primary_ids = earliest_sample_ids(
        sample_ids[test_mask], user_ids[test_mask], timestamps[test_mask]
    )
    limit = config.get("max_train_targets_per_user")
    if limit is None:
        train_ids = np.sort(sample_ids[train_mask])
    else:
        train_ids = first_n_sample_ids_per_user(
            sample_ids[train_mask], user_ids[train_mask], timestamps[train_mask], int(limit)
        )

    with ExitStack() as stack:
        sinks = {
            name: stack.enter_context(ParquetSink(paths[name], SAMPLE_SCHEMA, args.overwrite))
            for name in ("train", "val_all", "test_all", "val_primary", "test_primary")
        }
        scanner = dataset.scanner(batch_size=65536)
        for batch_number, batch in enumerate(scanner.to_batches(), start=1):
            table = pa.Table.from_batches([batch]).select(SAMPLE_SCHEMA.names)
            batch_ids = np.asarray(table["sample_id"].to_numpy(), dtype=np.int64)
            batch_times = np.asarray(
                table["target_exposure_timestamp"].to_numpy(), dtype=np.int64
            )
            batch_splits = assign_splits(batch_times, cutoffs)
            masks = {
                "train": (batch_splits == "train") & contains_sorted(train_ids, batch_ids),
                "val_all": batch_splits == "val",
                "test_all": batch_splits == "test",
                "val_primary": contains_sorted(val_primary_ids, batch_ids),
                "test_primary": contains_sorted(test_primary_ids, batch_ids),
            }
            for name, mask in masks.items():
                sinks[name].write_table(table.filter(pa.array(mask)))
            if batch_number % 20 == 0:
                print(f"wrote split batches={batch_number}")
        counts = {name: sink.row_count for name, sink in sinks.items()}

    train_bounds, val_bounds, test_bounds = (
        bounds(timestamps, train_mask),
        bounds(timestamps, val_mask),
        bounds(timestamps, test_mask),
    )
    if not (train_bounds["max"] < val_bounds["min"] < test_bounds["min"]):
        raise AssertionError("Train < Validation < Test invariant failed")

    total = timestamps.size
    manifest = {
        "stage": "3.3",
        "schema_version": 1,
        "debug": bool(args.debug),
        "sample_time_field": "target_exposure_timestamp",
        "requested_ratios": {
            "train": train_ratio,
            "validation": val_ratio,
            "test": test_ratio,
        },
        "actual_all_target_counts": {
            "train_before_per_user_cap": int(train_mask.sum()),
            "train": counts["train"],
            "validation": counts["val_all"],
            "test": counts["test_all"],
        },
        "actual_all_target_ratios": {
            "train_before_per_user_cap": float(train_mask.sum() / total),
            "validation": float(val_mask.sum() / total),
            "test": float(test_mask.sum() / total),
        },
        "primary_counts": {
            "validation": counts["val_primary"],
            "test": counts["test_primary"],
        },
        "max_train_targets_per_user": limit,
        "validation_start_timestamp": cutoffs.validation_start,
        "test_start_timestamp": cutoffs.test_start,
        "train_raw_event_cutoff_exclusive": cutoffs.validation_start,
        "time_bounds": {"train": train_bounds, "validation": val_bounds, "test": test_bounds},
        "timestamp_ties_cross_splits": False,
    }
    save_json(manifest, paths["manifest"], args.overwrite)
    print(f"split counts: {counts}")


if __name__ == "__main__":
    main()
