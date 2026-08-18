"""Stage 3.7: audit Train-unseen target drift over evaluation time."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.temporal_drift import (
    SPLITS,
    STRENGTH_GROUPS,
    aggregate_temporal_buckets,
    build_split_summary,
    encode_strength_groups,
    lookup_strength_codes,
    quantify_unseen_drift,
    units_per_day,
    validate_fixed_strength_groups,
    verify_split_summary,
)
from src.data.stage3_runtime import (
    common_parser_arguments,
    guard_outputs,
    load_config,
    require_paths,
    require_protocol_manifest,
    runtime_paths,
    save_csv,
    save_json,
)


BUCKET_FIELDS = [
    "analysis_scope",
    "date",
    "calendar_date",
    "bucket_start_timestamp",
    "days_from_train_cutoff",
    "split",
    "target_count",
    "head_count",
    "mid_count",
    "tail_count",
    "unseen_count",
    "head_ratio",
    "mid_ratio",
    "tail_ratio",
    "unseen_ratio",
    "used_for_trend",
]
SUMMARY_FIELDS = [
    "analysis_scope",
    "split",
    "target_count",
    "head_count",
    "mid_count",
    "tail_count",
    "unseen_count",
    "head_ratio",
    "mid_ratio",
    "tail_ratio",
    "unseen_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common_parser_arguments(parser)
    parser.add_argument(
        "--include-all-targets",
        action="store_true",
        help="also write the auxiliary val/test all-target daily audit",
    )
    parser.add_argument(
        "--min-targets-per-bucket",
        type=int,
        help="minimum combined daily targets required for trend fitting",
    )
    return parser.parse_args()


def load_strength_lookup(
    path: Path,
    p50_train: float,
    p90_train: float,
) -> tuple[np.ndarray, np.ndarray]:
    dataset = ds.dataset(path, format="parquet")
    required = {"item_oid", "train_event_count", "strength_group"}
    if not required.issubset(dataset.schema.names):
        raise ValueError(f"item_train_counts missing columns: {sorted(required - set(dataset.schema.names))}")
    table = dataset.to_table(columns=["item_oid", "train_event_count", "strength_group"])
    item_oids = np.asarray(table["item_oid"].to_numpy(), dtype=np.int64)
    counts = np.asarray(table["train_event_count"].to_numpy(), dtype=np.int64)
    codes = encode_strength_groups(table["strength_group"].to_pylist())
    validate_fixed_strength_groups(counts, codes, p50_train, p90_train)
    order = np.argsort(item_oids, kind="stable")
    sorted_oids = item_oids[order]
    sorted_codes = codes[order]
    if sorted_oids.size > 1 and np.any(sorted_oids[1:] == sorted_oids[:-1]):
        raise ValueError("item_train_counts contains duplicate item_oid values")
    return sorted_oids, sorted_codes


def load_target_arrays(
    paths: list[tuple[str, Path]],
    sorted_item_oids: np.ndarray,
    sorted_strength_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if [split for split, _ in paths] != list(SPLITS):
        raise ValueError(f"target paths must be ordered as {SPLITS}")
    timestamp_parts: list[np.ndarray] = []
    strength_parts: list[np.ndarray] = []
    split_parts: list[np.ndarray] = []
    for split_code, (split, path) in enumerate(paths):
        dataset = ds.dataset(path, format="parquet")
        required = {"target_timestamp", "target_item_oid"}
        if not required.issubset(dataset.schema.names):
            raise ValueError(f"target dataset missing columns: {sorted(required - set(dataset.schema.names))}")
        direct_group_column = next(
            (name for name in ("item_strength_group", "strength_group") if name in dataset.schema.names),
            None,
        )
        columns = ["target_timestamp", "target_item_oid"]
        if direct_group_column:
            columns.append(direct_group_column)
        scanner = dataset.scanner(columns=columns, batch_size=65536)
        row_count = 0
        for batch_number, batch in enumerate(scanner.to_batches(), start=1):
            timestamps = np.asarray(batch.column(0).to_numpy(), dtype=np.int64)
            if direct_group_column:
                codes = encode_strength_groups(batch.column(2).to_pylist())
            else:
                target_oids = np.asarray(batch.column(1).to_numpy(), dtype=np.int64)
                codes = lookup_strength_codes(
                    target_oids, sorted_item_oids, sorted_strength_codes
                )
            timestamp_parts.append(timestamps)
            strength_parts.append(codes)
            split_parts.append(np.full(batch.num_rows, split_code, dtype=np.int8))
            row_count += batch.num_rows
            if batch_number % 20 == 0:
                print(f"loaded {split} target batches={batch_number}, rows={row_count:,}")
        print(f"loaded {split} targets={row_count:,} from {path.resolve()}")
    if not timestamp_parts:
        raise ValueError("target datasets are empty")
    return (
        np.concatenate(timestamp_parts),
        np.concatenate(strength_parts),
        np.concatenate(split_parts),
    )


def load_stage3_5_expected(
    validation_path: Path,
    test_path: Path,
) -> dict[str, dict[str, tuple[int, float]]]:
    result: dict[str, dict[str, tuple[int, float]]] = {}
    for split, path in (("Validation", validation_path), ("Test", test_path)):
        groups: dict[str, tuple[int, float]] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                group = row["strength_group"]
                if group not in STRENGTH_GROUPS:
                    raise ValueError(f"unexpected strength group in {path}: {group!r}")
                groups[group] = (int(row["target_count"]), float(row["target_ratio"]))
        if set(groups) != set(STRENGTH_GROUPS):
            raise ValueError(f"Stage 3.5 distribution is incomplete: {path.resolve()}")
        result[split] = groups
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, output_root = runtime_paths(config, args.data_root, args.output_root, args.debug)
    protocol_version = str(config.get("protocol_version", "click_target_prefix_v2"))
    drift_config = config.get("temporal_drift", {})
    if not isinstance(drift_config, dict):
        raise ValueError("temporal_drift config must be a mapping")
    timestamp_unit = str(config.get("timestamp_unit", ""))
    units_per_day(timestamp_unit)  # Validate the explicit unit before reading data.
    min_targets = (
        args.min_targets_per_bucket
        if args.min_targets_per_bucket is not None
        else int(drift_config.get("min_targets_per_bucket", 1))
    )
    include_all_targets = bool(
        args.include_all_targets or drift_config.get("include_all_targets", False)
    )
    if min_targets <= 0:
        raise ValueError("min_targets_per_bucket must be positive")

    split_manifest_path = output_root / "splits" / "split_manifest.json"
    threshold_path = output_root / "item_strength" / "item_strength_thresholds.json"
    count_path = output_root / "item_strength" / "item_train_counts.parquet"
    val_primary_path = output_root / "samples" / "val_primary.parquet"
    test_primary_path = output_root / "samples" / "test_primary.parquet"
    val_distribution_path = output_root / "item_strength" / "val_target_strength_distribution.csv"
    test_distribution_path = output_root / "item_strength" / "test_target_strength_distribution.csv"
    required_paths = [
        split_manifest_path,
        threshold_path,
        count_path,
        val_primary_path,
        test_primary_path,
        val_distribution_path,
        test_distribution_path,
    ]
    val_all_path = output_root / "samples" / "val_all_targets.parquet"
    test_all_path = output_root / "samples" / "test_all_targets.parquet"
    if include_all_targets:
        required_paths.extend([val_all_path, test_all_path])
    require_paths(required_paths)

    split_manifest = require_protocol_manifest(split_manifest_path, protocol_version)
    thresholds = require_protocol_manifest(threshold_path, protocol_version)
    train_cutoff = int(split_manifest["train_raw_event_cutoff_exclusive"])
    if int(thresholds["train_raw_event_cutoff_exclusive"]) != train_cutoff:
        raise ValueError("Stage 3.3 and Stage 3.5 train cutoffs do not match")
    p50_train = float(thresholds["p50_train"])
    p90_train = float(thresholds["p90_train"])

    output_dir = output_root / "temporal_drift"
    primary_path = output_dir / "temporal_drift_primary.csv"
    summary_path = output_dir / "temporal_drift_split_summary.csv"
    report_path = output_dir / "temporal_drift_report.json"
    output_paths = [primary_path, summary_path, report_path]
    all_targets_path = output_dir / "temporal_drift_all_targets.csv"
    if include_all_targets:
        output_paths.append(all_targets_path)
    guard_outputs(output_paths, args.overwrite)

    sorted_oids, sorted_codes = load_strength_lookup(
        count_path, p50_train, p90_train
    )
    primary_timestamps, primary_codes, primary_splits = load_target_arrays(
        [("Validation", val_primary_path), ("Test", test_primary_path)],
        sorted_oids,
        sorted_codes,
    )
    primary_rows = aggregate_temporal_buckets(
        primary_timestamps,
        primary_codes,
        primary_splits,
        train_cutoff,
        timestamp_unit,
        min_targets,
        "primary",
    )
    summary_rows = build_split_summary(primary_rows)
    expected = load_stage3_5_expected(val_distribution_path, test_distribution_path)
    verify_split_summary(summary_rows, expected)
    drift = quantify_unseen_drift(primary_rows, min_targets)

    summary_by_split = {str(row["split"]): row for row in summary_rows}
    report = {
        "stage": "3.7",
        "schema_version": 1,
        "protocol_version": protocol_version,
        "debug": bool(args.debug),
        "timestamp_unit": timestamp_unit,
        "timestamp_unit_source": "configs/stage3.yaml (consistent with Stage 2 temporal audit)",
        "calendar_timezone": "UTC",
        "bucket_definition": "fixed 24-hour buckets anchored at train_cutoff",
        "train_cutoff": train_cutoff,
        "p50_train_reused": p50_train,
        "p90_train_reused": p90_train,
        "analysis_start_timestamp": int(primary_timestamps.min()),
        "analysis_end_timestamp": int(primary_timestamps.max()),
        "primary_target_count": int(primary_timestamps.size),
        "validation_target_count": int(summary_by_split["Validation"]["target_count"]),
        "test_target_count": int(summary_by_split["Test"]["target_count"]),
        "validation_unseen_ratio": float(summary_by_split["Validation"]["unseen_ratio"]),
        "test_unseen_ratio": float(summary_by_split["Test"]["unseen_ratio"]),
        **drift,
        "min_targets_per_bucket": min_targets,
        "stage3_5_distribution_consistency_passed": True,
        "all_target_analysis_generated": include_all_targets,
        "unseen_definition": "n_i_train == 0 before the fixed Train cutoff",
        "interpretation_guardrail": (
            "descriptive temporal correlation only; not a causal or model-performance claim"
        ),
    }

    all_target_rows: list[dict[str, object]] | None = None
    if include_all_targets:
        all_timestamps, all_codes, all_splits = load_target_arrays(
            [("Validation", val_all_path), ("Test", test_all_path)],
            sorted_oids,
            sorted_codes,
        )
        all_target_rows = aggregate_temporal_buckets(
            all_timestamps,
            all_codes,
            all_splits,
            train_cutoff,
            timestamp_unit,
            min_targets,
            "all_targets",
        )

    # Write only after the Stage 3.5 consistency check has passed.
    save_csv(primary_rows, BUCKET_FIELDS, primary_path, args.overwrite)
    save_csv(summary_rows, SUMMARY_FIELDS, summary_path, args.overwrite)
    if all_target_rows is not None:
        save_csv(all_target_rows, BUCKET_FIELDS, all_targets_path, args.overwrite)
    save_json(report, report_path, args.overwrite)
    print(f"Stage 3.7 primary targets={primary_timestamps.size:,}")
    print(f"Stage 3.5 distribution consistency: passed")
    print(f"wrote: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
