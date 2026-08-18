"""Stage 3.5: compute train-only item strength and target distributions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.item_strength import STRENGTH_GROUPS, classify_strength, strength_thresholds
from src.data.stage3_runtime import (
    ParquetSink,
    common_parser_arguments,
    guard_outputs,
    load_config,
    load_item_mapping,
    require_paths,
    require_protocol_manifest,
    runtime_paths,
    save_csv,
    save_json,
)


ITEM_STRENGTH_SCHEMA = pa.schema(
    [
        ("item_oid", pa.int64()),
        ("item_rid", pa.int64()),
        ("train_event_count", pa.int64()),
        ("strength_group", pa.string()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common_parser_arguments(parser)
    parser.add_argument("--max-users", type=int, help="limit raw users in debug mode")
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def target_distribution(
    target_path: Path, counts: np.ndarray, p50: float, p90: float
) -> list[dict[str, int | float | str]]:
    group_counts = {group: 0 for group in STRENGTH_GROUPS}
    total = 0
    dataset = ds.dataset(target_path, format="parquet")
    for batch in dataset.scanner(columns=["target_item_rid"], batch_size=65536).to_batches():
        for rid_value in batch.column(0).to_pylist():
            rid = int(rid_value)
            count = int(counts[rid]) if 0 < rid < counts.size else 0
            group_counts[classify_strength(count, p50, p90)] += 1
            total += 1
    return [
        {
            "strength_group": group,
            "target_count": group_counts[group],
            "target_ratio": group_counts[group] / total if total else 0.0,
        }
        for group in STRENGTH_GROUPS
    ]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    debug = args.debug or args.max_users is not None
    data_root, output_root = runtime_paths(config, args.data_root, args.output_root, debug)
    split_manifest_path = output_root / "splits" / "split_manifest.json"
    candidate_path = output_root / "candidates" / "eval_candidates.parquet"
    candidate_manifest_path = output_root / "candidates" / "eval_candidate_manifest.json"
    val_path = output_root / "samples" / "val_primary.parquet"
    test_path = output_root / "samples" / "test_primary.parquet"
    require_paths(
        [
            split_manifest_path,
            candidate_path,
            candidate_manifest_path,
            val_path,
            test_path,
            data_root / "seq",
            data_root / "indexer.pkl",
        ]
    )
    output_paths = {
        "counts": output_root / "item_strength" / "item_train_counts.parquet",
        "thresholds": output_root / "item_strength" / "item_strength_thresholds.json",
        "val": output_root / "item_strength" / "val_target_strength_distribution.csv",
        "test": output_root / "item_strength" / "test_target_strength_distribution.csv",
    }
    guard_outputs(output_paths.values(), args.overwrite)

    manifest = load_manifest(split_manifest_path)
    protocol_version = str(config.get("protocol_version", "click_target_prefix_v2"))
    if manifest.get("protocol_version") != protocol_version:
        raise ValueError("split manifest protocol_version does not match Stage 3 config")
    require_protocol_manifest(candidate_manifest_path, protocol_version)
    cutoff = int(manifest["train_raw_event_cutoff_exclusive"])
    mapping = load_item_mapping(data_root / "indexer.pkl")
    max_rid = max(int(value) for value in mapping.values())
    counts = np.zeros(max_rid + 1, dtype=np.int64)

    seq_dataset = ds.dataset(data_root / "seq", format="parquet")
    processed_users = 0
    counted_events = 0
    batch_size = int(config.get("scan_batch_size", 8192))
    scanner = seq_dataset.scanner(columns=["seq"], batch_size=batch_size)
    for batch_number, batch in enumerate(scanner.to_batches(), start=1):
        if args.max_users is not None:
            remaining = args.max_users - processed_users
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
        sequence_array = batch.column(0)
        events = pc.list_flatten(sequence_array)
        item_values = np.asarray(events.field("item_id").to_numpy(), dtype=np.int64)
        timestamps = np.asarray(events.field("timestamp").to_numpy(), dtype=np.int64)
        selected = item_values[timestamps < cutoff]
        if selected.size and (selected.min() <= 0 or selected.max() > max_rid):
            raise ValueError("raw sequence contains item RID outside the official mapping")
        counts += np.bincount(selected, minlength=counts.size)
        counted_events += int(selected.size)
        processed_users += batch.num_rows
        if batch_number % 20 == 0:
            print(f"processed raw users={processed_users:,}, train events={counted_events:,}")

    p50, p90 = strength_thresholds(counts)
    candidate_dataset = ds.dataset(candidate_path, format="parquet")
    with ParquetSink(output_paths["counts"], ITEM_STRENGTH_SCHEMA, args.overwrite) as sink:
        for batch in candidate_dataset.scanner(
            columns=["item_oid", "item_rid"], batch_size=65536
        ).to_batches():
            rows = []
            for oid_value, rid_value in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
                count = 0 if rid_value is None else int(counts[int(rid_value)])
                rows.append(
                    {
                        "item_oid": int(oid_value),
                        "item_rid": None if rid_value is None else int(rid_value),
                        "train_event_count": count,
                        "strength_group": classify_strength(count, p50, p90),
                    }
                )
            sink.write_rows(rows)
        candidate_count = sink.row_count

    val_rows = target_distribution(val_path, counts, p50, p90)
    test_rows = target_distribution(test_path, counts, p50, p90)
    fields = ["strength_group", "target_count", "target_ratio"]
    save_csv(val_rows, fields, output_paths["val"], args.overwrite)
    save_csv(test_rows, fields, output_paths["test"], args.overwrite)
    threshold_report = {
        "stage": "3.5",
        "schema_version": 2,
        "protocol_version": protocol_version,
        "debug": debug,
        "max_users": args.max_users,
        "definition": "count raw events whose timestamp is strictly before the train cutoff",
        "train_raw_event_cutoff_exclusive": cutoff,
        "processed_user_count": processed_users,
        "counted_train_event_count": counted_events,
        "train_seen_item_count": int(np.count_nonzero(counts)),
        "p50_train": p50,
        "p90_train": p90,
        "groups": {
            "Unseen": "count == 0",
            "Tail": "0 < count <= p50_train",
            "Mid": "p50_train < count <= p90_train",
            "Head": "count > p90_train",
        },
        "item_train_counts_scope": "evaluation candidate pool",
        "evaluation_candidate_count": candidate_count,
        "target_distribution_scope": "primary validation/test targets",
    }
    save_json(threshold_report, output_paths["thresholds"], args.overwrite)
    print(f"train thresholds: p50={p50}, p90={p90}")


if __name__ == "__main__":
    main()
