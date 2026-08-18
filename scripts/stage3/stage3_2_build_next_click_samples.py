"""Stage 3.2: build the compact next-click sample index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.click_target import ClickTargetStats
from src.data.sample_builder import SAMPLE_SCHEMA, build_samples_for_users, make_rid_to_oid
from src.data.stage3_runtime import (
    ParquetSink,
    common_parser_arguments,
    guard_outputs,
    iter_sequence_rows,
    load_config,
    load_item_mapping,
    runtime_paths,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common_parser_arguments(parser)
    parser.add_argument("--max-users", type=int, help="limit complete users for smoke/debug runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if bool(config.get("materialize_history", False)):
        raise ValueError("Stage 3 compact protocol requires materialize_history: false")
    data_root, output_root = runtime_paths(
        config, args.data_root, args.output_root, args.debug or args.max_users is not None
    )
    if args.max_users is not None and args.max_users <= 0:
        raise ValueError("--max-users must be positive")

    sample_path = output_root / "samples" / "all_samples.parquet"
    manifest_path = output_root / "samples" / "sample_manifest.json"
    guard_outputs([sample_path, manifest_path], args.overwrite)

    print("Stage 3.2: build compact next-click sample index")
    mapping = load_item_mapping(data_root / "indexer.pkl")
    rid_to_oid = make_rid_to_oid(mapping)
    del mapping

    total = ClickTargetStats()
    user_count = 0
    next_sample_id = 0
    batch_size = int(config.get("scan_batch_size", 8192))
    with ParquetSink(sample_path, SAMPLE_SCHEMA, args.overwrite) as sink:
        for batch_number, (user_ids, sequences) in enumerate(
            iter_sequence_rows(data_root / "seq", batch_size, args.max_users), start=1
        ):
            rows, batch_stats = build_samples_for_users(
                user_ids, sequences, rid_to_oid, first_sample_id=next_sample_id
            )
            sink.write_rows(rows)
            next_sample_id += len(rows)
            # This stage only needs exact counters; Stage 3.1 owns distributions.
            total.merge(batch_stats, retain_distributions=False)
            user_count += len(user_ids)
            if batch_number % 10 == 0:
                print(f"processed users={user_count:,}, samples={sink.row_count:,}")
        sample_count = sink.row_count

    expected_sample_count = total.click_target_count - total.empty_history_target_count
    if sample_count != expected_sample_count:
        raise AssertionError(
            "written samples must equal click targets minus empty-history targets"
        )
    manifest = {
        "stage": "3.2",
        "schema_version": 2,
        "protocol_version": str(config.get("protocol_version", "click_target_prefix_v2")),
        "debug": bool(args.debug or args.max_users is not None),
        "max_users": args.max_users,
        "processed_user_count": user_count,
        "sample_count": sample_count,
        "total_event_count": total.total_event_count,
        "click_target_count": total.click_target_count,
        "unknown_action_count": total.unknown_action_count,
        "empty_history_target_count": total.empty_history_target_count,
        "empty_history_target_ratio": (
            total.empty_history_target_count / total.click_target_count
            if total.click_target_count
            else None
        ),
        "skipped_empty_history_target_count": total.empty_history_target_count,
        "empty_history_policy": "count in audit and manifest; exclude from formal samples",
        "materialize_history": False,
        "history_slice": "seq[:history_end_position]",
        "target_definition": "action_type == 1 interaction",
        "history_timestamp_invariant": "history timestamp < target_timestamp",
        "sample_index_path": str(sample_path.relative_to(PROJECT_ROOT)),
        "columns": SAMPLE_SCHEMA.names,
    }
    save_json(manifest, manifest_path, args.overwrite)
    print(f"wrote {sample_count:,} samples: {sample_path.resolve()}")


if __name__ == "__main__":
    main()
