"""Stage 3.1: audit nearest preceding same-item exposure attribution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.attribution import AttributionStats
from src.data.sample_builder import build_samples_for_users, make_rid_to_oid
from src.data.stage3_runtime import (
    common_parser_arguments,
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


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root, output_root = runtime_paths(
        config, args.data_root, args.output_root, args.debug or args.max_users is not None
    )
    max_users = args.max_users
    if max_users is not None and max_users <= 0:
        raise ValueError("--max-users must be positive")

    print("Stage 3.1: click -> exposure attribution audit")
    print(f"seq input: {(data_root / 'seq').resolve()}")
    print(f"output root: {output_root.resolve()}")
    mapping = load_item_mapping(data_root / "indexer.pkl")
    rid_to_oid = make_rid_to_oid(mapping)
    del mapping

    total = AttributionStats()
    user_count = 0
    batch_size = int(config.get("scan_batch_size", 8192))
    for batch_number, (user_ids, sequences) in enumerate(
        iter_sequence_rows(data_root / "seq", batch_size, max_users), start=1
    ):
        _, batch_stats = build_samples_for_users(user_ids, sequences, rid_to_oid)
        total.merge(batch_stats)
        user_count += len(user_ids)
        if batch_number % 10 == 0:
            print(f"processed users={user_count:,}, clicks={total.click_count:,}")

    gaps = np.asarray(total.gaps, dtype=np.int64)
    quantile_names = ("min", "median", "p90", "p95", "p99", "max")
    if gaps.size:
        quantile_values = np.quantile(gaps, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
        gap_summary = {
            name: float(value) for name, value in zip(quantile_names, quantile_values)
        }
    else:
        gap_summary = {name: None for name in quantile_names}

    report = {
        "stage": "3.1",
        "protocol": "nearest preceding exposure for the same user and item; no window",
        "debug": bool(args.debug or args.max_users is not None),
        "max_users": max_users,
        "processed_user_count": user_count,
        "click_count": total.click_count,
        "attributed_click_count": total.attributed_count,
        "attribution_failure_count": total.failure_count,
        "attribution_coverage": ratio(total.attributed_count, total.click_count),
        "exposure_to_click_gap_seconds": gap_summary,
        "multiple_preceding_exposure_count": total.multiple_preceding_exposure_count,
        "multiple_preceding_exposure_ratio": ratio(
            total.multiple_preceding_exposure_count, total.attributed_count
        ),
        "same_timestamp_count": total.same_timestamp_count,
        "same_timestamp_ratio": ratio(total.same_timestamp_count, total.attributed_count),
    }
    output_path = output_root / "attribution" / "attribution_report.json"
    save_json(report, output_path, args.overwrite)
    print(f"wrote: {output_path.resolve()}")


if __name__ == "__main__":
    main()
