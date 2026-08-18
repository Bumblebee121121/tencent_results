"""Stage 3.1: audit click-labeled pseudo targets and strict history prefixes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.click_target import ClickTargetStats, find_click_targets
from src.data.stage3_runtime import (
    common_parser_arguments,
    iter_sequence_rows,
    load_config,
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


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    names = ("min", "p25", "p50", "p75", "p90", "p95", "p99", "max")
    if values.size == 0:
        return {**{name: None for name in names}, "mean": None}
    array = np.asarray(values, dtype=np.int64)
    quantiles = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {
        **{name: float(value) for name, value in zip(names, quantiles)},
        "mean": float(np.mean(array, dtype=np.float64)),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root, output_root = runtime_paths(
        config, args.data_root, args.output_root, args.debug or args.max_users is not None
    )
    if args.max_users is not None and args.max_users <= 0:
        raise ValueError("--max-users must be positive")

    print("Stage 3.1: click-target prefix audit")
    print(f"seq input: {(data_root / 'seq').resolve()}")
    print(f"output root: {output_root.resolve()}")
    total = ClickTargetStats()
    per_user_click_parts: list[np.ndarray] = []
    history_length_parts: list[np.ndarray] = []
    user_count = 0
    batch_size = int(config.get("scan_batch_size", 8192))
    for batch_number, (user_ids, sequences) in enumerate(
        iter_sequence_rows(data_root / "seq", batch_size, args.max_users), start=1
    ):
        batch_stats = ClickTargetStats()
        for events in sequences:
            _, user_stats = find_click_targets(events)
            batch_stats.merge(user_stats)
        per_user_click_parts.append(
            np.asarray(batch_stats.click_targets_per_user, dtype=np.int32)
        )
        history_length_parts.append(np.asarray(batch_stats.history_lengths, dtype=np.int32))
        total.merge(batch_stats, retain_distributions=False)
        user_count += len(user_ids)
        if batch_number % 10 == 0:
            print(f"processed users={user_count:,}, click targets={total.click_target_count:,}")

    per_user_click_counts = (
        np.concatenate(per_user_click_parts)
        if per_user_click_parts
        else np.empty(0, dtype=np.int32)
    )
    history_lengths = (
        np.concatenate(history_length_parts)
        if history_length_parts
        else np.empty(0, dtype=np.int32)
    )
    if per_user_click_counts.size != user_count:
        raise AssertionError("per-user click-target statistics do not match processed users")
    if history_lengths.size != total.click_target_count:
        raise AssertionError("history-length statistics do not match click targets")
    users_with_click_count = int(np.count_nonzero(per_user_click_counts > 0))
    report = {
        "stage": "3.1",
        "schema_version": 2,
        "protocol_version": str(config.get("protocol_version", "click_target_prefix_v2")),
        "protocol": "action_type=1 interaction is target; history timestamp < target_timestamp",
        "debug": bool(args.debug or args.max_users is not None),
        "max_users": args.max_users,
        "processed_user_count": user_count,
        "total_event_count": total.total_event_count,
        "click_target_count": total.click_target_count,
        "users_with_click_count": users_with_click_count,
        "users_with_click_ratio": ratio(users_with_click_count, user_count),
        "unknown_action_count": total.unknown_action_count,
        "click_targets_per_user": distribution(per_user_click_counts),
        "history_length_before_target": distribution(history_lengths),
        "empty_history_target_count": total.empty_history_target_count,
        "empty_history_target_ratio": ratio(
            total.empty_history_target_count, total.click_target_count
        ),
        "same_timestamp_prefix_excluded_count": (
            total.same_timestamp_prefix_excluded_count
        ),
        "same_timestamp_prefix_excluded_definition": (
            "targets with at least one earlier sequence element at target_timestamp"
        ),
    }
    output_path = (
        output_root / "click_target_audit" / "click_target_audit_report.json"
    )
    save_json(report, output_path, args.overwrite)
    print(f"wrote: {output_path.resolve()}")


if __name__ == "__main__":
    main()
