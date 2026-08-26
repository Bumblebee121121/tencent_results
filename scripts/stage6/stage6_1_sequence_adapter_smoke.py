"""Stage 6.1: audit session definitions, truncation and Train-only time statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_store import FeatureStore
from src.recall.stage6_data import Stage6ParquetDataset, fit_time_normalization, split_long_short_session
from src.recall.stage6_runtime import (
    add_common_arguments, configured_session, guard_outputs, load_config,
    require_contracts, save_json, session_gap_candidates, stage6_paths,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__); add_common_arguments(parser); parser.add_argument("--max-samples", type=int); return parser.parse_args()


def quantiles(histogram, count):
    if not count:
        return {name: 0.0 for name in ("p50", "p90", "p95", "p99")}
    result = {}
    ordered = sorted(histogram.items())
    for name, q in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99)):
        threshold = q * (count - 1)
        cumulative = 0
        for value, frequency in ordered:
            cumulative += frequency
            if cumulative - 1 >= threshold:
                result[name] = float(value)
                break
    return result


def main():
    args = parse_args(); config = load_config(args.config); paths = stage6_paths(config, args.debug); require_contracts(paths, config)
    if args.debug:
        gap, short_max, long_max = configured_session(config, True, paths["output_root"])
        gaps = (gap,)
        session_path = paths["output_root"] / "audits" / "session_definition.json"
    else:
        gaps = session_gap_candidates(config)
        short = config["user_tower"]["short_session"]
        short_max = None if short.get("max_events") is None else int(short["max_events"])
        long_value = config["user_tower"]["long_history"].get("max_events")
        long_max = None if long_value is None else int(long_value)
        session_path = paths["output_root"] / "audits" / "session_gap_candidate_statistics.json"
    time_path = paths["output_root"] / "audits" / "time_normalization.json"
    guard_outputs([session_path, time_path], args.overwrite)
    maximum = args.max_samples or (int(config["debug"]["max_train_samples"]) if args.debug else None)
    store = FeatureStore(paths["stage4_root"])
    sample_path = paths["stage3_root"] / "samples" / "train_samples.parquet"
    dataset = Stage6ParquetDataset(sample_path, store, maximum, int(config["scan_batch_size"]))
    stats = fit_time_normalization(dataset)
    save_json({
        **stats.to_json(),
        "full_train_scan": maximum is None,
        "max_samples": maximum,
    }, time_path, args.overwrite)
    candidate_stats = {
        gap: {"count": 0, "short": {}, "long": {}, "empty_long": 0,
              "single_short": 0, "short_trunc": 0, "long_trunc": 0}
        for gap in gaps
    }
    for item in Stage6ParquetDataset(sample_path, store, maximum, int(config["scan_batch_size"])):
        for gap in gaps:
            split = split_long_short_session(item["hist_timestamp"], gap, short_max, long_max)
            values = candidate_stats[gap]; values["count"] += 1
            short_length, long_length = len(split.short_indices), len(split.long_indices)
            values["short"][short_length] = values["short"].get(short_length, 0) + 1
            values["long"][long_length] = values["long"].get(long_length, 0) + 1
            values["empty_long"] += int(long_length == 0); values["single_short"] += int(short_length == 1)
            values["short_trunc"] += int(split.short_truncated); values["long_trunc"] += int(split.long_truncated)
    reports = []
    for gap in gaps:
        values = candidate_stats[gap]; count = values["count"]
        reports.append({
            "session_gap_seconds": gap, "sample_count": count,
            "short_session_length": quantiles(values["short"], count),
            "long_history_length": quantiles(values["long"], count),
            "empty_long_ratio": values["empty_long"] / count if count else 0.0,
            "single_event_short_ratio": values["single_short"] / count if count else 0.0,
            "short_truncation_ratio": values["short_trunc"] / count if count else 0.0,
            "long_truncation_ratio": values["long_trunc"] / count if count else 0.0,
        })
    payload = {
        "candidate_gap_seconds": list(gaps), "train_only_statistics": True,
        "full_train_scan": maximum is None, "max_samples": maximum,
        "candidates": reports, "debug_results_must_not_be_used_for_conclusions": bool(args.debug),
    }
    if args.debug:
        payload.update({"selected_session_gap_seconds": gaps[0], "source": "debug_provisional_not_for_selection"})
    else:
        payload.update({"selected_session_gap_seconds": None, "selection_pending_validation_recall100": True})
    save_json(payload, session_path, args.overwrite)


if __name__ == "__main__": main()
