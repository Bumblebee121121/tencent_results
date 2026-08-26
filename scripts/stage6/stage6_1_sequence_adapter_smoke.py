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
from src.recall.stage6_runtime import add_common_arguments, configured_session, guard_outputs, load_config, require_contracts, save_json, stage6_paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__); add_common_arguments(parser); parser.add_argument("--max-samples", type=int); return parser.parse_args()


def quantiles(values):
    return {name: float(np.quantile(values, q)) if values else 0.0 for name, q in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99))}


def main():
    args = parse_args(); config = load_config(args.config); paths = stage6_paths(config, args.debug); require_contracts(paths, config)
    gap, short_max, long_max = configured_session(config, args.debug)
    session_path = paths["output_root"] / "audits" / "session_definition.json"
    time_path = paths["output_root"] / "audits" / "time_normalization.json"
    guard_outputs([session_path, time_path], args.overwrite)
    maximum = args.max_samples or (int(config["debug"]["max_train_samples"]) if args.debug else None)
    store = FeatureStore(paths["stage4_root"])
    dataset = Stage6ParquetDataset(paths["stage3_root"] / "samples" / "train_samples.parquet", store, maximum, int(config["scan_batch_size"]))
    items = list(dataset)
    stats = fit_time_normalization(items); save_json(stats.to_json(), time_path, args.overwrite)
    short_lengths=[]; long_lengths=[]; short_trunc=long_trunc=0
    for item in items:
        split = split_long_short_session(item["hist_timestamp"], gap, short_max, long_max)
        short_lengths.append(len(split.short_indices)); long_lengths.append(len(split.long_indices))
        short_trunc += int(split.short_truncated); long_trunc += int(split.long_truncated)
    count = len(items)
    save_json({
        "selected_session_gap_seconds": gap,
        "source": "debug_provisional_not_for_selection" if args.debug else "validation_only_selection_frozen_in_config",
        "candidate_gap_seconds": config["user_tower"]["short_session"]["candidate_gap_seconds"],
        "sample_count": count, "short_session_length": quantiles(short_lengths), "long_history_length": quantiles(long_lengths),
        "empty_long_ratio": sum(v == 0 for v in long_lengths) / count if count else 0.0,
        "single_event_short_ratio": sum(v == 1 for v in short_lengths) / count if count else 0.0,
        "short_truncation_ratio": short_trunc / count if count else 0.0,
        "long_truncation_ratio": long_trunc / count if count else 0.0,
        "debug_results_must_not_be_used_for_conclusions": bool(args.debug),
    }, session_path, args.overwrite)


if __name__ == "__main__": main()

