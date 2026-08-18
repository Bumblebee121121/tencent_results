"""Stage 3.6: write the shared retrieval evaluation protocol manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.stage3_runtime import (
    common_parser_arguments,
    load_config,
    runtime_paths,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common_parser_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, output_root = runtime_paths(config, args.data_root, args.output_root, args.debug)
    output_path = output_root / "evaluation" / "evaluation_protocol.json"
    protocol = {
        "stage": "3.6",
        "schema_version": 2,
        "protocol_version": str(config.get("protocol_version", "click_target_prefix_v2")),
        "debug": bool(args.debug),
        "primary_evaluation_files": {
            "validation": "samples/val_primary.parquet",
            "test": "samples/test_primary.parquet",
        },
        "ground_truth_per_record": 1,
        "target_field": "target_item_oid",
        "target_time_field": "target_timestamp",
        "sample_target_definition": "action_type == 1 interaction",
        "history_timestamp_invariant": "history timestamp < target_timestamp",
        "empty_history_policy": "excluded from formal Train/Validation/Test samples",
        "candidate_pool": "candidates/eval_candidates.parquet",
        "item_strength": "item_strength/item_train_counts.parquet",
        "metrics_implementation": "src/evaluation/retrieval_metrics.py",
        "recall_ks": [10, 50, 100, 500],
        "hitrate_ks": [10, 50, 100, 500],
        "ndcg_ks": [10, 50, 100],
        "groups": ["Overall", "Head", "Mid", "Tail", "Unseen"],
        "definitions": {
            "Recall@K": "mean(ground_truth appears in top K)",
            "HitRate@K": "equal to Recall@K for one ground truth",
            "NDCG@K": "1/log2(rank+1) when rank <= K, otherwise 0",
        },
        "channel_analysis": [
            "intersection_count",
            "union_count",
            "jaccard",
            "incremental_hits",
            "incremental_recall",
        ],
    }
    save_json(protocol, output_path, args.overwrite)
    print(f"wrote: {output_path.resolve()}")


if __name__ == "__main__":
    main()
