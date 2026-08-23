"""Stage 5.1: build weighted, windowed ItemCF neighbor tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recall.data import Stage5SequenceStore
from src.recall.itemcf import build_partitioned_itemcf
from src.recall.runtime import (
    Timer, add_common_arguments, configure_logging, guard_outputs, load_config,
    require_contracts, save_json, stage5_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-users", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, stage3_root, stage4_root, output_root, log_root = stage5_paths(config, args.debug)
    logger = configure_logging(log_root, "stage5_1_build_itemcf")
    contracts = require_contracts(stage3_root, stage4_root, config)
    timer = Timer()
    section = config["itemcf"]
    max_users = args.max_users
    buffer_size = int(section["pair_buffer_size"])
    if args.debug:
        max_users = max_users or int(config["debug"]["max_users"])
        buffer_size = int(config["debug"]["pair_buffer_size"])
    if max_users is not None and max_users <= 0:
        raise ValueError("--max-users must be positive")

    itemcf_root = output_root / "itemcf"
    neighbors_path = itemcf_root / "item_neighbors.parquet"
    manifest_path = itemcf_root / "manifest.json"
    shard_root = itemcf_root / "neighbor_shards"
    guard_outputs([neighbors_path, manifest_path, shard_root], args.overwrite)
    table = ds.dataset(stage3_root / "candidates" / "eval_candidates.parquet", format="parquet").to_table(
        columns=["item_oid", "item_rid"]
    )
    candidate_oid_by_rid = {
        int(rid): int(oid)
        for oid, rid in zip(table.column("item_oid").to_pylist(), table.column("item_rid").to_pylist())
        if rid is not None
    }
    store = Stage5SequenceStore(stage4_root)
    stats = build_partitioned_itemcf(
        store.offsets, store.item_rids, store.actions, store.timestamps,
        int(contracts["splits"]["train_raw_event_cutoff_exclusive"]), candidate_oid_by_rid,
        neighbors_path, shard_root, int(section["fit_window"]),
        int(section["neighbor_topn"]), int(section["pair_partitions"]), buffer_size,
        args.overwrite, max_users=max_users, logger=logger,
    )
    manifest = {
        "stage": "5.1",
        "schema_version": 1,
        "recall_protocol_version": config["recall_protocol_version"],
        "stage3_protocol_version": config["stage3_protocol_version"],
        "stage4_protocol_version": config["stage4_protocol_version"],
        "debug": bool(args.debug),
        "train_cutoff_exclusive": int(contracts["splits"]["train_raw_event_cutoff_exclusive"]),
        "fit_window": int(section["fit_window"]),
        "neighbor_topn": int(section["neighbor_topn"]),
        "pair_partitions": int(section["pair_partitions"]),
        "candidate_neighbor_count": len(candidate_oid_by_rid),
        "seed_scope": "all train-period history items, including non-candidates",
        "neighbor_scope": "Stage 3 eval candidates with a historical RID",
        "similarity_variants": ["equal", "click3"],
        "unknown_action_weight": 0.0,
        **stats,
        "elapsed_seconds": timer.elapsed_seconds,
    }
    save_json(manifest, manifest_path, args.overwrite)
    logger.info("wrote neighbor_rows=%d elapsed_seconds=%.2f", stats["neighbor_rows"], timer.elapsed_seconds)


if __name__ == "__main__":
    main()
