"""Stage 5.2: evaluate the three fixed ItemCF experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.item_strength import classify_strength
from src.recall.data import Stage5SequenceStore
from src.recall.evaluation import metrics_from_ranks
from src.recall.itemcf import NeighborShardLookup, retrieve_itemcf
from src.recall.runtime import (
    Timer, add_common_arguments, configure_logging, guard_outputs, load_config,
    require_contracts, require_paths, save_json, stage5_paths,
)


RANK_SCHEMA = pa.schema(
    [
        ("sample_id", pa.int64()), ("target_item_oid", pa.int64()),
        ("target_item_rid", pa.int64()), ("target_strength_group", pa.string()),
        ("target_rank", pa.int32()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def evaluate_split(path, output_path, store, lookup, experiment, recall_ks, ndcg_ks, max_samples, p50, p90):
    writer = pq.ParquetWriter(output_path, RANK_SCHEMA, compression="snappy")
    ranks: list[int | None] = []
    groups: list[str] = []
    processed = 0
    try:
        scanner = ds.dataset(path, format="parquet").scanner(batch_size=4096)
        for batch in scanner.to_batches():
            rows = []
            for row in batch.to_pylist():
                if max_samples is not None and processed >= max_samples:
                    break
                history = store.history(row)
                ranking = retrieve_itemcf(
                    history.item_rid, history.action_token, lookup, max(recall_ks),
                    float(experiment["exposure_weight"]), float(experiment["click_weight"]),
                    float(experiment["unknown_weight"]), experiment.get("history_limit"),
                )
                target_rid = int(row["target_item_rid"])
                rank = ranking.index(target_rid) + 1 if target_rid in ranking else None
                count = int(store.train_counts[target_rid]) if 0 < target_rid < store.train_counts.size else 0
                group = classify_strength(count, p50, p90)
                rows.append(
                    {"sample_id": int(row["sample_id"]), "target_item_oid": int(row["target_item_oid"]),
                     "target_item_rid": target_rid, "target_strength_group": group, "target_rank": rank}
                )
                ranks.append(rank)
                groups.append(group)
                processed += 1
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=RANK_SCHEMA))
            if max_samples is not None and processed >= max_samples:
                break
    finally:
        writer.close()
    return metrics_from_ranks(ranks, groups, recall_ks, ndcg_ks)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, stage3_root, stage4_root, output_root, log_root = stage5_paths(config, args.debug)
    logger = configure_logging(log_root, "stage5_2_evaluate_itemcf")
    contracts = require_contracts(stage3_root, stage4_root, config)
    timer = Timer()
    itemcf_root = output_root / "itemcf"
    require_paths([itemcf_root / "manifest.json", itemcf_root / "item_neighbors.parquet"])
    with (itemcf_root / "manifest.json").open("r", encoding="utf-8") as handle:
        build_manifest = json.load(handle)
    if build_manifest.get("recall_protocol_version") != config["recall_protocol_version"]:
        raise ValueError("ItemCF build protocol mismatch")
    ranks_root = itemcf_root / "ranks"
    metrics_path = itemcf_root / "metrics.json"
    expected = [metrics_path]
    for experiment_name in config["itemcf"]["experiments"]:
        for split in ("validation", "test"):
            expected.append(ranks_root / f"{experiment_name}_{split}.parquet")
    guard_outputs(expected, args.overwrite)
    max_samples = args.max_samples
    if args.debug and max_samples is None:
        max_samples = int(config["debug"]["max_eval_samples"])
    store = Stage5SequenceStore(stage4_root)
    p50 = float(contracts["stage4"]["p50_train"])
    p90 = float(contracts["stage4"]["p90_train"])
    metrics = {}
    split_paths = {
        "validation": stage3_root / "samples" / "val_primary.parquet",
        "test": stage3_root / "samples" / "test_primary.parquet",
    }
    for name, experiment in config["itemcf"]["experiments"].items():
        similarity_variant = "equal" if name == "equal_all" else "click3"
        lookup = NeighborShardLookup(
            itemcf_root / "neighbor_shards", similarity_variant, int(config["itemcf"]["pair_partitions"])
        )
        metrics[name] = {}
        for split, path in split_paths.items():
            logger.info("evaluating experiment=%s split=%s", name, split)
            metrics[name][split] = evaluate_split(
                path, ranks_root / f"{name}_{split}.parquet", store, lookup, experiment,
                config["recall_ks"], config["ndcg_ks"], max_samples, p50, p90,
            )
    save_json(
        {"stage": "5.2", "schema_version": 1, "recall_protocol_version": config["recall_protocol_version"],
         "debug": bool(args.debug), "metrics": metrics, "elapsed_seconds": timer.elapsed_seconds},
        metrics_path, args.overwrite,
    )
    logger.info("ItemCF evaluation complete elapsed_seconds=%.2f", timer.elapsed_seconds)


if __name__ == "__main__":
    main()
