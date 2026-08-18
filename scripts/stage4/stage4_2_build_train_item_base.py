"""Stage 4.2: build train-only item counts and model item-token mapping."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.stage3_runtime import load_item_mapping
from src.features.id_semantics import ACTION_TOKENS, build_rid_to_model_item_token
from src.features.item_feature_store import verify_stage3_counts
from src.features.runtime import (
    Timer,
    add_common_arguments,
    configure_logging,
    guard_outputs,
    load_stage4_config,
    require_paths,
    require_stage3_contracts,
    save_json,
    stage4_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-users", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_stage4_config(args.config)
    data_root, stage3_root, output_root, log_root = stage4_paths(config, args.debug)
    logger = configure_logging(log_root, "stage4_2_build_train_item_base", args.debug)
    timer = Timer()
    protocol = str(config["stage3_protocol_version"])
    manifests = require_stage3_contracts(stage3_root, protocol)
    cutoff = int(manifests["splits"]["train_raw_event_cutoff_exclusive"])
    stage3_limit = manifests["strength"].get("max_users")
    max_users = args.max_users
    if args.debug and max_users is None and stage3_limit is not None:
        max_users = int(stage3_limit)
    if max_users is not None and max_users <= 0:
        raise ValueError("--max-users must be positive")

    seq_path = data_root / "seq"
    indexer_path = data_root / "indexer.pkl"
    stage3_counts_path = stage3_root / "item_strength" / "item_train_counts.parquet"
    require_paths([seq_path, indexer_path, stage3_counts_path])
    mapping_dir = output_root / "mappings"
    count_path = mapping_dir / "train_item_count_by_rid.npy"
    token_path = mapping_dir / "rid_to_model_item_token.npy"
    action_path = mapping_dir / "action_token_map.json"
    manifest_path = output_root / "manifests" / "train_item_base_manifest.json"
    guard_outputs([count_path, token_path, action_path, manifest_path], args.overwrite)
    logger.info("input seq=%s stage3_counts=%s cutoff=%d", seq_path, stage3_counts_path, cutoff)

    oid_to_rid = load_item_mapping(indexer_path)
    max_rid = max(int(value) for value in oid_to_rid.values())
    del oid_to_rid
    counts = np.zeros(max_rid + 1, dtype=np.int64)
    dataset = ds.dataset(seq_path, format="parquet")
    processed_users = 0
    counted_events = 0
    scanner = dataset.scanner(columns=["seq"], batch_size=int(config["scan_batch_size"]))
    for batch_number, batch in enumerate(scanner.to_batches(), start=1):
        if max_users is not None:
            remaining = max_users - processed_users
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
        events = pc.list_flatten(batch.column(0))
        if events.field("item_id").null_count or events.field("timestamp").null_count:
            raise ValueError("sequence item_id/timestamp cannot be null")
        item_rids = np.asarray(events.field("item_id").to_numpy(), dtype=np.int64)
        timestamps = np.asarray(events.field("timestamp").to_numpy(), dtype=np.int64)
        selected = item_rids[timestamps < cutoff]
        if selected.size and (selected.min() <= 0 or selected.max() > max_rid):
            raise ValueError("sequence contains item RID outside indexer range")
        counts += np.bincount(selected, minlength=counts.size)
        processed_users += batch.num_rows
        counted_events += int(selected.size)
        if batch_number % 20 == 0:
            logger.info("processed_users=%d counted_train_events=%d", processed_users, counted_events)

    stage3_dataset = ds.dataset(stage3_counts_path, format="parquet")
    checked_candidates = 0
    for batch in stage3_dataset.scanner(
        columns=["item_rid", "train_event_count"], batch_size=65536
    ).to_batches():
        verify_stage3_counts(counts, batch.column(0).to_pylist(), batch.column(1).to_pylist())
        checked_candidates += batch.num_rows

    max_count = int(counts.max(initial=0))
    count_dtype = np.int32 if max_count <= np.iinfo(np.int32).max else np.int64
    stored_counts = counts.astype(count_dtype)
    model_tokens = build_rid_to_model_item_token(stored_counts)
    np.save(count_path, stored_counts, allow_pickle=False)
    np.save(token_path, model_tokens, allow_pickle=False)
    save_json(
        {
            "raw_to_token": {"0": ACTION_TOKENS["exposure"], "1": ACTION_TOKENS["click"], "null_or_other": ACTION_TOKENS["unknown"]},
            "padding_token": ACTION_TOKENS["pad"],
        },
        action_path,
        args.overwrite,
    )
    manifest = {
        "stage": "4.2",
        "schema_version": 1,
        "feature_protocol_version": config["feature_protocol_version"],
        "stage3_protocol_version": protocol,
        "debug": bool(args.debug),
        "max_users": max_users,
        "stage3_train_cutoff": cutoff,
        "p50_train": float(manifests["strength"]["p50_train"]),
        "p90_train": float(manifests["strength"]["p90_train"]),
        "processed_user_count": processed_users,
        "counted_train_event_count": counted_events,
        "item_rid_count": max_rid,
        "train_seen_item_count": int(np.count_nonzero(stored_counts)),
        "maximum_train_event_count": max_count,
        "count_dtype": np.dtype(count_dtype).name,
        "model_item_token_rule": "train-seen RID -> RID+1; train-unseen/null -> UNK=1; PAD=0",
        "stage3_candidate_count_consistency_checked": checked_candidates,
        "stage3_count_consistency_passed": True,
        "elapsed_seconds": timer.elapsed_seconds,
    }
    save_json(manifest, manifest_path, args.overwrite)
    logger.info("wrote counts=%s tokens=%s elapsed_seconds=%.2f", count_path, token_path, timer.elapsed_seconds)


if __name__ == "__main__":
    main()
