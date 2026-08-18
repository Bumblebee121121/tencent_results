"""Stage 4.6: build an aligned CSR/memmap user sequence store."""

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

from src.features.id_semantics import ACTION_TOKENS
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


def selected_batches(dataset: ds.Dataset, batch_size: int, max_users: int | None):
    processed = 0
    for batch in dataset.scanner(columns=["user_id", "seq"], batch_size=batch_size).to_batches():
        if max_users is not None:
            remaining = max_users - processed
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
        processed += batch.num_rows
        yield batch


def main() -> None:
    args = parse_args()
    config = load_stage4_config(args.config)
    data_root, stage3_root, output_root, log_root = stage4_paths(config, args.debug)
    logger = configure_logging(log_root, "stage4_6_build_sequence_store", args.debug)
    timer = Timer()
    manifests = require_stage3_contracts(stage3_root, str(config["stage3_protocol_version"]))
    max_users = args.max_users
    if args.debug and max_users is None:
        debug_limit = manifests["samples"].get("processed_user_count")
        max_users = None if debug_limit is None else int(debug_limit)
    if max_users is not None and max_users <= 0:
        raise ValueError("--max-users must be positive")

    seq_path = data_root / "seq"
    require_paths([seq_path])
    output_dir = output_root / "feature_store"
    offsets_path = output_dir / "user_seq_offsets.npy"
    item_path = output_dir / "seq_item_rid.npy"
    action_path = output_dir / "seq_action_token.npy"
    timestamp_path = output_dir / "seq_timestamp.npy"
    manifest_path = output_root / "manifests" / "sequence_store_manifest.json"
    guard_outputs(
        [offsets_path, item_path, action_path, timestamp_path, manifest_path],
        args.overwrite,
    )
    dataset = ds.dataset(seq_path, format="parquet")
    batch_size = int(config["scan_batch_size"])

    lengths_by_user: dict[int, int] = {}
    processed_users = 0
    max_user_id = 0
    for batch in selected_batches(dataset, batch_size, max_users):
        user_ids = batch.column(0).to_pylist()
        lengths = pc.list_value_length(batch.column(1)).to_pylist()
        for user_value, length_value in zip(user_ids, lengths):
            user_id = int(user_value)
            if user_id in lengths_by_user:
                raise ValueError(f"duplicate sequence row for user_id={user_id}")
            lengths_by_user[user_id] = int(length_value)
            max_user_id = max(max_user_id, user_id)
        processed_users += batch.num_rows

    user_lengths = np.zeros(max_user_id + 1, dtype=np.int64)
    for user_id, length in lengths_by_user.items():
        user_lengths[user_id] = length
    offsets = np.zeros(max_user_id + 2, dtype=np.int64)
    np.cumsum(user_lengths, out=offsets[1 : max_user_id + 2])
    event_count = int(offsets[-1])
    np.save(offsets_path, offsets, allow_pickle=False)
    item_store = np.lib.format.open_memmap(item_path, mode="w+", dtype=np.int32, shape=(event_count,))
    action_store = np.lib.format.open_memmap(action_path, mode="w+", dtype=np.int8, shape=(event_count,))
    timestamp_store = np.lib.format.open_memmap(timestamp_path, mode="w+", dtype=np.int64, shape=(event_count,))

    written_users: set[int] = set()
    unknown_action_count = 0
    for batch_number, batch in enumerate(selected_batches(dataset, batch_size, max_users), start=1):
        user_ids = [int(value) for value in batch.column(0).to_pylist()]
        lengths = [int(value) for value in pc.list_value_length(batch.column(1)).to_pylist()]
        events = pc.list_flatten(batch.column(1))
        item_array = events.field("item_id")
        timestamp_array = events.field("timestamp")
        action_array = events.field("action_type")
        if item_array.null_count or timestamp_array.null_count:
            raise ValueError("sequence item_id/timestamp cannot be null")
        items = np.asarray(item_array.to_numpy(), dtype=np.int64)
        if items.size and (items.min() <= 0 or items.max() > np.iinfo(np.int32).max):
            raise ValueError("sequence item RID cannot be represented as int32")
        timestamps = np.asarray(timestamp_array.to_numpy(), dtype=np.int64)
        raw_actions = np.asarray(pc.fill_null(action_array, -1).to_numpy(), dtype=np.int32)
        actions = np.full(raw_actions.shape, ACTION_TOKENS["unknown"], dtype=np.int8)
        actions[raw_actions == 0] = ACTION_TOKENS["exposure"]
        actions[raw_actions == 1] = ACTION_TOKENS["click"]
        unknown_action_count += int(np.count_nonzero((raw_actions != 0) & (raw_actions != 1)))
        cursor = 0
        for user_id, length in zip(user_ids, lengths):
            if user_id in written_users:
                raise ValueError(f"duplicate user during sequence write: {user_id}")
            start = int(offsets[user_id])
            stop = start + length
            source_stop = cursor + length
            user_timestamps = timestamps[cursor:source_stop]
            if length > 1 and np.any(np.diff(user_timestamps) < 0):
                raise ValueError(f"timestamps are not nondecreasing for user_id={user_id}")
            item_store[start:stop] = items[cursor:source_stop]
            action_store[start:stop] = actions[cursor:source_stop]
            timestamp_store[start:stop] = user_timestamps
            cursor = source_stop
            written_users.add(user_id)
        if cursor != len(items):
            raise AssertionError("flattened sequence length does not match row lengths")
        if batch_number % 20 == 0:
            logger.info("written_users=%d", len(written_users))
    item_store.flush()
    action_store.flush()
    timestamp_store.flush()
    if len(written_users) != processed_users:
        raise AssertionError("not every selected user was written")
    manifest = {
        "stage": "4.6",
        "schema_version": 1,
        "feature_protocol_version": config["feature_protocol_version"],
        "debug": bool(args.debug),
        "max_users": max_users,
        "user_count": processed_users,
        "maximum_user_id": max_user_id,
        "sequence_event_count": event_count,
        "unknown_action_count": unknown_action_count,
        "sequence_store_backend": config["sequence_store"]["backend"],
        "materialize_history_per_sample": False,
        "aligned_fields": ["seq_item_rid", "seq_action_token", "seq_timestamp"],
        "elapsed_seconds": timer.elapsed_seconds,
    }
    save_json(manifest, manifest_path, args.overwrite)
    logger.info("wrote sequence events=%d users=%d elapsed_seconds=%.2f", event_count, processed_users, timer.elapsed_seconds)


if __name__ == "__main__":
    main()
