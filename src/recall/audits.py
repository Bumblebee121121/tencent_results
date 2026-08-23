"""Pre-run protocol audits for Stage 5."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from .data import SAMPLE_COLUMNS, Stage5SequenceStore


def reservoir_sample_rows(
    sample_path: Path, sample_size: int, seed: int, batch_size: int = 8192
) -> tuple[list[dict[str, object]], int]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    rng = np.random.default_rng(seed)
    reservoir: list[dict[str, object]] = []
    seen = 0
    for batch in ds.dataset(sample_path, format="parquet").scanner(
        columns=SAMPLE_COLUMNS, batch_size=batch_size
    ).to_batches():
        for row in batch.to_pylist():
            seen += 1
            if len(reservoir) < sample_size:
                reservoir.append(row)
            else:
                replacement = int(rng.integers(0, seen))
                if replacement < sample_size:
                    reservoir[replacement] = row
    return reservoir, seen


def target_is_repeated(target_item_rid: int, history_item_rids: np.ndarray) -> bool:
    return bool(np.any(np.asarray(history_item_rids, dtype=np.int64) == int(target_item_rid)))


def repeated_target_stats(
    sample_path: Path,
    store: Stage5SequenceStore,
    batch_size: int = 8192,
    max_samples: int | None = None,
) -> dict[str, int | float]:
    sample_count = 0
    repeated_count = 0
    train_seen_repeated_count = 0
    current_user: int | None = None
    current_history_end = 0
    current_seen_items: set[int] = set()
    nonmonotonic_history_rows = 0
    for batch in ds.dataset(sample_path, format="parquet").scanner(
        columns=SAMPLE_COLUMNS, batch_size=batch_size
    ).to_batches():
        for row in batch.to_pylist():
            if max_samples is not None and sample_count >= max_samples:
                break
            user_id = int(row["user_id"])
            target_rid = int(row["target_item_rid"])
            history_end = int(row["history_end_position"])
            if user_id < 0 or user_id + 1 >= store.offsets.size:
                raise IndexError(f"user_id is outside sequence store: {user_id}")
            user_start, user_stop = int(store.offsets[user_id]), int(store.offsets[user_id + 1])
            absolute_end = user_start + history_end
            if history_end < 0 or absolute_end > user_stop:
                raise IndexError("history_end_position exceeds user sequence")
            if absolute_end > user_start and int(store.timestamps[absolute_end - 1]) >= int(row["target_timestamp"]):
                raise ValueError("history includes an event at or after target timestamp")
            if user_id != current_user or history_end < current_history_end:
                if user_id == current_user:
                    nonmonotonic_history_rows += 1
                current_user = user_id
                current_history_end = 0
                current_seen_items = set()
            if history_end > current_history_end:
                current_seen_items.update(
                    map(int, store.item_rids[user_start + current_history_end : absolute_end])
                )
                current_history_end = history_end
            repeated = target_rid in current_seen_items
            repeated_count += int(repeated)
            if repeated and 0 < target_rid < store.train_counts.size and int(store.train_counts[target_rid]) > 0:
                train_seen_repeated_count += 1
            sample_count += 1
        if max_samples is not None and sample_count >= max_samples:
            break
    return {
        "sample_count": sample_count,
        "repeated_target_count": repeated_count,
        "repeated_target_ratio": repeated_count / sample_count if sample_count else 0.0,
        "train_seen_repeated_target_count": train_seen_repeated_count,
        "nonmonotonic_history_rows": nonmonotonic_history_rows,
        "scan_method": "incremental per-user history set; exact fallback reset on nonmonotonic history_end_position",
    }
