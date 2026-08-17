"""Deterministic global temporal split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TemporalCutoffs:
    """Exclusive timestamp boundaries: train < val_start <= val < test_start."""

    validation_start: int
    test_start: int


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = np.asarray([train_ratio, val_ratio, test_ratio], dtype=np.float64)
    if np.any(ratios <= 0) or not np.isclose(float(ratios.sum()), 1.0):
        raise ValueError("train/val/test ratios must be positive and sum to 1")


def choose_temporal_cutoffs(
    timestamps: Iterable[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> TemporalCutoffs:
    """Choose global ratio cutoffs while keeping equal timestamps together."""

    validate_ratios(train_ratio, val_ratio, test_ratio)
    values = np.asarray(list(timestamps) if not isinstance(timestamps, np.ndarray) else timestamps)
    values = values.astype(np.int64, copy=False)
    if values.size < 3:
        raise ValueError("at least three samples are required for a three-way split")
    unique_timestamps, counts = np.unique(values, return_counts=True)
    if unique_timestamps.size < 3:
        raise ValueError(
            "timestamp ties leave fewer than three non-empty temporal windows; "
            "use more data or inspect timestamp granularity"
        )
    # cumulative_before[j] is the number of samples assigned before timestamp j.
    cumulative_before = np.r_[0, np.cumsum(counts[:-1], dtype=np.int64)]
    validation_candidates = np.arange(1, unique_timestamps.size - 1)
    validation_index = int(
        validation_candidates[
            np.argmin(
                np.abs(cumulative_before[validation_candidates] - values.size * train_ratio)
            )
        ]
    )
    test_candidates = np.arange(validation_index + 1, unique_timestamps.size)
    test_index = int(
        test_candidates[
            np.argmin(
                np.abs(
                    cumulative_before[test_candidates]
                    - values.size * (train_ratio + val_ratio)
                )
            )
        ]
    )
    validation_start = int(unique_timestamps[validation_index])
    test_start = int(unique_timestamps[test_index])
    groups = assign_splits(values, TemporalCutoffs(validation_start, test_start))
    if set(groups.tolist()) != {"train", "val", "test"}:
        raise ValueError("timestamp boundaries produced an empty split")
    return TemporalCutoffs(validation_start, test_start)


def assign_splits(timestamps: np.ndarray, cutoffs: TemporalCutoffs) -> np.ndarray:
    values = np.asarray(timestamps, dtype=np.int64)
    result = np.full(values.shape, "val", dtype="<U5")
    result[values < cutoffs.validation_start] = "train"
    result[values >= cutoffs.test_start] = "test"
    return result


def earliest_sample_ids(
    sample_ids: np.ndarray,
    user_ids: np.ndarray,
    timestamps: np.ndarray,
) -> np.ndarray:
    """Return one deterministic earliest sample per user."""

    sample_ids = np.asarray(sample_ids, dtype=np.int64)
    user_ids = np.asarray(user_ids, dtype=np.int64)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if not (sample_ids.size == user_ids.size == timestamps.size):
        raise ValueError("sample_ids, user_ids and timestamps must have equal length")
    if sample_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((sample_ids, timestamps, user_ids))
    sorted_users = user_ids[order]
    first = np.r_[True, sorted_users[1:] != sorted_users[:-1]]
    result = sample_ids[order[first]]
    result.sort()
    return result


def first_n_sample_ids_per_user(
    sample_ids: np.ndarray,
    user_ids: np.ndarray,
    timestamps: np.ndarray,
    limit: int,
) -> np.ndarray:
    if limit <= 0:
        raise ValueError("per-user target limit must be positive")
    order = np.lexsort((sample_ids, timestamps, user_ids))
    sorted_users = np.asarray(user_ids, dtype=np.int64)[order]
    positions = np.empty(order.size, dtype=np.int64)
    group_start = 0
    for index in range(order.size):
        if index == 0 or sorted_users[index] != sorted_users[index - 1]:
            group_start = index
        positions[index] = index - group_start
    result = np.asarray(sample_ids, dtype=np.int64)[order[positions < limit]]
    result.sort()
    return result


def contains_sorted(sorted_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_values = np.asarray(sorted_values, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)
    positions = np.searchsorted(sorted_values, values)
    found = positions < sorted_values.size
    valid = np.flatnonzero(found)
    found[valid] &= sorted_values[positions[valid]] == values[valid]
    return found
