"""Model-neutral dynamic time feature utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def recency_features(
    history_timestamps: Sequence[int] | np.ndarray,
    target_timestamp: int,
) -> tuple[np.ndarray, np.ndarray]:
    history = np.asarray(history_timestamps, dtype=np.int64)
    recency = int(target_timestamp) - history
    if np.any(recency <= 0):
        raise ValueError("every history timestamp must be strictly earlier than target")
    return recency, np.log1p(recency.astype(np.float64)).astype(np.float32)


def inter_event_gap_features(
    history_timestamps: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.asarray(history_timestamps, dtype=np.int64)
    gaps = np.zeros(timestamps.shape, dtype=np.int64)
    if timestamps.size > 1:
        gaps[1:] = np.diff(timestamps)
    if np.any(gaps < 0):
        raise ValueError("history timestamps must be nondecreasing")
    first_event_mask = np.zeros(timestamps.shape, dtype=np.bool_)
    if timestamps.size:
        first_event_mask[0] = True
    return gaps, np.log1p(gaps.astype(np.float64)).astype(np.float32), first_event_mask


def bucketize_time(
    seconds: Sequence[int] | np.ndarray,
    boundaries: Sequence[int] | np.ndarray,
) -> np.ndarray:
    values = np.asarray(seconds)
    edges = np.asarray(boundaries)
    if edges.ndim != 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("time bucket boundaries must be strictly increasing")
    if np.any(values < 0):
        raise ValueError("time values cannot be negative")
    return np.searchsorted(edges, values, side="right").astype(np.int32)
