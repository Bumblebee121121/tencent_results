"""Train-only item history strength definitions."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


STRENGTH_GROUPS = ("Head", "Mid", "Tail", "Unseen")


def count_events_before_cutoff(
    sequences: Iterable[Sequence[Mapping[str, Any]]],
    cutoff_exclusive: int,
    max_item_rid: int,
) -> np.ndarray:
    """Reference implementation used by tests and small inputs."""

    counts = np.zeros(max_item_rid + 1, dtype=np.int64)
    for events in sequences:
        for event in events:
            timestamp = event.get("timestamp")
            item_rid = event.get("item_id")
            if timestamp is None or item_rid is None:
                raise ValueError("item_id and timestamp must not be null")
            rid = int(item_rid)
            if rid <= 0 or rid > max_item_rid:
                raise ValueError(f"item RID out of range: {rid}")
            if int(timestamp) < cutoff_exclusive:
                counts[rid] += 1
    return counts


def strength_thresholds(counts: np.ndarray) -> tuple[float, float]:
    seen = np.asarray(counts, dtype=np.int64)
    seen = seen[seen > 0]
    if seen.size == 0:
        raise ValueError("no train events exist before the cutoff")
    p50, p90 = np.quantile(seen, [0.5, 0.9])
    return float(p50), float(p90)


def classify_strength(count: int, p50: float, p90: float) -> str:
    if count <= 0:
        return "Unseen"
    if count <= p50:
        return "Tail"
    if count <= p90:
        return "Mid"
    return "Head"
