"""Compact sample-index construction helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa

from .attribution import AttributionStats, attribute_sequence


SAMPLE_SCHEMA = pa.schema(
    [
        ("sample_id", pa.int64()),
        ("user_id", pa.int64()),
        ("target_item_rid", pa.int64()),
        ("target_item_oid", pa.int64()),
        ("target_exposure_timestamp", pa.int64()),
        ("target_click_timestamp", pa.int64()),
        ("target_exposure_position", pa.int32()),
        ("target_click_position", pa.int32()),
        ("history_end_position", pa.int32()),
        ("history_length", pa.int32()),
        ("attribution_gap", pa.int64()),
    ]
)


def make_rid_to_oid(item_mapping: Mapping[int, int]) -> np.ndarray:
    """Invert the official OID->RID map without inventing missing RIDs."""

    if not item_mapping:
        raise ValueError("item mapping is empty")
    max_rid = max(int(rid) for rid in item_mapping.values())
    result = np.full(max_rid + 1, -1, dtype=np.int64)
    for oid, rid_value in item_mapping.items():
        rid = int(rid_value)
        if rid <= 0 or result[rid] != -1:
            raise ValueError(f"invalid or duplicate item RID: {rid}")
        result[rid] = int(oid)
    return result


def oid_lookup_from_array(rid_to_oid: np.ndarray):
    def lookup(rid: int) -> int:
        if rid <= 0 or rid >= rid_to_oid.size or rid_to_oid[rid] < 0:
            raise KeyError(f"history RID {rid} has no official OID mapping")
        return int(rid_to_oid[rid])

    return lookup


def build_samples_for_users(
    user_ids: Iterable[int],
    sequences: Iterable[Sequence[Mapping[str, Any]]],
    rid_to_oid: np.ndarray,
    first_sample_id: int = 0,
) -> tuple[list[dict[str, int]], AttributionStats]:
    """Build deterministic compact samples for a batch of complete user rows."""

    lookup = oid_lookup_from_array(rid_to_oid)
    records: list[dict[str, int]] = []
    stats = AttributionStats()
    next_sample_id = int(first_sample_id)
    for user_id, events in zip(user_ids, sequences):
        user_samples, user_stats = attribute_sequence(int(user_id), events, lookup)
        for sample in user_samples:
            sample["sample_id"] = next_sample_id
            next_sample_id += 1
            records.append(sample)
        stats.merge(user_stats)
    return records, stats
