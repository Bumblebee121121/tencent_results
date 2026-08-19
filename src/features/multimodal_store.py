"""Multimodal validation and missing-mask semantics."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def validate_mm_vector(value: Sequence[float], expected_dim: int = 32) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or vector.size != expected_dim:
        raise ValueError(
            f"multimodal vector must have dimension {expected_dim}, found {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("multimodal vector contains NaN or Inf")
    return vector


def optional_mm_vector(
    value: Sequence[float] | None,
    expected_dim: int = 32,
) -> tuple[np.ndarray, bool]:
    if value is None:
        return np.zeros(expected_dim, dtype=np.float32), False
    return validate_mm_vector(value, expected_dim), True


def candidate_row_index(
    item_oids: Sequence[int],
    retrieval_ids: Sequence[int],
) -> dict[int, int]:
    """Index candidates by physical row order, never by retrieval_id value."""

    if len(item_oids) != len(retrieval_ids):
        raise ValueError("candidate OID and retrieval_id vectors must have equal length")
    oid_index: dict[int, int] = {}
    seen_retrieval_ids: set[int] = set()
    for row_index, (oid_value, retrieval_value) in enumerate(
        zip(item_oids, retrieval_ids)
    ):
        oid = int(oid_value)
        retrieval_id = int(retrieval_value)
        if oid in oid_index:
            raise ValueError(f"eval candidate contains duplicate item_oid: {oid}")
        if retrieval_id in seen_retrieval_ids:
            raise ValueError(
                f"eval candidate contains duplicate retrieval_id: {retrieval_id}"
            )
        oid_index[oid] = row_index
        seen_retrieval_ids.add(retrieval_id)
    return oid_index
