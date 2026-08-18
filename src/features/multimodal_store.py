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
