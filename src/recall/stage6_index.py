"""Candidate alignment and vector audits for Stage 6 indexes."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def audit_item_embeddings(embeddings: np.ndarray, groups: Sequence[str]) -> dict[str, int]:
    values = np.asarray(embeddings)
    if values.ndim != 2 or len(groups) != values.shape[0]:
        raise ValueError("embedding rows and candidate groups must align")
    finite = np.isfinite(values).all(axis=1)
    zero = np.linalg.norm(np.where(np.isfinite(values), values, 0.0), axis=1) == 0
    seen = np.asarray([group != "Unseen" for group in groups])
    rounded = np.round(values[finite], decimals=7)
    duplicate = int(rounded.shape[0] - np.unique(rounded, axis=0).shape[0]) if rounded.size else 0
    return {
        "indexed_candidate_count": int(values.shape[0]),
        "seen_indexed_count": int(seen.sum()),
        "unseen_indexed_count": int((~seen).sum()),
        "zero_vector_count": int(zero.sum()),
        "nan_inf_count": int((~finite).sum()),
        "duplicate_vector_count": duplicate,
    }

