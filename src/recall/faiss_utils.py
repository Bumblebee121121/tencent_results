"""FAISS index construction and row-alignment helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def train_seen_candidate_rows(candidate_rows: Sequence[dict[str, object]], rid_to_token: np.ndarray):
    """Preserve physical candidate row order and assign a fresh FAISS row."""

    result = []
    for row in candidate_rows:
        rid = row.get("item_rid")
        if rid is None or int(rid) <= 0 or int(rid) >= rid_to_token.size:
            continue
        token = int(rid_to_token[int(rid)])
        if token <= 1:
            continue
        result.append({**row, "model_item_token": token, "faiss_row": len(result)})
    return result


def build_hnsw_ip(embeddings: np.ndarray, m: int, ef_construction: int, ef_search: int):
    import faiss

    values = np.ascontiguousarray(embeddings, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("embeddings must be a matrix")
    index = faiss.IndexHNSWFlat(values.shape[1], int(m), faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(ef_construction)
    index.hnsw.efSearch = int(ef_search)
    index.add(values)
    return index


def filter_history_from_faiss_rows(
    retrieved_rows: Sequence[int], indexed_item_rids: np.ndarray, history_rids: Sequence[int], max_k: int
) -> list[int]:
    history = set(map(int, history_rids))
    result = []
    for row in retrieved_rows:
        if int(row) < 0:
            continue
        rid = int(indexed_item_rids[int(row)])
        if rid not in history:
            result.append(rid)
            if len(result) == max_k:
                break
    return result
