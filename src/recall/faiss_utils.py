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
    retrieved_rows: Sequence[int],
    indexed_item_rids: np.ndarray,
    history_rids: Sequence[int],
    max_k: int,
    exclude_history_items: bool = False,
) -> list[int]:
    history = set(map(int, history_rids)) if exclude_history_items else set()
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


def search_nonzero_queries(index, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Search only nonzero queries; return -1 rows for zero-vector samples."""

    values = np.ascontiguousarray(queries, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("queries must be a matrix")
    if k <= 0:
        raise ValueError("k must be positive")
    nonzero = np.linalg.norm(values, axis=1) > 0
    retrieved = np.full((values.shape[0], int(k)), -1, dtype=np.int64)
    if np.any(nonzero):
        _, valid_rows = index.search(values[nonzero], int(k))
        retrieved[nonzero] = valid_rows
    return retrieved, nonzero


def hnsw_retrieval_recall(
    approximate_rows: np.ndarray,
    exact_rows: np.ndarray,
    ks: Sequence[int],
) -> dict[str, dict[str, float | int]]:
    """Measure ANN Top-K set recall against exact Inner Product Top-K."""

    approximate = np.asarray(approximate_rows)
    exact = np.asarray(exact_rows)
    if approximate.shape != exact.shape or approximate.ndim != 2:
        raise ValueError("approximate and exact row matrices must have equal 2-D shape")
    output: dict[str, dict[str, float | int]] = {}
    for k in sorted(set(map(int, ks))):
        if k <= 0 or k > approximate.shape[1]:
            raise ValueError("audit K is outside the retrieved width")
        values = np.asarray(
            [len(set(map(int, left[:k])) & set(map(int, right[:k]))) / k for left, right in zip(approximate, exact)],
            dtype=np.float64,
        )
        output[f"@{k}"] = {
            "query_count": int(values.size),
            "mean_recall": float(values.mean()) if values.size else 0.0,
            "p05_recall": float(np.quantile(values, 0.05)) if values.size else 0.0,
            "median_recall": float(np.median(values)) if values.size else 0.0,
            "minimum_recall": float(values.min()) if values.size else 0.0,
        }
    return output
