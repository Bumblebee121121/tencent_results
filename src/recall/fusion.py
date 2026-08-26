"""Rank-based multi-channel recall fusion."""

from __future__ import annotations

from typing import Mapping, Sequence


def reciprocal_rank_fusion(
    first: Sequence[int], second: Sequence[int], alpha: float, c: float = 60.0, top_k: int | None = None,
) -> list[int]:
    if not 0.0 <= alpha <= 1.0 or c <= 0:
        raise ValueError("RRF requires alpha in [0,1] and c > 0")
    scores: dict[int, float] = {}
    for weight, ranking in ((alpha, first), (1.0 - alpha, second)):
        for rank, item in enumerate(ranking, start=1):
            key = int(item)
            scores[key] = scores.get(key, 0.0) + weight / (c + rank)
    ordered = sorted(scores, key=lambda item: (-scores[item], item))
    return ordered if top_k is None else ordered[:int(top_k)]


def select_rrf_weight(
    examples: Sequence[Mapping[str, object]], weights: Sequence[float], c: float, k: int,
) -> tuple[float, list[dict[str, float]]]:
    if not examples:
        raise ValueError("validation examples are required for RRF weight selection")
    rows = []
    for weight in weights:
        hits = 0
        for example in examples:
            ranking = reciprocal_rank_fusion(example["first"], example["second"], float(weight), c, k)
            hits += int(int(example["target"]) in ranking)
        rows.append({"alpha": float(weight), "Recall": hits / len(examples)})
    best = max(rows, key=lambda row: (row["Recall"], -abs(row["alpha"] - 0.5), -row["alpha"]))
    return float(best["alpha"]), rows

