"""Unified retrieval metrics for all recommendation models."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Hashable, Mapping, Sequence


Item = Hashable


def _validate_ks(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(sorted({int(value) for value in values}))
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive integers")
    return result


def target_rank(ranking: Sequence[Item], target: Item) -> int | None:
    """Return the one-based first rank of target, or None when absent."""

    for index, item in enumerate(ranking, start=1):
        if item == target:
            return index
    return None


def evaluate_retrieval(
    rankings: Sequence[Sequence[Item]],
    targets: Sequence[Item],
    groups: Sequence[str] | None = None,
    recall_ks: Sequence[int] = (10, 50, 100, 500),
    ndcg_ks: Sequence[int] = (10, 50, 100),
) -> dict[str, dict[str, float | int]]:
    """Evaluate one-ground-truth retrieval and optional strength groups."""

    if len(rankings) != len(targets):
        raise ValueError("rankings and targets must have equal length")
    if groups is not None and len(groups) != len(targets):
        raise ValueError("groups and targets must have equal length")
    recall_ks = _validate_ks(recall_ks, "recall_ks")
    ndcg_ks = _validate_ks(ndcg_ks, "ndcg_ks")

    indices: dict[str, list[int]] = {"Overall": list(range(len(targets)))}
    if groups is not None:
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, group in enumerate(groups):
            grouped[str(group)].append(index)
        for group in ("Head", "Mid", "Tail", "Unseen"):
            indices[group] = grouped.get(group, [])
        for group in sorted(set(grouped) - {"Head", "Mid", "Tail", "Unseen"}):
            indices[group] = grouped[group]

    ranks = [target_rank(ranking, target) for ranking, target in zip(rankings, targets)]
    output: dict[str, dict[str, float | int]] = {}
    for group, selected in indices.items():
        metrics: dict[str, float | int] = {"count": len(selected)}
        for k in recall_ks:
            hits = sum(ranks[index] is not None and ranks[index] <= k for index in selected)
            value = hits / len(selected) if selected else 0.0
            metrics[f"Recall@{k}"] = value
            metrics[f"HitRate@{k}"] = value
        for k in ndcg_ks:
            gain = sum(
                1.0 / math.log2(ranks[index] + 1)
                for index in selected
                if ranks[index] is not None and ranks[index] <= k
            )
            metrics[f"NDCG@{k}"] = gain / len(selected) if selected else 0.0
        output[group] = metrics
    return output


def channel_set_metrics(first: Sequence[Item], second: Sequence[Item]) -> dict[str, float | int]:
    first_set, second_set = set(first), set(second)
    intersection = first_set & second_set
    union = first_set | second_set
    return {
        "intersection_count": len(intersection),
        "union_count": len(union),
        "jaccard": len(intersection) / len(union) if union else 0.0,
    }


def channel_incremental_metrics(
    baseline_rankings: Sequence[Sequence[Item]],
    added_rankings: Sequence[Sequence[Item]],
    targets: Sequence[Item],
    k: int,
) -> Mapping[str, float | int]:
    if not (len(baseline_rankings) == len(added_rankings) == len(targets)):
        raise ValueError("channel rankings and targets must have equal length")
    if k <= 0:
        raise ValueError("k must be positive")
    baseline_hits = 0
    incremental_hits = 0
    union_hits = 0
    for baseline, added, target in zip(baseline_rankings, added_rankings, targets):
        baseline_hit = target in baseline[:k]
        added_hit = target in added[:k]
        baseline_hits += baseline_hit
        incremental_hits += added_hit and not baseline_hit
        union_hits += baseline_hit or added_hit
    count = len(targets)
    return {
        "count": count,
        "baseline_hits": baseline_hits,
        "incremental_hits": incremental_hits,
        "union_hits": union_hits,
        "incremental_recall": incremental_hits / count if count else 0.0,
        "union_recall": union_hits / count if count else 0.0,
    }
