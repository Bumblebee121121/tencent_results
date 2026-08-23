"""Rank-based retrieval metrics and channel complementarity."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence


STANDARD_GROUPS = ("Overall", "Head", "Mid", "Tail", "Unseen")


def metrics_from_ranks(
    ranks: Sequence[int | None],
    groups: Sequence[str],
    recall_ks: Sequence[int] = (10, 50, 100, 500),
    ndcg_ks: Sequence[int] = (10, 50, 100),
) -> dict[str, dict[str, float | int]]:
    if len(ranks) != len(groups):
        raise ValueError("ranks and groups must have equal length")
    selected: dict[str, list[int]] = defaultdict(list)
    selected["Overall"] = list(range(len(ranks)))
    for index, group in enumerate(groups):
        selected[str(group)].append(index)
    output: dict[str, dict[str, float | int]] = {}
    ordered_groups = list(STANDARD_GROUPS) + sorted(set(selected) - set(STANDARD_GROUPS))
    for group in ordered_groups:
        indices = selected.get(group, [])
        metrics: dict[str, float | int] = {"count": len(indices)}
        for k in sorted(set(map(int, recall_ks))):
            hits = sum(ranks[i] is not None and int(ranks[i]) <= k for i in indices)
            value = hits / len(indices) if indices else 0.0
            metrics[f"Recall@{k}"] = value
            metrics[f"HitRate@{k}"] = value
        for k in sorted(set(map(int, ndcg_ks))):
            gain = sum(
                1.0 / math.log2(int(ranks[i]) + 1)
                for i in indices
                if ranks[i] is not None and int(ranks[i]) <= k
            )
            metrics[f"NDCG@{k}"] = gain / len(indices) if indices else 0.0
        output[group] = metrics
    return output


def complementarity_from_ranks(
    first_ranks: Sequence[int | None], second_ranks: Sequence[int | None], ks: Iterable[int]
) -> dict[str, dict[str, float | int]]:
    if len(first_ranks) != len(second_ranks):
        raise ValueError("rank arrays must have equal length")
    count = len(first_ranks)
    output = {}
    for k in sorted(set(map(int, ks))):
        first = [rank is not None and int(rank) <= k for rank in first_ranks]
        second = [rank is not None and int(rank) <= k for rank in second_ranks]
        both = sum(a and b for a, b in zip(first, second))
        first_only = sum(a and not b for a, b in zip(first, second))
        second_only = sum(b and not a for a, b in zip(first, second))
        union = both + first_only + second_only
        output[f"@{k}"] = {
            "count": count,
            "both_hit": both,
            "first_only_hit": first_only,
            "second_only_hit": second_only,
            "neither_hit": count - union,
            "union_recall": union / count if count else 0.0,
            "second_incremental_recall": second_only / count if count else 0.0,
        }
    return output
