"""Item side and train-only strength helpers."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from src.data.item_strength import classify_strength

from .categorical_encoder import CategoricalVocabulary


def encode_item_feature_row(
    values: Sequence[object | None],
    vocabularies: Sequence[CategoricalVocabulary],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(values) != len(vocabularies):
        raise ValueError("feature values and vocabularies must have equal length")
    encoded = [vocab.encode(value) for vocab, value in zip(vocabularies, values)]
    return (
        np.asarray([value.token for value in encoded], dtype=np.int32),
        np.asarray([value.missing for value in encoded], dtype=np.bool_),
        np.asarray([value.oov for value in encoded], dtype=np.bool_),
    )


def item_strength_features(count: int, p50_train: float, p90_train: float) -> dict[str, object]:
    value = int(count)
    if value < 0:
        raise ValueError("item train count cannot be negative")
    return {
        "item_train_count": value,
        "item_train_count_log1p": float(math.log1p(value)),
        "strength_group": classify_strength(value, p50_train, p90_train),
    }


def verify_stage3_counts(
    full_counts: np.ndarray,
    item_rids: Sequence[int | None],
    expected_counts: Sequence[int],
) -> None:
    if len(item_rids) != len(expected_counts):
        raise ValueError("Stage 3 count vectors must have equal length")
    for index, (rid_value, expected) in enumerate(zip(item_rids, expected_counts)):
        actual = 0 if rid_value is None else int(full_counts[int(rid_value)])
        if actual != int(expected):
            raise ValueError(
                "Stage 4 train count disagrees with Stage 3 at row "
                f"{index}: rid={rid_value}, actual={actual}, expected={int(expected)}"
            )
