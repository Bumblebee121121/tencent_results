"""Train-scope categorical vocabularies with explicit missing and OOV tokens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .id_semantics import CATEGORICAL_TOKENS


def normalize_integer_value(value: object | None) -> int | None:
    """Normalize item/user integer values and candidate decimal strings."""

    if value is None:
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped, 10)
        except ValueError as error:
            raise ValueError(f"categorical value is not a decimal integer: {value!r}") from error
    raise TypeError(f"unsupported categorical value type: {type(value).__name__}")


@dataclass(frozen=True)
class EncodedValue:
    token: int
    missing: bool
    oov: bool


class CategoricalVocabulary:
    """Sorted train-known integer values; token 3+ is stable by value order."""

    def __init__(self, known_values: Sequence[int] | np.ndarray):
        values = np.asarray(known_values, dtype=np.int64)
        if values.ndim != 1:
            raise ValueError("known_values must be one-dimensional")
        self.known_values = np.unique(values)

    @classmethod
    def fit(cls, values: Iterable[object | None]) -> "CategoricalVocabulary":
        normalized = [normalize_integer_value(value) for value in values]
        return cls([value for value in normalized if value is not None])

    @property
    def known_size(self) -> int:
        return int(self.known_values.size)

    @property
    def vocabulary_size_with_specials(self) -> int:
        return self.known_size + 3

    def encode(self, value: object | None) -> EncodedValue:
        normalized = normalize_integer_value(value)
        if normalized is None:
            return EncodedValue(CATEGORICAL_TOKENS["missing"], True, False)
        index = int(np.searchsorted(self.known_values, normalized))
        if index < self.known_values.size and int(self.known_values[index]) == normalized:
            return EncodedValue(index + 3, False, False)
        return EncodedValue(CATEGORICAL_TOKENS["oov"], False, True)

    def encode_many(self, values: Sequence[object | None]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        encoded = [self.encode(value) for value in values]
        return (
            np.asarray([value.token for value in encoded], dtype=np.int32),
            np.asarray([value.missing for value in encoded], dtype=np.bool_),
            np.asarray([value.oov for value in encoded], dtype=np.bool_),
        )

    def encode_list(self, values: Sequence[object | None] | None) -> tuple[list[int], bool, list[bool]]:
        if values is None:
            return [], True, []
        encoded = [self.encode(value) for value in values]
        return [value.token for value in encoded], False, [value.oov for value in encoded]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.known_values, allow_pickle=False)

    @classmethod
    def load(cls, path: Path, mmap_mode: str | None = "r") -> "CategoricalVocabulary":
        return cls(np.load(path, mmap_mode=mmap_mode, allow_pickle=False))
