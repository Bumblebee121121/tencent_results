"""Cross-source feature comparison and deterministic source precedence."""

from __future__ import annotations

from dataclasses import dataclass

from .categorical_encoder import normalize_integer_value


OFFICIAL_SOURCE = "official_candidate"
ADDED_TARGET_SOURCE = "item_feat_added_target"


@dataclass
class FeatureComparison:
    both_non_null_count: int = 0
    equal_count: int = 0
    mismatch_count: int = 0
    candidate_null_item_non_null: int = 0
    candidate_non_null_item_null: int = 0

    def update(self, candidate_value: object | None, item_value: object | None) -> None:
        candidate = normalize_integer_value(candidate_value)
        item = normalize_integer_value(item_value)
        if candidate is not None and item is not None:
            self.both_non_null_count += 1
            if candidate == item:
                self.equal_count += 1
            else:
                self.mismatch_count += 1
        elif candidate is None and item is not None:
            self.candidate_null_item_non_null += 1
        elif candidate is not None and item is None:
            self.candidate_non_null_item_null += 1

    def as_row(self, field: str) -> dict[str, object]:
        return {
            "feature": field,
            "both_non_null_count": self.both_non_null_count,
            "equal_count": self.equal_count,
            "mismatch_count": self.mismatch_count,
            "match_ratio": (
                self.equal_count / self.both_non_null_count
                if self.both_non_null_count
                else None
            ),
            "candidate_null_item_non_null": self.candidate_null_item_non_null,
            "candidate_non_null_item_null": self.candidate_non_null_item_null,
        }


def select_item_side_source(eval_source: str) -> str:
    if eval_source == "official":
        return OFFICIAL_SOURCE
    if eval_source in {"validation_target", "test_target"}:
        return ADDED_TARGET_SOURCE
    raise ValueError(f"unknown evaluation candidate source: {eval_source!r}")
