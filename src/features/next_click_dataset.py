"""Model-neutral next-click sample adapter over the unified FeatureStore."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .feature_store import FeatureStore
from .time_features import inter_event_gap_features, recency_features


class NextClickFeatureDataset:
    """Feature adapter; negative sampling and model architecture are intentionally absent."""

    def __init__(
        self,
        samples: Sequence[Mapping[str, object]],
        feature_store: FeatureStore,
        user_features: Mapping[int, Mapping[str, object]] | None = None,
    ):
        self.samples = samples
        self.feature_store = feature_store
        self.user_features = user_features or {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        user_id = int(sample["user_id"])
        item_rid_value = sample.get("target_item_rid")
        item_rid = None if item_rid_value is None else int(item_rid_value)
        target_timestamp = int(sample["target_timestamp"])
        history_end = int(sample["history_end_position"])
        history_length = int(sample["history_length"])
        history = self.feature_store.history(
            user_id, history_end, target_timestamp
        )
        if history.length != history_length:
            raise ValueError(
                f"sample history_length={history_length} but store slice has {history.length}"
            )
        side, side_missing, side_oov = self.feature_store.item_side_features(item_rid)
        mm, mm_valid = self.feature_store.item_mm(item_rid)
        result: dict[str, object] = {
            "sample_id": int(sample["sample_id"]),
            "user_id": user_id,
            "target_item_oid": int(sample["target_item_oid"]),
            "target_item_rid": item_rid,
            "target_item_token": self.feature_store.item_token(item_rid),
            "target_timestamp": target_timestamp,
            "hist_item_rid": history.item_rid,
            "hist_item_token": self.feature_store.history_item_tokens(history),
            "hist_action_token": history.action_token,
            "hist_timestamp": history.timestamp,
            "hist_length": history.length,
            "user_features": self.user_features.get(user_id),
            "target_item_side": side,
            "target_item_side_missing": side_missing,
            "target_item_side_oov": side_oov,
            "target_mm": mm,
            "target_mm_valid": mm_valid,
        }
        result.update(self.feature_store.item_train_strength(item_rid))
        return result


def add_dynamic_time_features(batch_item: Mapping[str, object]) -> dict[str, object]:
    result = dict(batch_item)
    timestamps = np.asarray(result["hist_timestamp"], dtype=np.int64)
    recency, recency_log = recency_features(timestamps, int(result["target_timestamp"]))
    gaps, gaps_log, first_mask = inter_event_gap_features(timestamps)
    result.update(
        {
            "hist_recency": recency,
            "hist_recency_log1p": recency_log,
            "hist_time_gap": gaps,
            "hist_time_gap_log1p": gaps_log,
            "hist_first_event_mask": first_mask,
        }
    )
    return result
