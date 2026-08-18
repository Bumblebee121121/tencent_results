from __future__ import annotations

import unittest

import numpy as np

from src.features.id_semantics import ITEM_TOKENS, build_rid_to_model_item_token, encode_item_rid
from src.features.item_feature_store import verify_stage3_counts
from src.features.next_click_dataset import NextClickFeatureDataset, add_dynamic_time_features
from src.features.sequence_store import HistorySlice


class FakeFeatureStore:
    def __init__(self) -> None:
        self.counts = np.array([0, 2, 0], dtype=np.int32)
        self.tokens = build_rid_to_model_item_token(self.counts)

    def history(self, user_id: int, history_end_position: int, target_timestamp: int) -> HistorySlice:
        history = HistorySlice(
            np.array([1, 2], dtype=np.int32),
            np.array([1, 3], dtype=np.int8),
            np.array([10, 20], dtype=np.int64),
        )
        if history_end_position != history.length or int(history.timestamp.max()) >= target_timestamp:
            raise ValueError("invalid toy prefix")
        return history

    def history_item_tokens(self, history: HistorySlice) -> np.ndarray:
        return self.tokens[history.item_rid]

    def item_side_features(self, item_rid: int | None):
        return np.array([2, 3]), np.array([False, False]), np.array([True, False])

    def item_mm(self, item_rid: int | None):
        return np.zeros(32, dtype=np.float32), False

    def item_token(self, item_rid: int | None) -> int:
        return encode_item_rid(item_rid, rid_to_token=self.tokens)

    def item_train_strength(self, item_rid: int | None):
        count = 0 if item_rid is None else int(self.counts[item_rid])
        return {
            "target_train_count": count,
            "target_train_count_log1p": float(np.log1p(count)),
            "target_strength_group": "Unseen" if count == 0 else "Tail",
        }


class CountAndDatasetContractsTest(unittest.TestCase):
    def test_stage3_count_consistency_passes_and_mismatch_fails(self) -> None:
        counts = np.array([0, 3, 0, 7], dtype=np.int32)
        verify_stage3_counts(counts, [1, None, 3], [3, 0, 7])
        with self.assertRaises(ValueError):
            verify_stage3_counts(counts, [1], [4])

    def test_feature_dataset_smoke_and_unseen_target_unk(self) -> None:
        sample = {
            "sample_id": 9,
            "user_id": 1,
            "target_item_oid": 20000000001,
            "target_item_rid": 2,
            "target_timestamp": 30,
            "history_end_position": 2,
            "history_length": 2,
        }
        dataset = NextClickFeatureDataset([sample], FakeFeatureStore(), {1: {"f103_token": 3}})  # type: ignore[arg-type]
        item = dataset[0]
        self.assertEqual(item["target_item_token"], ITEM_TOKENS["unk"])
        self.assertEqual(item["hist_length"], 2)
        self.assertEqual(item["hist_item_rid"].tolist(), [1, 2])
        self.assertEqual(item["target_item_side_oov"].tolist(), [True, False])
        timed = add_dynamic_time_features(item)
        self.assertEqual(timed["hist_recency"].tolist(), [20, 10])
        self.assertTrue(np.all(np.isfinite(timed["hist_time_gap_log1p"])))


if __name__ == "__main__":
    unittest.main()
