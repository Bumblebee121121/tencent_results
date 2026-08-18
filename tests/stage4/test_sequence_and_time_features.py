from __future__ import annotations

import unittest

import numpy as np

from src.features.sequence_store import slice_history, validate_aligned_sequence
from src.features.time_features import bucketize_time, inter_event_gap_features, recency_features


class SequenceAndTimeFeaturesTest(unittest.TestCase):
    def test_history_alignment_length_and_strict_target_prefix(self) -> None:
        offsets = np.array([0, 0, 3], dtype=np.int64)
        items = np.array([5, 6, 7], dtype=np.int32)
        actions = np.array([1, 2, 1], dtype=np.int8)
        timestamps = np.array([10, 20, 30], dtype=np.int64)
        history = slice_history(1, 2, offsets, items, actions, timestamps, target_timestamp=30)
        self.assertEqual(history.length, 2)
        self.assertEqual(history.item_rid.tolist(), [5, 6])
        self.assertEqual(history.action_token.tolist(), [1, 2])
        self.assertLess(int(history.timestamp.max()), 30)
        with self.assertRaises(ValueError):
            slice_history(1, 3, offsets, items, actions, timestamps, target_timestamp=30)

    def test_alignment_and_nondecreasing_guards(self) -> None:
        with self.assertRaises(ValueError):
            validate_aligned_sequence(np.array([1]), np.array([1, 2]), np.array([10]))
        with self.assertRaises(ValueError):
            validate_aligned_sequence(np.array([1, 2]), np.array([1, 2]), np.array([20, 10]))

    def test_recency_gap_and_log_features_are_finite(self) -> None:
        timestamps = np.array([10, 10, 20], dtype=np.int64)
        recency, recency_log = recency_features(timestamps, 30)
        gaps, gap_log, first = inter_event_gap_features(timestamps)
        self.assertTrue(np.all(recency > 0))
        self.assertTrue(np.all(gaps >= 0))
        self.assertTrue(np.all(np.isfinite(recency_log)))
        self.assertTrue(np.all(np.isfinite(gap_log)))
        self.assertEqual(gaps.tolist(), [0, 0, 10])
        self.assertEqual(first.tolist(), [True, False, False])
        self.assertEqual(bucketize_time([0, 10, 11], [10]).tolist(), [0, 1, 1])


if __name__ == "__main__":
    unittest.main()
