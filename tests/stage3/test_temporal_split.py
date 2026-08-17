from __future__ import annotations

import unittest

import numpy as np

from src.data.temporal_split import assign_splits, choose_temporal_cutoffs, earliest_sample_ids


class TemporalSplitTest(unittest.TestCase):
    def test_timestamp_ties_never_cross_splits(self) -> None:
        timestamps = np.array([1, 1, 2, 3, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        cutoffs = choose_temporal_cutoffs(timestamps, 0.6, 0.2, 0.2)
        groups = assign_splits(timestamps, cutoffs)
        for timestamp in np.unique(timestamps):
            self.assertEqual(len(set(groups[timestamps == timestamp])), 1)
        self.assertLess(timestamps[groups == "train"].max(), timestamps[groups == "val"].min())
        self.assertLess(timestamps[groups == "val"].max(), timestamps[groups == "test"].min())

    def test_primary_is_earliest_target_per_user(self) -> None:
        selected = earliest_sample_ids(
            np.array([10, 11, 20, 21]),
            np.array([1, 1, 2, 2]),
            np.array([5, 4, 4, 4]),
        )
        np.testing.assert_array_equal(selected, np.array([11, 20]))

    def test_large_tie_moves_to_available_boundaries(self) -> None:
        timestamps = np.array([1] + [2] * 98 + [3], dtype=np.int64)
        cutoffs = choose_temporal_cutoffs(timestamps, 0.8, 0.1, 0.1)
        groups = assign_splits(timestamps, cutoffs)
        self.assertEqual(set(groups.tolist()), {"train", "val", "test"})
        self.assertEqual(set(groups[timestamps == 2].tolist()), {"val"})


if __name__ == "__main__":
    unittest.main()
