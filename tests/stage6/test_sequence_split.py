import unittest
import numpy as np

from src.recall.stage6_data import split_long_short_session


class SessionSplitTest(unittest.TestCase):
    def test_last_large_gap_defines_short_session(self):
        split = split_long_short_session([0, 10, 1000, 1010], 100)
        np.testing.assert_array_equal(split.long_indices, [0, 1])
        np.testing.assert_array_equal(split.short_indices, [2, 3])

    def test_truncation_preserves_recent_order(self):
        split = split_long_short_session([0, 1, 2, 3, 4], 10, short_max_events=3)
        np.testing.assert_array_equal(split.short_indices, [2, 3, 4])
        self.assertTrue(split.short_truncated)

    def test_negative_gap_fails(self):
        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            split_long_short_session([2, 1], 10)

