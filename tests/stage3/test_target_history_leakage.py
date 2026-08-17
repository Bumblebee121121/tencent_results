from __future__ import annotations

import unittest

from src.data.attribution import attribute_sequence


class TargetHistoryLeakageTest(unittest.TestCase):
    def test_history_stops_before_target_exposure_timestamp(self) -> None:
        events = [
            {"item_id": 2, "action_type": None, "timestamp": 5},
            {"item_id": 1, "action_type": 0, "timestamp": 10},
            {"item_id": 3, "action_type": 0, "timestamp": 10},
            {"item_id": 1, "action_type": 1, "timestamp": 20},
        ]
        samples, stats = attribute_sequence(7, events, lambda rid: 1000 + rid)
        self.assertEqual(stats.attributed_count, 1)
        sample = samples[0]
        history = events[: sample["history_end_position"]]
        self.assertEqual(sample["history_end_position"], 1)
        self.assertTrue(all(event["timestamp"] < sample["target_exposure_timestamp"] for event in history))
        self.assertNotIn(events[sample["target_exposure_position"]], history)
        self.assertNotIn(events[sample["target_click_position"]], history)

    def test_nearest_preceding_exposure_is_used(self) -> None:
        events = [
            {"item_id": 1, "action_type": 0, "timestamp": 1},
            {"item_id": 1, "action_type": 0, "timestamp": 3},
            {"item_id": 1, "action_type": 1, "timestamp": 8},
        ]
        samples, stats = attribute_sequence(8, events, lambda rid: rid + 10)
        self.assertEqual(samples[0]["target_exposure_position"], 1)
        self.assertEqual(samples[0]["attribution_gap"], 5)
        self.assertEqual(stats.multiple_preceding_exposure_count, 1)


if __name__ == "__main__":
    unittest.main()
