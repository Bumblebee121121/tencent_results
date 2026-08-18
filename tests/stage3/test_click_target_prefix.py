from __future__ import annotations

import unittest

from src.data.click_target import find_click_targets
from src.data.sample_builder import SAMPLE_SCHEMA, build_samples_for_users, make_rid_to_oid


class ClickTargetPrefixTest(unittest.TestCase):
    def test_click_labeled_interaction_becomes_target(self) -> None:
        events = [
            {"item_id": 1, "action_type": 0, "timestamp": 10},
            {"item_id": 2, "action_type": 1, "timestamp": 20},
            {"item_id": 3, "action_type": 0, "timestamp": 30},
            {"item_id": 4, "action_type": 1, "timestamp": 40},
        ]
        targets, stats = find_click_targets(events)
        self.assertEqual(stats.click_target_count, 2)
        self.assertEqual([target["target_item_rid"] for target in targets], [2, 4])
        self.assertEqual(events[: targets[0]["history_end_position"]], events[:1])
        self.assertEqual(events[: targets[1]["history_end_position"]], events[:3])

    def test_non_click_interaction_is_not_target(self) -> None:
        events = [
            {"item_id": 1, "action_type": 0, "timestamp": 10},
            {"item_id": 2, "action_type": None, "timestamp": 20},
        ]
        targets, stats = find_click_targets(events)
        self.assertEqual(targets, [])
        self.assertEqual(stats.click_target_count, 0)
        self.assertEqual(stats.unknown_action_count, 1)

    def test_history_stops_before_target_timestamp(self) -> None:
        events = [
            {"item_id": 1, "action_type": 0, "timestamp": 10},
            {"item_id": 2, "action_type": 0, "timestamp": 15},
            {"item_id": 3, "action_type": 1, "timestamp": 20},
        ]
        targets, _ = find_click_targets(events)
        target = targets[0]
        history = events[: target["history_end_position"]]
        self.assertTrue(all(event["timestamp"] < target["target_timestamp"] for event in history))

    def test_same_timestamp_events_are_excluded(self) -> None:
        events = [
            {"item_id": 1, "action_type": 0, "timestamp": 10},
            {"item_id": 2, "action_type": 0, "timestamp": 20},
            {"item_id": 3, "action_type": 1, "timestamp": 20},
        ]
        targets, stats = find_click_targets(events)
        target = targets[0]
        self.assertEqual(events[: target["history_end_position"]], events[:1])
        self.assertEqual(stats.same_timestamp_prefix_excluded_count, 1)

    def test_target_itself_is_not_in_history(self) -> None:
        events = [
            {"item_id": 1, "action_type": 0, "timestamp": 10},
            {"item_id": 2, "action_type": 1, "timestamp": 20},
        ]
        targets, _ = find_click_targets(events)
        target = targets[0]
        history_positions = range(target["history_end_position"])
        self.assertNotIn(target["target_position"], history_positions)

    def test_empty_history_target_is_counted_and_skipped(self) -> None:
        events = [
            {"item_id": 1, "action_type": 1, "timestamp": 10},
            {"item_id": 2, "action_type": 0, "timestamp": 20},
        ]
        reverse = make_rid_to_oid({20001: 1, 20002: 2})
        samples, stats = build_samples_for_users([7], [events], reverse)
        self.assertEqual(stats.click_target_count, 1)
        self.assertEqual(stats.empty_history_target_count, 1)
        self.assertEqual(samples, [])

    def test_compact_schema_uses_v2_target_fields(self) -> None:
        self.assertIn("target_timestamp", SAMPLE_SCHEMA.names)
        self.assertIn("target_action_type", SAMPLE_SCHEMA.names)
        self.assertNotIn("target_exposure_timestamp", SAMPLE_SCHEMA.names)
        self.assertNotIn("attribution_gap", SAMPLE_SCHEMA.names)


if __name__ == "__main__":
    unittest.main()
