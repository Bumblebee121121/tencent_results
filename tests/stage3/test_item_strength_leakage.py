from __future__ import annotations

import unittest

from src.data.item_strength import count_events_before_cutoff


class ItemStrengthLeakageTest(unittest.TestCase):
    def test_validation_events_do_not_change_train_count(self) -> None:
        sequence = [
            {"item_id": 1, "action_type": 0, "timestamp": 1},
            {"item_id": 1, "action_type": 1, "timestamp": 2},
            *[
                {"item_id": 1, "action_type": 0, "timestamp": 10 + index}
                for index in range(100)
            ],
        ]
        counts = count_events_before_cutoff([sequence], cutoff_exclusive=10, max_item_rid=2)
        self.assertEqual(int(counts[1]), 2)


if __name__ == "__main__":
    unittest.main()
