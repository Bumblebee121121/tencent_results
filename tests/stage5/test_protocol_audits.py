from __future__ import annotations

import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from src.recall.audits import repeated_target_stats, target_is_repeated
from tests.stage5._temp import workspace_tempdir
from tests.stage5.test_streaming_data import make_store


class ProtocolAuditTest(unittest.TestCase):
    def test_repeated_target_detection_and_split_ratio(self):
        self.assertTrue(target_is_repeated(1, [1, 2]))
        self.assertFalse(target_is_repeated(3, [1, 2]))
        with workspace_tempdir() as root:
            store = make_store(root / "stage4")
            rows = [
                {"sample_id": 1, "user_id": 0, "target_item_rid": 2, "target_item_oid": 20,
                 "target_timestamp": 20, "history_end_position": 1, "history_length": 1},
                {"sample_id": 2, "user_id": 0, "target_item_rid": 1, "target_item_oid": 10,
                 "target_timestamp": 30, "history_end_position": 2, "history_length": 2},
            ]
            pq.write_table(pa.Table.from_pylist(rows), root / "samples.parquet")
            stats = repeated_target_stats(root / "samples.parquet", store)
            self.assertEqual(2, stats["sample_count"])
            self.assertEqual(1, stats["repeated_target_count"])
            self.assertEqual(0.5, stats["repeated_target_ratio"])
            del store


if __name__ == "__main__":
    unittest.main()
