from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.recall.itemcf import (
    NeighborShardLookup, build_partitioned_itemcf, cosine_pair_scores,
    local_directed_pairs, retrieve_itemcf, top_neighbors,
)
from tests.stage5._temp import workspace_tempdir


class ItemCFTest(unittest.TestCase):
    def test_window_weights_cosine_candidate_and_topn(self):
        users = [([1, 2, 3], [1, 2, 1]), ([1, 2], [1, 1])]
        scores = cosine_pair_scores(users, {2}, window=1, exposure_weight=1, click_weight=3)
        self.assertIn((1, 2), scores)
        self.assertIn((3, 2), scores)
        self.assertNotIn((2, 1), scores)  # neighbor must be a candidate
        self.assertAlmostEqual(scores[(1, 2)], 4 / np.sqrt(2 * 10))
        result = top_neighbors({(1, 2): 0.5, (1, 3): 0.8, (1, 4): 0.2}, 2)
        self.assertEqual([3, 2], [rid for rid, _ in result[1]])

    def test_unknown_window_and_self_are_ignored(self):
        pairs = local_directed_pairs([1, 2, 1], [True, False, True], {1, 2}, 2)
        self.assertEqual(set(), pairs)

    def test_partitioned_build_cutoff_non_candidate_seed_and_retrieval(self):
        offsets = np.array([0, 3], dtype=np.int64)
        items = np.array([10, 20, 30], dtype=np.int32)
        actions = np.array([1, 2, 2], dtype=np.int8)
        timestamps = np.array([1, 2, 99], dtype=np.int64)
        with workspace_tempdir() as root:
            stats = build_partitioned_itemcf(
                offsets, items, actions, timestamps, cutoff=10,
                candidate_oid_by_rid={20: 2000, 30: 3000},
                output_path=root / "neighbors.parquet", shard_root=root / "shards",
                window=2, topn=5, partitions=2, buffer_size=1, overwrite=False,
            )
            rows = pq.read_table(root / "neighbors.parquet").to_pylist()
            self.assertGreater(stats["neighbor_rows"], 0)
            self.assertTrue(any(row["item_rid"] == 10 and row["neighbor_item_rid"] == 20 for row in rows))
            self.assertFalse(any(row["neighbor_item_rid"] == 30 for row in rows))
            lookup = NeighborShardLookup(root / "shards", "click3", 2)
            self.assertEqual([20], retrieve_itemcf([10], [1], lookup, 5, 1, 3, 0, None))
            # Repeated historical items remain legal targets by default.
            self.assertEqual([20], retrieve_itemcf([10, 20], [1, 2], lookup, 5, 1, 3, 0, None))
            self.assertEqual(
                [],
                retrieve_itemcf(
                    [10, 20], [1, 2], lookup, 5, 1, 3, 0, None,
                    exclude_history_items=True,
                ),
            )
            del lookup


if __name__ == "__main__":
    unittest.main()
