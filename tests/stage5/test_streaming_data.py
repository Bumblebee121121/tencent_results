from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.recall.data import (
    ParquetSampleIterableDataset, Stage5SequenceStore, TwoTowerCollator, UniformNegativeSampler,
)
from tests.stage5._temp import workspace_tempdir


def make_store(root: Path) -> Stage5SequenceStore:
    feature = root / "feature_store"
    mappings = root / "mappings"
    feature.mkdir(parents=True)
    mappings.mkdir(parents=True)
    np.save(feature / "user_seq_offsets.npy", np.array([0, 3], dtype=np.int64))
    np.save(feature / "seq_item_rid.npy", np.array([1, 2, 3], dtype=np.int32))
    np.save(feature / "seq_action_token.npy", np.array([1, 2, 1], dtype=np.int8))
    np.save(feature / "seq_timestamp.npy", np.array([10, 20, 30], dtype=np.int64))
    np.save(mappings / "rid_to_model_item_token.npy", np.array([1, 2, 3, 4, 5, 6], dtype=np.int32))
    np.save(mappings / "train_item_count_by_rid.npy", np.array([0, 1, 1, 1, 1, 1], dtype=np.int64))
    return Stage5SequenceStore(root)


class StreamingDataTest(unittest.TestCase):
    def test_strict_prefix_collation_and_negative_exclusion(self):
        with workspace_tempdir() as temporary:
            store = make_store(temporary)
            row = {
                "sample_id": 1, "user_id": 0, "target_item_rid": 3, "target_item_oid": 30,
                "target_timestamp": 30, "history_end_position": 2, "history_length": 2,
            }
            sampler = UniformNegativeSampler([2, 3, 4, 5, 6], negatives=2, seed=7)
            collator = TwoTowerCollator(store, sampler)
            batch = collator([row])
            self.assertEqual([[2, 3]], batch["history_tokens"].tolist())
            self.assertEqual([4], batch["target_tokens"].tolist())
            self.assertTrue(set(batch["negative_tokens"][0].tolist()).isdisjoint({2, 3, 4}))
            bad = dict(row, target_timestamp=20)
            with self.assertRaises(ValueError):
                store.history(bad)
            del batch, collator, store

    def test_iterable_dataset_uses_requested_split_and_limit(self):
        with workspace_tempdir() as root:
            schema = pa.schema([(name, pa.int64()) for name in (
                "sample_id", "user_id", "target_item_rid", "target_item_oid", "target_timestamp",
                "history_end_position", "history_length",
            )])
            for name, sample_id in (("train", 1), ("validation", 2)):
                pq.write_table(pa.Table.from_pylist([{
                    "sample_id": sample_id, "user_id": 0, "target_item_rid": 3,
                    "target_item_oid": 30, "target_timestamp": 30,
                    "history_end_position": 2, "history_length": 2,
                }], schema=schema), root / f"{name}.parquet")
            self.assertEqual([1], [row["sample_id"] for row in ParquetSampleIterableDataset(root / "train.parquet", 1)])
            self.assertEqual([2], [row["sample_id"] for row in ParquetSampleIterableDataset(root / "validation.parquet", 1)])


if __name__ == "__main__":
    unittest.main()
