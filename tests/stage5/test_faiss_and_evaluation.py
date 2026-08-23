from __future__ import annotations

import unittest

import numpy as np

from src.recall.evaluation import complementarity_from_ranks, metrics_from_ranks
from src.recall.faiss_utils import build_hnsw_ip, filter_history_from_faiss_rows, train_seen_candidate_rows


class FaissEvaluationTest(unittest.TestCase):
    def test_unseen_exclusion_and_physical_row_alignment(self):
        rows = [
            {"item_oid": 10, "item_rid": 1, "retrieval_id": 99},
            {"item_oid": 20, "item_rid": 2, "retrieval_id": 3},
            {"item_oid": 30, "item_rid": None, "retrieval_id": 0},
        ]
        # RID 2 is shared UNK despite having a historical RID.
        selected = train_seen_candidate_rows(rows, np.array([1, 2, 1], dtype=np.int32))
        self.assertEqual([0], [row["faiss_row"] for row in selected])
        self.assertEqual([99], [row["retrieval_id"] for row in selected])

    def test_toy_faiss_retrieval_and_history_filter(self):
        embeddings = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        index = build_hnsw_ip(embeddings, 4, 20, 20)
        _, rows = index.search(np.array([[1, 0]], dtype=np.float32), 3)
        ranking = filter_history_from_faiss_rows(rows[0], np.array([10, 20, 30]), [10], 2)
        self.assertEqual(20, ranking[0])

    def test_group_metrics_unseen_denominator_and_complementarity(self):
        metrics = metrics_from_ranks([1, None, 50], ["Head", "Unseen", "Tail"], [10, 50], [10])
        self.assertEqual(3, metrics["Overall"]["count"])
        self.assertAlmostEqual(1 / 3, metrics["Overall"]["Recall@10"])
        self.assertEqual(0.0, metrics["Unseen"]["Recall@50"])
        comp = complementarity_from_ranks([1, None, 100], [1, 5, None], [10])
        self.assertEqual(1, comp["@10"]["both_hit"])
        self.assertEqual(1, comp["@10"]["second_only_hit"])
        self.assertAlmostEqual(2 / 3, comp["@10"]["union_recall"])


if __name__ == "__main__":
    unittest.main()
