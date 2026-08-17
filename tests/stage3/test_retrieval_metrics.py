from __future__ import annotations

import math
import unittest

from src.evaluation.retrieval_metrics import (
    channel_incremental_metrics,
    channel_set_metrics,
    evaluate_retrieval,
)


class RetrievalMetricsTest(unittest.TestCase):
    def test_recall_and_ndcg_match_manual_values(self) -> None:
        result = evaluate_retrieval(
            rankings=[[1, 2, 3], [4, 5, 6], [9, 8, 7]],
            targets=[1, 6, 0],
            groups=["Head", "Tail", "Unseen"],
            recall_ks=[1, 3],
            ndcg_ks=[1, 3],
        )
        self.assertAlmostEqual(result["Overall"]["Recall@1"], 1 / 3)
        self.assertAlmostEqual(result["Overall"]["Recall@3"], 2 / 3)
        expected_ndcg = (1.0 + 1.0 / math.log2(4)) / 3
        self.assertAlmostEqual(result["Overall"]["NDCG@3"], expected_ndcg)
        self.assertEqual(result["Tail"]["Recall@3"], 1.0)
        self.assertEqual(result["Unseen"]["Recall@3"], 0.0)

    def test_channel_metrics(self) -> None:
        self.assertEqual(channel_set_metrics([1, 2], [2, 3])["jaccard"], 1 / 3)
        result = channel_incremental_metrics([[1], [4]], [[2], [3]], [2, 4], k=1)
        self.assertEqual(result["baseline_hits"], 1)
        self.assertEqual(result["incremental_hits"], 1)
        self.assertEqual(result["union_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
