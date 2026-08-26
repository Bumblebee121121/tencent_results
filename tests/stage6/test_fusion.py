import unittest

from src.recall.fusion import reciprocal_rank_fusion, select_rrf_weight


class FusionTest(unittest.TestCase):
    def test_duplicate_accumulates_both_scores(self):
        ranking = reciprocal_rank_fusion([1, 2], [2, 3], 0.5, 60)
        self.assertEqual(ranking[0], 2)
        self.assertEqual(len(ranking), 3)

    def test_weight_selection_uses_examples(self):
        examples = [{"target": 1, "first": [1], "second": [2]}]
        alpha, rows = select_rrf_weight(examples, [0., 1.], 60, 1)
        self.assertEqual(alpha, 1.0)
        self.assertEqual(len(rows), 2)

