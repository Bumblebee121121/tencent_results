import unittest
import numpy as np

from src.recall.stage6_index import audit_item_embeddings


class RetrievalTest(unittest.TestCase):
    def test_unseen_rows_are_counted(self):
        audit = audit_item_embeddings(np.eye(3, dtype=np.float32), ["Head", "Tail", "Unseen"])
        self.assertEqual(audit["indexed_candidate_count"], 3)
        self.assertEqual(audit["unseen_indexed_count"], 1)
        self.assertEqual(audit["nan_inf_count"], 0)

