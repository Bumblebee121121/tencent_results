import unittest
import torch
from torch import nn

from src.models.content_item_tower import ContentItemTower


class UnseenCandidateTest(unittest.TestCase):
    def test_e1_ignores_shared_unk_id_for_unseen(self):
        embedding = nn.Embedding(5, 8, sparse=True)
        tower = ContentItemTower(embedding, "E1", [6], 8, 2, 4)
        common = {"item_tokens": torch.tensor([1, 1]), "side_tokens": torch.tensor([[3], [4]]),
                  "mm": torch.tensor([[1., 0, 0, 0], [0., 1, 0, 0]]), "mm_valid": torch.tensor([True, True]),
                  "train_counts": torch.tensor([0., 0.])}
        output, gate = tower(**common, return_gate=True)
        torch.testing.assert_close(gate, torch.zeros(2))
        self.assertFalse(torch.allclose(output[0], output[1]))

