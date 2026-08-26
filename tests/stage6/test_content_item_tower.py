import unittest
import torch
from torch import nn

from src.models.content_item_tower import ContentItemTower


class ContentItemTowerTest(unittest.TestCase):
    def test_side_field_count_is_strict(self):
        tower = ContentItemTower(nn.Embedding(10, 8, sparse=True), "I1", [5, 6], 8, 2, 4)
        with self.assertRaisesRegex(ValueError, "one token"):
            tower(torch.tensor([1]), torch.tensor([[2]]), torch.zeros((1, 4)), torch.tensor([True]), torch.tensor([0.]))

    def test_missing_mm_uses_explicit_embedding(self):
        tower = ContentItemTower(nn.Embedding(10, 8, sparse=True), "I2", [], 8, 2, 4)
        output = tower(torch.tensor([1]), mm=torch.zeros((1, 4)), mm_valid=torch.tensor([False]), train_counts=torch.tensor([0.]))
        self.assertTrue(torch.isfinite(output).all())

