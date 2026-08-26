import unittest
import torch
from torch import nn

from src.models.sdm_user_tower import SDMUserTower


class SDMUserTowerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.embedding = nn.Embedding(20, 8, padding_idx=0, sparse=True)

    def test_shape_and_normalization(self):
        model = SDMUserTower(self.embedding, 8, 2, 0.0)
        short = torch.tensor([[2, 3], [4, 0]])
        long = torch.tensor([[5, 6], [0, 0]])
        sm = short.eq(0); lm = long.eq(0)
        output = model(short, long, sm, lm)
        self.assertEqual(tuple(output.shape), (2, 8))
        torch.testing.assert_close(output.norm(dim=1), torch.ones(2))

    def test_all_padding_is_zero_not_nan(self):
        model = SDMUserTower(self.embedding, 8, 2, 0.0)
        values = torch.zeros((1, 2), dtype=torch.long); mask = values.eq(0)
        output = model(values, values, mask, mask)
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, torch.zeros_like(output))

