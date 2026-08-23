from __future__ import annotations

import unittest

import torch

from src.models.vanilla_two_tower import VanillaTwoTower
from src.recall.runtime import select_device


class TwoTowerModelTest(unittest.TestCase):
    def test_forward_shape_pad_mean_and_gradient(self):
        model = VanillaTwoTower(8, embedding_dim=2)
        with torch.no_grad():
            model.item_embedding.weight.zero_()
            model.item_embedding.weight[1] = torch.tensor([1.0, 0.0])
            model.item_embedding.weight[2] = torch.tensor([0.0, 1.0])
            model.item_embedding.weight[3] = torch.tensor([1.0, 1.0])
            model.item_embedding.weight[4] = torch.tensor([1.0, 0.0])
            model.item_embedding.weight[5] = torch.tensor([0.0, 1.0])
            model.item_embedding.weight[6] = torch.tensor([-1.0, 0.0])
        histories = torch.tensor([[1, 2, 0], [3, 0, 0]])
        pooled = model.encode_user(histories)
        expected = torch.tensor([[2 ** -0.5, 2 ** -0.5], [2 ** -0.5, 2 ** -0.5]])
        self.assertTrue(torch.allclose(pooled, expected, atol=1e-6))
        logits = model(histories, torch.tensor([3, 3]), torch.tensor([[4, 5], [5, 6]]))
        self.assertEqual((2, 3), tuple(logits.shape))
        loss = model.sampled_softmax_loss(histories, torch.tensor([3, 3]), torch.tensor([[4, 5], [5, 6]]))
        loss.backward()
        self.assertTrue(model.item_embedding.weight.grad.is_sparse)
        before = model.item_embedding.weight.detach().clone()
        torch.optim.SparseAdam(model.parameters(), lr=0.01).step()
        self.assertFalse(torch.equal(before, model.item_embedding.weight.detach()))

    def test_device_fallback(self):
        if not torch.cuda.is_available():
            self.assertEqual("cpu", select_device("cuda").type)
        self.assertEqual("cpu", select_device("cpu").type)


if __name__ == "__main__":
    unittest.main()
