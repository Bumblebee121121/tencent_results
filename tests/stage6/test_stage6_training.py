import unittest
import torch

from src.recall.stage6_training import build_model, partition_sparse_dense_parameters


CONFIG = {
    "model": {"embedding_dim": 8},
    "user_tower": {"attention": {"num_heads": 2, "dropout": 0.0}, "time_feature_dim": 3, "time_hidden_dim": 4},
    "item_tower": {"side_embedding_dim": 2, "mm_input_dim": 4},
}


class TrainingTest(unittest.TestCase):
    def test_sparse_dense_partition_disjoint(self):
        model = build_model("E1", CONFIG, 20, [6, 7])
        sparse, dense = partition_sparse_dense_parameters(model)
        self.assertTrue(sparse); self.assertTrue(dense)
        self.assertFalse(set(map(id, sparse)) & set(map(id, dense)))

    def test_e1_sampled_softmax_end_to_end(self):
        model = build_model("E1", CONFIG, 20, [6, 7])
        user = {
            "short_tokens": torch.tensor([[2, 3], [4, 0]]),
            "long_tokens": torch.tensor([[5, 0], [6, 7]]),
            "short_padding_mask": torch.tensor([[False, False], [False, True]]),
            "long_padding_mask": torch.tensor([[False, True], [False, False]]),
            "short_actions": torch.tensor([[2, 1], [1, 0]]),
            "long_actions": torch.tensor([[1, 0], [2, 1]]),
            "short_time_features": torch.zeros((2, 2, 3)),
            "long_time_features": torch.zeros((2, 2, 3)),
        }
        positive = {"item_tokens": torch.tensor([8, 9]), "side_tokens": torch.tensor([[3, 4], [4, 5]]),
                    "mm": torch.ones((2, 4)), "mm_valid": torch.tensor([True, False]), "train_counts": torch.tensor([2., 0.])}
        negative = {"item_tokens": torch.tensor([[10, 11], [12, 13]]),
                    "side_tokens": torch.tensor([[[3, 4], [4, 5]], [[3, 5], [5, 4]]]),
                    "mm": torch.ones((2, 2, 4)), "mm_valid": torch.ones((2, 2), dtype=torch.bool),
                    "train_counts": torch.ones((2, 2))}
        loss = model.sampled_softmax_loss(user, positive, negative)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
