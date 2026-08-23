from __future__ import annotations

import unittest
from pathlib import Path

import torch

from src.models.vanilla_two_tower import VanillaTwoTower
from src.recall.checkpoint import load_checkpoint, save_checkpoint
from tests.stage5._temp import workspace_tempdir


class CheckpointTest(unittest.TestCase):
    def test_restore_produces_identical_output(self):
        torch.manual_seed(3)
        model = VanillaTwoTower(10, 4)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
        history = torch.tensor([[2, 3, 0]])
        before = model.encode_user(history).detach().clone()
        with workspace_tempdir() as temporary:
            path = temporary / "best.pt"
            save_checkpoint(path, model, optimizer, 1, {"two_tower": {}}, {"recall": "v1"}, 0.5)
            restored = VanillaTwoTower(10, 4)
            restored_optimizer = torch.optim.SparseAdam(restored.parameters(), lr=0.01)
            checkpoint = load_checkpoint(path, restored, restored_optimizer)
            after = restored.encode_user(history).detach()
            self.assertTrue(torch.equal(before, after))
            self.assertIn("optimizer", checkpoint)
            self.assertIn("random_state", checkpoint)


if __name__ == "__main__":
    unittest.main()
