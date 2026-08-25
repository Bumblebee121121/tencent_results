from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import torch

from src.models.vanilla_two_tower import VanillaTwoTower
from src.recall.checkpoint import (
    CpuCheckpointBuffer, load_checkpoint, load_model_checkpoint, load_training_checkpoint,
    save_checkpoint,
)
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
            save_checkpoint(
                path, model, optimizer, 1, {"two_tower": {}}, {"recall": "v1"}, 0.5,
                training_state={"bad_epochs": 1},
            )
            restored = VanillaTwoTower(10, 4)
            restored_optimizer = torch.optim.SparseAdam(restored.parameters(), lr=0.01)
            checkpoint = load_checkpoint(path, restored, restored_optimizer)
            after = restored.encode_user(history).detach()
            self.assertTrue(torch.equal(before, after))
            self.assertIn("optimizer", checkpoint)
            self.assertIn("random_state", checkpoint)
            self.assertEqual(checkpoint["training_state"]["bad_epochs"], 1)

    def test_training_load_returns_metadata_without_large_state_dicts(self):
        model = VanillaTwoTower(10, 4)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
        with workspace_tempdir() as temporary:
            path = temporary / "best.pt"
            save_checkpoint(path, model, optimizer, 3, {}, {"recall": "v1"}, 0.4)
            restored = VanillaTwoTower(10, 4)
            restored_optimizer = torch.optim.SparseAdam(restored.parameters(), lr=0.01)

            metadata = load_training_checkpoint(path, restored, restored_optimizer)

            self.assertEqual(metadata["epoch"], 3)
            self.assertEqual(metadata["protocols"], {"recall": "v1"})
            self.assertNotIn("model", metadata)
            self.assertNotIn("optimizer", metadata)

    def test_model_load_returns_metadata_without_large_state_dicts(self):
        model = VanillaTwoTower(10, 4)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
        before = model.item_embedding.weight.detach().clone()
        with workspace_tempdir() as temporary:
            path = temporary / "best.pt"
            save_checkpoint(path, model, optimizer, 2, {}, {"recall": "v1"}, 0.45)
            restored = VanillaTwoTower(10, 4)

            metadata = load_model_checkpoint(path, restored)

            self.assertTrue(torch.equal(before, restored.item_embedding.weight))
            self.assertEqual(metadata["epoch"], 2)
            self.assertNotIn("model", metadata)
            self.assertNotIn("optimizer", metadata)

    def test_model_only_checkpoint_rejects_training_resume(self):
        model = VanillaTwoTower(10, 4)
        with workspace_tempdir() as temporary:
            path = temporary / "best.pt"
            save_checkpoint(path, model, None, 2, {}, {}, 0.45)
            restored = VanillaTwoTower(10, 4)
            optimizer = torch.optim.SparseAdam(restored.parameters(), lr=0.01)

            metadata = load_model_checkpoint(path, restored)
            self.assertEqual(metadata["epoch"], 2)
            with self.assertRaisesRegex(ValueError, "does not contain optimizer"):
                load_training_checkpoint(path, restored, optimizer)

    def test_mmap_resume_releases_file_before_atomic_model_only_replace(self):
        model = VanillaTwoTower(10, 4)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
        with workspace_tempdir() as temporary:
            path = temporary / "best.pt"
            save_checkpoint(path, model, optimizer, 2, {}, {}, 0.45)
            restored = VanillaTwoTower(10, 4)
            restored_optimizer = torch.optim.SparseAdam(restored.parameters(), lr=0.01)

            load_training_checkpoint(path, restored, restored_optimizer)
            save_checkpoint(path, restored, None, 3, {}, {}, 0.40)

            metadata = load_model_checkpoint(path, VanillaTwoTower(10, 4))
            self.assertEqual(metadata["epoch"], 3)

    def test_failed_atomic_save_preserves_previous_checkpoint(self):
        model = VanillaTwoTower(10, 4)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
        with workspace_tempdir() as temporary:
            path = temporary / "best.pt"
            path.write_bytes(b"known-good-checkpoint")

            def fail_after_partial_write(_payload, temporary_path):
                Path(temporary_path).write_bytes(b"partial")
                raise RuntimeError("simulated serialization failure")

            with mock.patch("src.recall.checkpoint.torch.save", side_effect=fail_after_partial_write):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    save_checkpoint(path, model, optimizer, 4, {}, {}, 0.3)

            self.assertEqual(path.read_bytes(), b"known-good-checkpoint")
            self.assertEqual(list(temporary.glob(".*.tmp")), [])

    def test_cpu_checkpoint_buffer_reuses_tensor_storage(self):
        model = VanillaTwoTower(10, 4)
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=0.01)
        history = torch.tensor([[2, 3, 0]])
        loss = model.sampled_softmax_loss(
            history, torch.tensor([4]), torch.tensor([[5, 6]]),
        )
        loss.backward()
        optimizer.step()
        buffer = CpuCheckpointBuffer()
        with workspace_tempdir() as temporary:
            path = temporary / "resume.pt"
            save_checkpoint(path, model, optimizer, 1, {}, {}, 0.5, cpu_buffer=buffer)
            self.assertEqual(buffer.tensor_count, 3)
            first_pointers = {
                key: tensor.data_ptr() for key, tensor in buffer._tensor_buffers.items()
            }
            first_bytes = buffer.allocated_bytes
            with torch.no_grad():
                model.item_embedding.weight[2].add_(1.0)

            save_checkpoint(path, model, optimizer, 2, {}, {}, 0.4, cpu_buffer=buffer)

            self.assertEqual(first_bytes, buffer.allocated_bytes)
            self.assertEqual(
                first_pointers,
                {key: tensor.data_ptr() for key, tensor in buffer._tensor_buffers.items()},
            )
            checkpoint = load_checkpoint(path)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertTrue(torch.equal(
                checkpoint["model"]["item_embedding.weight"],
                model.item_embedding.weight.detach().cpu(),
            ))
            restored = VanillaTwoTower(10, 4)
            restored_optimizer = torch.optim.SparseAdam(restored.parameters(), lr=0.01)
            load_training_checkpoint(path, restored, restored_optimizer)
            self.assertEqual(1, len(restored_optimizer.state))
            restored_state = next(iter(restored_optimizer.state.values()))
            self.assertEqual({"step", "exp_avg", "exp_avg_sq"}, set(restored_state))


if __name__ == "__main__":
    unittest.main()
