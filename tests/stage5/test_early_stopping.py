from __future__ import annotations

import unittest

from src.recall.early_stopping import ValidationLossEarlyStopping
from scripts.stage5.stage5_3_train_two_tower import replace_resume_baseline


class ValidationLossEarlyStoppingTest(unittest.TestCase):
    def test_legacy_resume_history_gets_complete_validation_baseline(self):
        rows = [
            {"epoch": "2", "validation_loss": "2.1"},
            {"epoch": "3", "validation_loss": "2.0"},
        ]
        replace_resume_baseline(rows, 3, 1.95, 284899, "complete_train_seen")

        self.assertEqual(rows[0]["validation_scope"], "legacy_limited_train_seen")
        self.assertEqual(rows[1]["validation_loss"], 1.95)
        self.assertEqual(rows[1]["validation_samples"], 284899)
        self.assertEqual(rows[1]["validation_scope"], "complete_train_seen")

    def test_stops_after_patience_without_meaningful_improvement(self):
        stopper = ValidationLossEarlyStopping(patience=2, min_delta=0.002)
        first = stopper.update(2.0000, 1)
        small = stopper.update(1.9990, 2)
        stop = stopper.update(1.9985, 3)

        self.assertTrue(first.significant_improvement)
        self.assertTrue(small.is_best)
        self.assertFalse(small.significant_improvement)
        self.assertFalse(small.should_stop)
        self.assertTrue(stop.is_best)
        self.assertTrue(stop.should_stop)
        self.assertEqual(stopper.best_epoch, 3)

    def test_meaningful_improvement_resets_patience(self):
        stopper = ValidationLossEarlyStopping(patience=2, min_delta=0.002)
        stopper.update(2.0000, 1)
        stopper.update(1.9990, 2)
        decision = stopper.update(1.9970, 3)

        self.assertTrue(decision.significant_improvement)
        self.assertFalse(decision.should_stop)
        self.assertEqual(stopper.bad_epochs, 0)

    def test_sub_threshold_improvements_do_not_accumulate(self):
        stopper = ValidationLossEarlyStopping(patience=2, min_delta=0.002)
        stopper.update(2.0000, 1)
        first_small = stopper.update(1.9989, 2)
        second_small = stopper.update(1.9978, 3)

        self.assertFalse(first_small.significant_improvement)
        self.assertFalse(second_small.significant_improvement)
        self.assertTrue(second_small.should_stop)
        self.assertEqual(stopper.best_epoch, 3)

    def test_resume_uses_saved_previous_epoch_loss(self):
        original = ValidationLossEarlyStopping(patience=2, min_delta=0.002)
        original.update(2.0000, 1)
        original.update(1.9990, 2)
        resumed = ValidationLossEarlyStopping.resume(
            original.state_dict(), patience=2, min_delta=0.002,
            checkpoint_loss=1.9990, checkpoint_epoch=2,
        )
        decision = resumed.update(1.9975, 3)

        self.assertFalse(decision.significant_improvement)
        self.assertTrue(decision.should_stop)

    def test_resume_uses_checkpoint_loss_for_legacy_checkpoint(self):
        stopper = ValidationLossEarlyStopping.resume(
            {"reference_loss": 2.100, "bad_epochs": 0},
            patience=2, min_delta=0.002,
            checkpoint_loss=2.046, checkpoint_epoch=3,
        )
        decision = stopper.update(2.045, 4)

        self.assertTrue(decision.is_best)
        self.assertFalse(decision.significant_improvement)
        self.assertEqual(stopper.best_epoch, 4)
        self.assertEqual(stopper.bad_epochs, 1)


if __name__ == "__main__":
    unittest.main()
