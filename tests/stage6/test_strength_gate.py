import unittest
import torch

from src.models.history_strength_gate import HistoryStrengthGate


class StrengthGateTest(unittest.TestCase):
    def test_unseen_zero_and_monotonic(self):
        gate = HistoryStrengthGate()
        values = gate(torch.tensor([0., 1., 2., 22., 100.]))
        self.assertEqual(float(values[0].detach()), 0.0)
        self.assertTrue(bool(torch.all(values[2:] >= values[1:-1])))

    def test_negative_count_fails(self):
        with self.assertRaisesRegex(ValueError, "negative"):
            HistoryStrengthGate()(torch.tensor([-1.]))
