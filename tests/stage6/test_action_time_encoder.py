import unittest
import torch

from src.models.action_time_encoder import ActionTimeEncoder


class ActionTimeEncoderTest(unittest.TestCase):
    def test_pad_event_contributes_zero(self):
        model = ActionTimeEncoder(8, True, True, time_hidden_dim=4)
        items = torch.ones((1, 2, 8)); actions = torch.tensor([[2, 0]])
        time = torch.zeros((1, 2, 3)); mask = torch.tensor([[False, True]])
        output = model(items, actions, time, mask)
        torch.testing.assert_close(output[:, 1], torch.zeros((1, 8)))

    def test_nonfinite_time_fails(self):
        model = ActionTimeEncoder(4, False, True)
        with self.assertRaisesRegex(ValueError, "NaN"):
            model(torch.zeros((1, 1, 4)), normalized_time_features=torch.tensor([[[float("nan"), 0, 0]]]))

