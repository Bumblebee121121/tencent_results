"""Interpretable monotonic train-history strength gate."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class HistoryStrengthGate(nn.Module):
    """Scalar ID weight, monotonic in train-only event count and zero for unseen."""

    def __init__(self, initial_slope: float = 0.0, initial_bias: float = 0.0) -> None:
        super().__init__()
        self.raw_slope = nn.Parameter(torch.tensor(float(initial_slope)))
        self.bias = nn.Parameter(torch.tensor(float(initial_bias)))

    def forward(self, train_counts: torch.Tensor) -> torch.Tensor:
        counts = train_counts.to(dtype=self.raw_slope.dtype)
        if torch.any(counts < 0):
            raise ValueError("train history counts cannot be negative")
        gate = torch.sigmoid(F.softplus(self.raw_slope) * torch.log1p(counts) + self.bias)
        return gate * counts.gt(0).to(gate.dtype)

