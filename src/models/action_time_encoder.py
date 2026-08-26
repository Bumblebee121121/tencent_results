"""Action and continuous-time event encoders for Stage 6 user variants."""

from __future__ import annotations

import torch
from torch import nn


class ActionTimeEncoder(nn.Module):
    """Add optional action and normalized continuous-time signals to item events."""

    def __init__(
        self,
        embedding_dim: int,
        use_action: bool,
        use_time: bool,
        num_action_tokens: int = 4,
        time_feature_dim: int = 3,
        time_hidden_dim: int = 32,
        action_padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.use_action = bool(use_action)
        self.use_time = bool(use_time)
        self.action_embedding = (
            nn.Embedding(num_action_tokens, embedding_dim, padding_idx=action_padding_idx)
            if self.use_action
            else None
        )
        self.time_mlp = (
            nn.Sequential(
                nn.Linear(time_feature_dim, time_hidden_dim),
                nn.ReLU(),
                nn.Linear(time_hidden_dim, embedding_dim),
            )
            if self.use_time
            else None
        )
        if self.action_embedding is not None:
            nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)
            with torch.no_grad():
                self.action_embedding.weight[action_padding_idx].zero_()

    def forward(
        self,
        item_events: torch.Tensor,
        action_tokens: torch.Tensor | None = None,
        normalized_time_features: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        result = item_events
        if self.use_action:
            if action_tokens is None:
                raise ValueError("action-aware variant requires action tokens")
            result = result + self.action_embedding(action_tokens)
        if self.use_time:
            if normalized_time_features is None:
                raise ValueError("time-aware variant requires normalized time features")
            if not torch.isfinite(normalized_time_features).all():
                raise ValueError("time features contain NaN or Inf")
            result = result + self.time_mlp(normalized_time_features)
        if padding_mask is not None:
            result = result.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return result

