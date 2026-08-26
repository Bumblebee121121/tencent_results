"""Lightweight SDM-style long/short-interest user tower."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .action_time_encoder import ActionTimeEncoder


def _masked_mean(values: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~padding_mask).unsqueeze(-1)
    return (values * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


class SDMUserTower(nn.Module):
    """One-layer short self-attention, query-aware long attention and gated fusion."""

    def __init__(
        self,
        item_embedding: nn.Embedding,
        embedding_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_action: bool = False,
        use_time: bool = False,
        time_feature_dim: int = 3,
        time_hidden_dim: int = 32,
        padding_idx: int = 0,
        unknown_idx: int = 1,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.item_embedding = item_embedding
        self.padding_idx = int(padding_idx)
        self.unknown_idx = int(unknown_idx)
        self.event_encoder = ActionTimeEncoder(
            embedding_dim, use_action, use_time,
            time_feature_dim=time_feature_dim, time_hidden_dim=time_hidden_dim,
        )
        self.short_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.long_key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.fusion_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.scale = embedding_dim ** -0.5

    def _encode_events(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor | None,
        time_features: torch.Tensor | None,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        embedded = self.item_embedding(tokens)
        return self.event_encoder(embedded, actions, time_features, padding_mask)

    def forward(
        self,
        short_tokens: torch.Tensor,
        long_tokens: torch.Tensor,
        short_padding_mask: torch.Tensor,
        long_padding_mask: torch.Tensor,
        short_actions: torch.Tensor | None = None,
        long_actions: torch.Tensor | None = None,
        short_time_features: torch.Tensor | None = None,
        long_time_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        empty_short = short_padding_mask.all(dim=1)
        safe_short_mask = short_padding_mask.clone()
        if empty_short.any():
            safe_short_mask[empty_short, 0] = False
        short_events = self._encode_events(
            short_tokens, short_actions, short_time_features, short_padding_mask,
        )
        attended, _ = self.short_attention(
            short_events, short_events, short_events,
            key_padding_mask=safe_short_mask,
            need_weights=False,
        )
        short_interest = _masked_mean(attended + short_events, safe_short_mask)
        short_interest = short_interest.masked_fill(empty_short.unsqueeze(-1), 0.0)

        long_events = self._encode_events(
            long_tokens, long_actions, long_time_features, long_padding_mask,
        )
        scores = torch.einsum(
            "bd,bld->bl", short_interest, self.long_key(long_events),
        ) * self.scale
        scores = scores.masked_fill(long_padding_mask, -torch.inf)
        empty_long = long_padding_mask.all(dim=1)
        if empty_long.any():
            scores = scores.clone()
            scores[empty_long] = 0.0
        weights = torch.softmax(scores, dim=1).masked_fill(long_padding_mask, 0.0)
        long_interest = torch.einsum("bl,bld->bd", weights, long_events)
        long_interest = torch.where(empty_long.unsqueeze(-1), short_interest, long_interest)

        gate = torch.sigmoid(self.fusion_gate(torch.cat([short_interest, long_interest], dim=-1)))
        output = F.normalize(gate * short_interest + (1.0 - gate) * long_interest, p=2, dim=-1)
        return output.masked_fill(empty_short.unsqueeze(-1), 0.0)
