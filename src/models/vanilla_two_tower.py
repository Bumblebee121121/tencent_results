"""Pure-ID PyTorch two-tower baseline with one shared item embedding table."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class VanillaTwoTower(nn.Module):
    def __init__(
        self,
        num_item_tokens: int,
        embedding_dim: int = 64,
        padding_idx: int = 0,
        unknown_idx: int = 1,
    ):
        super().__init__()
        self.padding_idx = int(padding_idx)
        self.unknown_idx = int(unknown_idx)
        if self.padding_idx == self.unknown_idx:
            raise ValueError("PAD and UNK token IDs must be distinct")
        self.item_embedding = nn.Embedding(
            int(num_item_tokens), int(embedding_dim), padding_idx=self.padding_idx, sparse=True
        )
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[self.padding_idx].zero_()

    def encode_user(
        self, history_tokens: torch.Tensor, history_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pure-ID Stage 5 has no reliable learning signal for the shared UNK embedding.
        if history_offsets is not None:
            if history_tokens.ndim != 1 or history_offsets.ndim != 1:
                raise ValueError("ragged histories require 1-D tokens and offsets")
            pooled = F.embedding_bag(
                history_tokens, self.item_embedding.weight, history_offsets,
                mode="mean", sparse=True, include_last_offset=True,
            )
            return F.normalize(pooled, p=2, dim=-1)
        mask = history_tokens.ne(self.padding_idx) & history_tokens.ne(self.unknown_idx)
        embedded = self.item_embedding(history_tokens)
        pooled = (embedded * mask.unsqueeze(-1)).sum(dim=1)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1)
        return F.normalize(pooled / denominator, p=2, dim=-1)

    def encode_item(self, item_tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_embedding(item_tokens), p=2, dim=-1)

    def forward(
        self, history_tokens: torch.Tensor, positive_tokens: torch.Tensor,
        negative_tokens: torch.Tensor, history_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user = self.encode_user(history_tokens, history_offsets)
        positive = self.encode_item(positive_tokens).unsqueeze(1)
        negative = self.encode_item(negative_tokens)
        candidates = torch.cat([positive, negative], dim=1)
        return torch.einsum("bd,bnd->bn", user, candidates)

    def sampled_softmax_loss(
        self, history_tokens: torch.Tensor, positive_tokens: torch.Tensor,
        negative_tokens: torch.Tensor, history_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = self(history_tokens, positive_tokens, negative_tokens, history_offsets)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)
