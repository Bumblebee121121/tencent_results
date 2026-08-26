"""Composable Stage 6 two-tower variants B0-excluded U1 through E1."""

from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .content_item_tower import ContentItemTower
from .sdm_user_tower import SDMUserTower


USER_VARIANT = {"U1": "U1", "U2": "U2", "U3": "U3", "I1": "U3", "I2": "U3", "I3": "U3", "E1": "U3"}


class EnhancedTwoTower(nn.Module):
    def __init__(
        self,
        variant: str,
        num_item_tokens: int,
        side_vocab_sizes: Sequence[int] = (),
        embedding_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        side_embedding_dim: int = 8,
        mm_input_dim: int = 32,
        time_feature_dim: int = 3,
        time_hidden_dim: int = 32,
        padding_idx: int = 0,
        unknown_idx: int = 1,
    ) -> None:
        super().__init__()
        if variant not in USER_VARIANT:
            raise ValueError(f"unknown Stage 6 variant: {variant}")
        self.variant = variant
        self.item_embedding = nn.Embedding(num_item_tokens, embedding_dim, padding_idx=padding_idx, sparse=True)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[padding_idx].zero_()
        user_variant = USER_VARIANT[variant]
        self.user_tower = SDMUserTower(
            self.item_embedding, embedding_dim, num_heads, dropout,
            use_action=user_variant in {"U2", "U3"}, use_time=user_variant == "U3",
            time_feature_dim=time_feature_dim, time_hidden_dim=time_hidden_dim,
            padding_idx=padding_idx, unknown_idx=unknown_idx,
        )
        self.item_tower = ContentItemTower(
            self.item_embedding, variant, side_vocab_sizes, embedding_dim,
            side_embedding_dim, mm_input_dim,
        )

    def encode_user(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.user_tower(
            batch["short_tokens"], batch["long_tokens"],
            batch["short_padding_mask"], batch["long_padding_mask"],
            batch.get("short_actions"), batch.get("long_actions"),
            batch.get("short_time_features"), batch.get("long_time_features"),
        )

    def encode_item(self, item_batch: Mapping[str, torch.Tensor], return_gate: bool = False):
        return self.item_tower(
            item_batch["item_tokens"], item_batch.get("side_tokens"),
            item_batch.get("mm"), item_batch.get("mm_valid"),
            item_batch.get("train_counts"), return_gate=return_gate,
        )

    def sampled_softmax_loss(
        self,
        user_batch: Mapping[str, torch.Tensor],
        positive_batch: Mapping[str, torch.Tensor],
        negative_batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        user = self.encode_user(user_batch)
        positive = self.encode_item(positive_batch).unsqueeze(1)
        batch_size, negative_count = negative_batch["item_tokens"].shape
        flat = {
            key: value.reshape(batch_size * negative_count, *value.shape[2:])
            for key, value in negative_batch.items()
        }
        negative = self.encode_item(flat).reshape(batch_size, negative_count, -1)
        logits = torch.cat([
            torch.einsum("bd,bnd->bn", user, positive),
            torch.einsum("bd,bnd->bn", user, negative),
        ], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=user.device)
        return F.cross_entropy(logits, labels)

