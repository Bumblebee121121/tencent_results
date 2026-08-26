"""ID, structured side-feature and multimodal item representations."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .history_strength_gate import HistoryStrengthGate


class ContentItemTower(nn.Module):
    VALID_VARIANTS = {"U1", "U2", "U3", "I1", "I2", "I3", "E1"}

    def __init__(
        self,
        item_embedding: nn.Embedding,
        variant: str,
        side_vocab_sizes: Sequence[int],
        embedding_dim: int = 64,
        side_embedding_dim: int = 8,
        mm_input_dim: int = 32,
    ) -> None:
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"unknown Stage 6 variant: {variant}")
        self.item_embedding = item_embedding
        self.variant = variant
        self.uses_side = variant in {"I1", "I3", "E1"}
        self.uses_mm = variant in {"I2", "I3", "E1"}
        self.uses_strength_gate = variant == "E1"
        self.side_embeddings = nn.ModuleList(
            [nn.Embedding(int(size), side_embedding_dim, padding_idx=0, sparse=True) for size in side_vocab_sizes]
        ) if self.uses_side else nn.ModuleList()
        self.side_projection = (
            nn.Sequential(
                nn.Linear(len(side_vocab_sizes) * side_embedding_dim, embedding_dim),
                nn.ReLU(), nn.Linear(embedding_dim, embedding_dim),
            ) if self.uses_side else None
        )
        self.mm_projection = (
            nn.Sequential(nn.Linear(mm_input_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
            if self.uses_mm else None
        )
        self.mm_missing = nn.Parameter(torch.zeros(embedding_dim)) if self.uses_mm else None
        non_id_parts = int(self.uses_side) + int(self.uses_mm)
        self.non_id_projection = (
            nn.Sequential(nn.Linear(non_id_parts * embedding_dim, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
            if non_id_parts else None
        )
        self.fixed_fusion = (
            nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim), nn.ReLU(), nn.Linear(embedding_dim, embedding_dim))
            if non_id_parts and not self.uses_strength_gate else None
        )
        self.strength_gate = HistoryStrengthGate() if self.uses_strength_gate else None

    def encode_non_id(
        self,
        side_tokens: torch.Tensor | None,
        mm: torch.Tensor | None,
        mm_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = []
        if self.uses_side:
            if side_tokens is None or side_tokens.ndim != 2 or side_tokens.shape[1] != len(self.side_embeddings):
                raise ValueError("side-aware item tower requires one token per configured field")
            side = torch.cat(
                [embedding(side_tokens[:, index]) for index, embedding in enumerate(self.side_embeddings)], dim=-1,
            )
            parts.append(self.side_projection(side))
        if self.uses_mm:
            if mm is None or mm_valid is None:
                raise ValueError("MM-aware item tower requires vectors and validity flags")
            if not torch.isfinite(mm).all():
                raise ValueError("MM vectors contain NaN or Inf")
            projected = self.mm_projection(mm)
            valid = mm_valid.to(dtype=projected.dtype).unsqueeze(-1)
            parts.append(valid * projected + (1.0 - valid) * self.mm_missing)
        if not parts:
            raise ValueError("pure-ID variants do not have a non-ID representation")
        return self.non_id_projection(torch.cat(parts, dim=-1))

    def forward(
        self,
        item_tokens: torch.Tensor,
        side_tokens: torch.Tensor | None = None,
        mm: torch.Tensor | None = None,
        mm_valid: torch.Tensor | None = None,
        train_counts: torch.Tensor | None = None,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        item_id = self.item_embedding(item_tokens)
        gate = None
        if not (self.uses_side or self.uses_mm):
            output = F.normalize(item_id, p=2, dim=-1)
        else:
            non_id = self.encode_non_id(side_tokens, mm, mm_valid)
            if self.uses_strength_gate:
                if train_counts is None:
                    raise ValueError("E1 requires train-only item counts")
                gate = self.strength_gate(train_counts).unsqueeze(-1)
                output = F.normalize(gate * item_id + (1.0 - gate) * non_id, p=2, dim=-1)
            else:
                output = F.normalize(self.fixed_fusion(torch.cat([item_id, non_id], dim=-1)), p=2, dim=-1)
        return (output, None if gate is None else gate.squeeze(-1)) if return_gate else output

