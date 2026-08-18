"""Unified read API for Stage 4 stores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from src.data.item_strength import classify_strength

from .id_semantics import encode_item_rid
from .sequence_store import HistorySlice, slice_history


class FeatureStore:
    """Memory-mapped access to item, sequence and multimodal features."""

    def __init__(self, root: Path, manifest_override: Mapping[str, object] | None = None):
        self.root = Path(root)
        feature_dir = self.root / "feature_store"
        mapping_dir = self.root / "mappings"
        manifest_path = self.root / "manifests" / "stage4_manifest.json"
        if manifest_override is None:
            with manifest_path.open("r", encoding="utf-8") as handle:
                self.manifest = json.load(handle)
        else:
            self.manifest = dict(manifest_override)
        self.train_counts = np.load(
            mapping_dir / "train_item_count_by_rid.npy", mmap_mode="r", allow_pickle=False
        )
        self.rid_to_token = np.load(
            mapping_dir / "rid_to_model_item_token.npy", mmap_mode="r", allow_pickle=False
        )
        self.item_side = np.load(
            feature_dir / "item_side_tokens_by_rid.npy", mmap_mode="r", allow_pickle=False
        )
        self.item_side_missing = np.load(
            feature_dir / "item_side_missing_by_rid.npy", mmap_mode="r", allow_pickle=False
        )
        self.item_side_oov = np.load(
            feature_dir / "item_side_oov_by_rid.npy", mmap_mode="r", allow_pickle=False
        )
        self.mm_by_rid = np.load(
            feature_dir / "mm_by_rid.npy", mmap_mode="r", allow_pickle=False
        )
        self.mm_valid_by_rid = np.load(
            feature_dir / "mm_valid_by_rid.npy", mmap_mode="r", allow_pickle=False
        )
        self.user_seq_offsets = np.load(
            feature_dir / "user_seq_offsets.npy", mmap_mode="r", allow_pickle=False
        )
        self.seq_item_rid = np.load(
            feature_dir / "seq_item_rid.npy", mmap_mode="r", allow_pickle=False
        )
        self.seq_action_token = np.load(
            feature_dir / "seq_action_token.npy", mmap_mode="r", allow_pickle=False
        )
        self.seq_timestamp = np.load(
            feature_dir / "seq_timestamp.npy", mmap_mode="r", allow_pickle=False
        )
        self.p50_train = float(self.manifest["p50_train"])
        self.p90_train = float(self.manifest["p90_train"])
        self.mm_dim = int(self.manifest["mm_dim"])

    def item_token(self, item_rid: int | None) -> int:
        return encode_item_rid(item_rid, rid_to_token=self.rid_to_token)

    def item_train_strength(self, item_rid: int | None) -> dict[str, object]:
        count = (
            int(self.train_counts[int(item_rid)])
            if item_rid is not None and 0 < int(item_rid) < self.train_counts.size
            else 0
        )
        return {
            "target_train_count": count,
            "target_train_count_log1p": float(np.log1p(count)),
            "target_strength_group": classify_strength(
                count, self.p50_train, self.p90_train
            ),
        }

    def item_side_features(self, item_rid: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if item_rid is None or not 0 < int(item_rid) < self.item_side.shape[0]:
            width = self.item_side.shape[1]
            return (
                np.full(width, 1, dtype=np.int32),
                np.ones(width, dtype=np.bool_),
                np.zeros(width, dtype=np.bool_),
            )
        rid = int(item_rid)
        return (
            np.asarray(self.item_side[rid]),
            np.asarray(self.item_side_missing[rid]),
            np.asarray(self.item_side_oov[rid]),
        )

    def item_mm(self, item_rid: int | None) -> tuple[np.ndarray, bool]:
        if item_rid is None or not 0 < int(item_rid) < self.mm_by_rid.shape[0]:
            return np.zeros(self.mm_dim, dtype=np.float32), False
        rid = int(item_rid)
        return np.asarray(self.mm_by_rid[rid]), bool(self.mm_valid_by_rid[rid])

    def history(
        self,
        user_id: int,
        history_end_position: int,
        target_timestamp: int,
    ) -> HistorySlice:
        return slice_history(
            user_id,
            history_end_position,
            self.user_seq_offsets,
            self.seq_item_rid,
            self.seq_action_token,
            self.seq_timestamp,
            target_timestamp,
        )

    def history_item_tokens(self, history: HistorySlice) -> np.ndarray:
        rids = np.asarray(history.item_rid, dtype=np.int64)
        if np.any((rids <= 0) | (rids >= self.rid_to_token.size)):
            raise ValueError("history item RID is outside the model-token mapping")
        return np.asarray(self.rid_to_token[rids], dtype=np.int32)
