"""Stage 6 data adapter over frozen Stage 3 samples and Stage 4 stores."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.features.feature_store import FeatureStore
from src.features.next_click_dataset import add_dynamic_time_features


SAMPLE_COLUMNS = [
    "sample_id", "user_id", "target_item_rid", "target_item_oid", "target_timestamp",
    "history_end_position", "history_length",
]


@dataclass(frozen=True)
class SessionSplit:
    long_indices: np.ndarray
    short_indices: np.ndarray
    long_truncated: bool
    short_truncated: bool


def split_long_short_session(
    timestamps: Sequence[int] | np.ndarray,
    session_gap_seconds: int,
    short_max_events: int | None = None,
    long_max_events: int | None = None,
) -> SessionSplit:
    values = np.asarray(timestamps, dtype=np.int64)
    if session_gap_seconds <= 0:
        raise ValueError("session_gap_seconds must be positive")
    if values.size > 1 and np.any(np.diff(values) < 0):
        raise ValueError("history timestamps must be nondecreasing")
    boundary = 0
    for index in range(values.size - 1, 0, -1):
        if int(values[index] - values[index - 1]) > session_gap_seconds:
            boundary = index
            break
    long_indices = np.arange(0, boundary, dtype=np.int64)
    short_indices = np.arange(boundary, values.size, dtype=np.int64)
    long_truncated = long_max_events is not None and long_indices.size > long_max_events
    short_truncated = short_max_events is not None and short_indices.size > short_max_events
    if long_truncated:
        long_indices = long_indices[-int(long_max_events):]
    if short_truncated:
        short_indices = short_indices[-int(short_max_events):]
    return SessionSplit(long_indices, short_indices, long_truncated, short_truncated)


@dataclass(frozen=True)
class TimeNormalization:
    mean: np.ndarray
    std: np.ndarray
    fit_split: str = "train"

    def __post_init__(self) -> None:
        if np.asarray(self.mean).shape != (3,) or np.asarray(self.std).shape != (3,):
            raise ValueError("time normalization requires three features")
        if np.any(np.asarray(self.std) <= 0):
            raise ValueError("time standard deviations must be positive")
        if self.fit_split != "train":
            raise ValueError("time normalization may only be fit on train")

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((np.asarray(features, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def to_json(self) -> dict[str, object]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "fit_split": self.fit_split}

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> "TimeNormalization":
        return cls(np.asarray(value["mean"], dtype=np.float32), np.asarray(value["std"], dtype=np.float32), str(value["fit_split"]))


def fit_time_normalization(items: Iterable[Mapping[str, object]]) -> TimeNormalization:
    count = 0
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    for item in items:
        enriched = add_dynamic_time_features(item)
        matrix = np.column_stack([
            enriched["hist_recency_log1p"], enriched["hist_time_gap_log1p"],
            np.asarray(enriched["hist_first_event_mask"], dtype=np.float32),
        ]).astype(np.float64)
        count += matrix.shape[0]
        total += matrix.sum(axis=0)
        total_sq += np.square(matrix).sum(axis=0)
    if count == 0:
        raise ValueError("cannot fit time normalization without train history events")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1e-12)
    return TimeNormalization(mean.astype(np.float32), np.sqrt(variance).astype(np.float32))


class Stage6ParquetDataset(IterableDataset):
    def __init__(self, path: Path, feature_store: FeatureStore, max_rows: int | None = None, batch_size: int = 8192):
        super().__init__()
        self.path = Path(path)
        self.store = feature_store
        self.max_rows = max_rows
        self.batch_size = int(batch_size)

    def __iter__(self) -> Iterator[dict[str, object]]:
        worker = get_worker_info()
        worker_id, workers = (0, 1) if worker is None else (worker.id, worker.num_workers)
        emitted = 0
        scanner = ds.dataset(self.path, format="parquet").scanner(columns=SAMPLE_COLUMNS, batch_size=self.batch_size)
        for batch_number, batch in enumerate(scanner.to_batches()):
            if batch_number % workers != worker_id:
                continue
            for row in batch.to_pylist():
                if self.max_rows is not None and emitted >= self.max_rows:
                    return
                history = self.store.history(int(row["user_id"]), int(row["history_end_position"]), int(row["target_timestamp"]))
                if history.length != int(row["history_length"]):
                    raise ValueError("Stage 3 sample and Stage 4 sequence history lengths differ")
                side, missing, oov = self.store.item_side_features(row.get("target_item_rid"))
                mm, mm_valid = self.store.item_mm(row.get("target_item_rid"))
                item = {
                    **row,
                    "hist_item_rid": history.item_rid,
                    "hist_item_token": self.store.history_item_tokens(history),
                    "hist_action_token": history.action_token,
                    "hist_timestamp": history.timestamp,
                    "target_item_token": self.store.item_token(row.get("target_item_rid")),
                    "target_item_side": side, "target_item_side_missing": missing,
                    "target_item_side_oov": oov, "target_mm": mm, "target_mm_valid": mm_valid,
                }
                item.update(self.store.item_train_strength(row.get("target_item_rid")))
                emitted += 1
                yield add_dynamic_time_features(item)


class Stage6ItemStore:
    """RID-aligned train features plus candidate-row-aligned inference features."""

    def __init__(self, stage4_root: Path, side_fields: Sequence[str]):
        root = Path(stage4_root)
        feature = root / "feature_store"
        mapping = root / "mappings"
        self.side_fields = tuple(map(str, side_fields))
        with (mapping / "feature_vocab_manifest.json").open("r", encoding="utf-8") as handle:
            vocab = json.load(handle)
        if tuple(vocab["item"].keys()) != self.side_fields:
            raise ValueError("configured side field order differs from the Stage 4 manifest")
        self.side_vocab_sizes = tuple(int(vocab["item"][field]["train_known_vocab_size"]) + 3 for field in self.side_fields)
        self.side_by_rid = np.load(feature / "item_side_tokens_by_rid.npy", mmap_mode="r", allow_pickle=False)
        self.mm_by_rid = np.load(feature / "mm_by_rid.npy", mmap_mode="r", allow_pickle=False)
        self.mm_valid_by_rid = np.load(feature / "mm_valid_by_rid.npy", mmap_mode="r", allow_pickle=False)
        self.count_by_rid = np.load(mapping / "train_item_count_by_rid.npy", mmap_mode="r", allow_pickle=False)
        self.rid_to_token = np.load(mapping / "rid_to_model_item_token.npy", mmap_mode="r", allow_pickle=False)
        self.token_to_rid = np.zeros(int(self.rid_to_token.max()) + 1, dtype=np.int64)
        seen = np.flatnonzero(np.asarray(self.count_by_rid) > 0)
        self.token_to_rid[np.asarray(self.rid_to_token[seen], dtype=np.int64)] = seen

    def for_rids(self, rids: np.ndarray) -> dict[str, torch.Tensor]:
        values = np.asarray(rids, dtype=np.int64)
        if values.size and np.any((values <= 0) | (values >= self.rid_to_token.size)):
            raise ValueError("training item RID is outside Stage 4 stores")
        return {
            "item_tokens": torch.from_numpy(np.asarray(self.rid_to_token[values], dtype=np.int64)),
            "side_tokens": torch.from_numpy(np.asarray(self.side_by_rid[values], dtype=np.int64)),
            "mm": torch.from_numpy(np.asarray(self.mm_by_rid[values], dtype=np.float32)),
            "mm_valid": torch.from_numpy(np.asarray(self.mm_valid_by_rid[values], dtype=np.bool_)),
            "train_counts": torch.from_numpy(np.asarray(self.count_by_rid[values], dtype=np.float32)),
        }

    def for_tokens(self, tokens: np.ndarray) -> dict[str, torch.Tensor]:
        values = np.asarray(tokens, dtype=np.int64)
        if values.size and np.any((values <= 1) | (values >= self.token_to_rid.size)):
            raise ValueError("negative tokens must be Train-Seen")
        return self.for_rids(self.token_to_rid[values])


def _pad(values: list[np.ndarray], dtype, pad_value=0) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(value) for value in values), default=0)
    width = max(width, 1)
    output = np.full((len(values), width) + (() if values[0].ndim == 1 else values[0].shape[1:]), pad_value, dtype=dtype)
    mask = np.ones((len(values), width), dtype=np.bool_)
    for index, value in enumerate(values):
        output[index, :len(value)] = value
        mask[index, :len(value)] = False
    return torch.from_numpy(output), torch.from_numpy(mask)


@dataclass
class Stage6Collator:
    session_gap_seconds: int
    item_store: Stage6ItemStore
    time_normalization: TimeNormalization
    negative_sampler: object | None = None
    short_max_events: int | None = None
    long_max_events: int | None = None
    require_seen_target: bool = True

    def __call__(self, rows: list[dict[str, object]]) -> dict[str, object]:
        kept, short, long = [], [], []
        targets = []
        for row in rows:
            target = int(row["target_item_token"])
            if self.require_seen_target and target <= 1:
                continue
            split = split_long_short_session(row["hist_timestamp"], self.session_gap_seconds, self.short_max_events, self.long_max_events)
            tokens = np.asarray(row["hist_item_token"], dtype=np.int64)
            actions = np.asarray(row["hist_action_token"], dtype=np.int64)
            time_matrix = np.column_stack([
                row["hist_recency_log1p"], row["hist_time_gap_log1p"],
                np.asarray(row["hist_first_event_mask"], dtype=np.float32),
            ]).astype(np.float32)
            time_matrix = self.time_normalization.transform(time_matrix)
            def select(indices):
                valid = tokens[indices] > 1
                return tokens[indices][valid], actions[indices][valid], time_matrix[indices][valid]
            long.append(select(split.long_indices)); short.append(select(split.short_indices))
            targets.append(target); kept.append(row)
        if not kept:
            return {"rows": []}
        user_batch = {}
        for name, values in (("short", short), ("long", long)):
            user_batch[f"{name}_tokens"], user_batch[f"{name}_padding_mask"] = _pad([value[0] for value in values], np.int64)
            user_batch[f"{name}_actions"], _ = _pad([value[1] for value in values], np.int64)
            user_batch[f"{name}_time_features"], _ = _pad([value[2] for value in values], np.float32)
        target_rids = np.asarray([int(row["target_item_rid"]) for row in kept], dtype=np.int64)
        positive = self.item_store.for_rids(target_rids)
        batch = {"rows": kept, "user": user_batch, "positive": positive}
        if self.negative_sampler is not None:
            histories = [np.concatenate([value[0] for value in pair]) for pair in zip(long, short)]
            negative_tokens = self.negative_sampler.sample_batch(targets, histories)
            negative = self.item_store.for_tokens(negative_tokens.reshape(-1))
            batch["negative"] = {
                key: value.reshape(len(kept), negative_tokens.shape[1], *value.shape[1:])
                for key, value in negative.items()
            }
        return batch


def move_tensor_tree(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_tensor_tree(item, device) for key, item in value.items()}
    return value

