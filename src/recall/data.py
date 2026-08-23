"""Lightweight Stage 4 sequence access and streaming Stage 3 PyTorch data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pyarrow.dataset as ds
import torch
from torch.utils.data import IterableDataset, get_worker_info

from src.features.sequence_store import HistorySlice, slice_history


SAMPLE_COLUMNS = [
    "sample_id", "user_id", "target_item_rid", "target_item_oid", "target_timestamp",
    "history_end_position", "history_length",
]


class Stage5SequenceStore:
    def __init__(self, stage4_root: Path):
        feature = Path(stage4_root) / "feature_store"
        mappings = Path(stage4_root) / "mappings"
        self.offsets = np.load(feature / "user_seq_offsets.npy", mmap_mode="r", allow_pickle=False)
        self.item_rids = np.load(feature / "seq_item_rid.npy", mmap_mode="r", allow_pickle=False)
        self.actions = np.load(feature / "seq_action_token.npy", mmap_mode="r", allow_pickle=False)
        self.timestamps = np.load(feature / "seq_timestamp.npy", mmap_mode="r", allow_pickle=False)
        self.rid_to_token = np.load(mappings / "rid_to_model_item_token.npy", mmap_mode="r", allow_pickle=False)
        self.train_counts = np.load(mappings / "train_item_count_by_rid.npy", mmap_mode="r", allow_pickle=False)

    def history(self, row: dict[str, object], max_length: int | None = None) -> HistorySlice:
        value = slice_history(
            int(row["user_id"]), int(row["history_end_position"]), self.offsets,
            self.item_rids, self.actions, self.timestamps, int(row["target_timestamp"]),
        )
        if max_length is not None and value.length > max_length:
            return HistorySlice(value.item_rid[-max_length:], value.action_token[-max_length:], value.timestamp[-max_length:])
        return value

    def tokens_for_rids(self, rids: np.ndarray) -> np.ndarray:
        values = np.asarray(rids, dtype=np.int64)
        if values.size and (values.min() <= 0 or values.max() >= self.rid_to_token.size):
            raise ValueError("RID is outside Stage 4 model-token mapping")
        return np.asarray(self.rid_to_token[values], dtype=np.int64)

    def target_token(self, rid: int | None) -> int:
        if rid is None or int(rid) <= 0 or int(rid) >= self.rid_to_token.size:
            return 1
        return int(self.rid_to_token[int(rid)])


class ParquetSampleIterableDataset(IterableDataset):
    """Stream Parquet batches; workers receive disjoint batch numbers."""

    def __init__(self, path: Path, max_rows: int | None = None, batch_size: int = 8192):
        super().__init__()
        self.path = Path(path)
        self.max_rows = max_rows
        self.batch_size = int(batch_size)

    def __iter__(self) -> Iterator[dict[str, object]]:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        emitted = 0
        scanner = ds.dataset(self.path, format="parquet").scanner(columns=SAMPLE_COLUMNS, batch_size=self.batch_size)
        for batch_number, batch in enumerate(scanner.to_batches()):
            if batch_number % workers != worker_id:
                continue
            for row in batch.to_pylist():
                if self.max_rows is not None and emitted >= self.max_rows:
                    return
                emitted += 1
                yield row


class UniformNegativeSampler:
    def __init__(self, candidate_tokens: Sequence[int], negatives: int, seed: int = 42):
        self.candidates = np.unique(np.asarray(candidate_tokens, dtype=np.int64))
        if self.candidates.size == 0 or np.any(self.candidates <= 1):
            raise ValueError("negative pool must contain train-seen item tokens only")
        self.negatives = int(negatives)
        self.rng = np.random.default_rng(seed)

    def sample(self, target: int, history: Sequence[int]) -> np.ndarray:
        excluded = set(map(int, history))
        excluded.add(int(target))
        if self.candidates.size - len(excluded) < self.negatives:
            available = int(np.count_nonzero(~np.isin(self.candidates, list(excluded))))
            if available < self.negatives:
                raise ValueError("not enough eligible negative candidates")
        result: list[int] = []
        chosen: set[int] = set()
        while len(result) < self.negatives:
            draws = self.rng.choice(self.candidates, size=max(8, 2 * (self.negatives - len(result))), replace=True)
            for value in draws:
                token = int(value)
                if token not in excluded and token not in chosen:
                    chosen.add(token)
                    result.append(token)
                    if len(result) == self.negatives:
                        break
        return np.asarray(result, dtype=np.int64)


@dataclass
class TwoTowerCollator:
    store: Stage5SequenceStore
    negative_sampler: UniformNegativeSampler | None = None
    max_history_length: int | None = None
    require_seen_target: bool = True

    def __call__(self, rows: list[dict[str, object]]) -> dict[str, object]:
        histories: list[np.ndarray] = []
        kept: list[dict[str, object]] = []
        targets: list[int] = []
        negatives: list[np.ndarray] = []
        for row in rows:
            target = self.store.target_token(row.get("target_item_rid"))
            if self.require_seen_target and target <= 1:
                continue
            history = self.store.history(row, self.max_history_length)
            tokens = self.store.tokens_for_rids(history.item_rid)
            if tokens.size == 0:
                continue
            histories.append(tokens)
            targets.append(target)
            kept.append(row)
            if self.negative_sampler is not None:
                negatives.append(self.negative_sampler.sample(target, tokens))
        if not kept:
            return {"rows": [], "history_tokens": torch.empty((0, 0), dtype=torch.long), "target_tokens": torch.empty(0, dtype=torch.long)}
        width = max(len(values) for values in histories)
        padded = np.zeros((len(histories), width), dtype=np.int64)
        for index, values in enumerate(histories):
            padded[index, -len(values):] = values
        batch: dict[str, object] = {
            "rows": kept,
            "history_tokens": torch.from_numpy(padded),
            "target_tokens": torch.tensor(targets, dtype=torch.long),
        }
        if negatives:
            batch["negative_tokens"] = torch.from_numpy(np.stack(negatives))
        return batch


def candidate_table(stage3_root: Path):
    return ds.dataset(Path(stage3_root) / "candidates" / "eval_candidates.parquet", format="parquet").to_table()


def train_seen_item_tokens(store: Stage5SequenceStore) -> np.ndarray:
    """Build the negative pool only from train-period counts, independent of eval candidates."""

    train_seen_rids = np.flatnonzero(np.asarray(store.train_counts) > 0)
    tokens = np.asarray(store.rid_to_token[train_seen_rids], dtype=np.int64)
    if tokens.size == 0 or np.any(tokens <= 1):
        raise ValueError("train-seen RID/token mapping is inconsistent")
    if np.unique(tokens).size != tokens.size:
        raise ValueError("train-seen item tokens must be one-to-one")
    return tokens
