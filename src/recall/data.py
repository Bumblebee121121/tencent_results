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
        return self.sample_batch([target], [history])[0]

    def sample_batch(
        self, targets: Sequence[int], histories: Sequence[Sequence[int]],
    ) -> np.ndarray:
        """Draw one vectorized proposal matrix, then apply exact per-row exclusions."""

        if len(targets) != len(histories):
            raise ValueError("targets and histories must have equal length")
        row_count = len(targets)
        result = np.empty((row_count, self.negatives), dtype=np.int64)
        if row_count == 0:
            return result
        proposal_width = max(32, self.negatives * 2)
        proposal_indices = self.rng.integers(
            0, self.candidates.size, size=(row_count, proposal_width), dtype=np.int64,
        )
        proposals = self.candidates[proposal_indices]
        for row_index, (target, history) in enumerate(zip(targets, histories)):
            excluded = {int(value) for value in history if int(value) > 1}
            excluded.add(int(target))
            if self.candidates.size - len(excluded) < self.negatives:
                available = int(np.count_nonzero(~np.isin(self.candidates, list(excluded))))
                if available < self.negatives:
                    raise ValueError("not enough eligible negative candidates")
            chosen: set[int] = set()
            values: list[int] = []
            pending = proposals[row_index]
            while len(values) < self.negatives:
                for value in pending:
                    token = int(value)
                    if token not in excluded and token not in chosen:
                        chosen.add(token)
                        values.append(token)
                        if len(values) == self.negatives:
                            break
                if len(values) < self.negatives:
                    pending = self.candidates[self.rng.integers(
                        0, self.candidates.size,
                        size=max(8, 2 * (self.negatives - len(values))), dtype=np.int64,
                    )]
            result[row_index] = values
        return result


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
        for row in rows:
            target = self.store.target_token(row.get("target_item_rid"))
            if self.require_seen_target and target <= 1:
                continue
            history = self.store.history(row, self.max_history_length)
            tokens = self.store.tokens_for_rids(history.item_rid)
            if tokens.size == 0:
                continue
            # PAD/UNK never contribute to pooling or exclusions because candidates are Train-Seen.
            histories.append(np.asarray(tokens[tokens > 1], dtype=np.int64))
            targets.append(target)
            kept.append(row)
        if not kept:
            return {
                "rows": [], "history_tokens": torch.empty(0, dtype=torch.long),
                "history_offsets": torch.zeros(1, dtype=torch.long),
                "target_tokens": torch.empty(0, dtype=torch.long),
            }
        lengths = np.fromiter((len(values) for values in histories), dtype=np.int64, count=len(kept))
        offsets = np.empty(len(kept) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        flattened = (
            np.concatenate(histories)
            if int(offsets[-1])
            else np.empty(0, dtype=np.int64)
        )
        batch: dict[str, object] = {
            "rows": kept,
            "history_tokens": torch.from_numpy(flattened),
            "history_offsets": torch.from_numpy(offsets),
            "target_tokens": torch.tensor(targets, dtype=torch.long),
        }
        if self.negative_sampler is not None:
            batch["negative_tokens"] = torch.from_numpy(
                self.negative_sampler.sample_batch(targets, histories)
            )
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
