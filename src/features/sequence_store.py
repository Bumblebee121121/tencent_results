"""CSR-style sequence access with strict alignment checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HistorySlice:
    item_rid: np.ndarray
    action_token: np.ndarray
    timestamp: np.ndarray

    @property
    def length(self) -> int:
        return int(self.item_rid.size)


def validate_aligned_sequence(
    item_rids: np.ndarray,
    action_tokens: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    if not (len(item_rids) == len(action_tokens) == len(timestamps)):
        raise ValueError("item/action/timestamp sequences are not aligned")
    if len(timestamps) > 1 and np.any(np.diff(np.asarray(timestamps, dtype=np.int64)) < 0):
        raise ValueError("sequence timestamps must be nondecreasing")


def slice_history(
    user_id: int,
    history_end_position: int,
    offsets: np.ndarray,
    seq_item_rid: np.ndarray,
    seq_action_token: np.ndarray,
    seq_timestamp: np.ndarray,
    target_timestamp: int | None = None,
) -> HistorySlice:
    user = int(user_id)
    end_position = int(history_end_position)
    if user < 0 or user + 1 >= offsets.size:
        raise IndexError(f"user_id is outside the sequence store: {user}")
    start = int(offsets[user])
    user_end = int(offsets[user + 1])
    if end_position < 0 or start + end_position > user_end:
        raise IndexError("history_end_position exceeds the user's sequence")
    end = start + end_position
    history = HistorySlice(
        np.asarray(seq_item_rid[start:end]),
        np.asarray(seq_action_token[start:end]),
        np.asarray(seq_timestamp[start:end]),
    )
    validate_aligned_sequence(history.item_rid, history.action_token, history.timestamp)
    if target_timestamp is not None and history.length:
        if int(history.timestamp.max()) >= int(target_timestamp):
            raise ValueError("history includes an event at or after the target timestamp")
    return history
