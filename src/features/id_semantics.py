"""Stage 4 ID and action token contracts."""

from __future__ import annotations

from typing import Mapping

import numpy as np


ITEM_TOKENS = {"pad": 0, "unk": 1}
CATEGORICAL_TOKENS = {"pad": 0, "missing": 1, "oov": 2}
ACTION_TOKENS = {"pad": 0, "exposure": 1, "click": 2, "unknown": 3}


def validate_special_tokens(configured: Mapping[str, Mapping[str, int]]) -> None:
    expected = {
        "item": ITEM_TOKENS,
        "categorical": CATEGORICAL_TOKENS,
        "action": ACTION_TOKENS,
    }
    for namespace, values in expected.items():
        actual = configured.get(namespace)
        if actual is None or {str(k): int(v) for k, v in actual.items()} != values:
            raise ValueError(
                f"special token contract mismatch for {namespace}: "
                f"expected {values}, found {actual}"
            )


def encode_action(raw_action: int | None, padding: bool = False) -> int:
    if padding:
        return ACTION_TOKENS["pad"]
    if raw_action is None:
        return ACTION_TOKENS["unknown"]
    value = int(raw_action)
    if value == 0:
        return ACTION_TOKENS["exposure"]
    if value == 1:
        return ACTION_TOKENS["click"]
    return ACTION_TOKENS["unknown"]


def build_rid_to_model_item_token(train_counts: np.ndarray) -> np.ndarray:
    """Use RID+1 only for train-seen RIDs; every unseen RID shares UNK."""

    counts = np.asarray(train_counts)
    if counts.ndim != 1:
        raise ValueError("train_counts must be one-dimensional")
    if np.any(counts < 0):
        raise ValueError("train_counts cannot be negative")
    max_token = counts.size
    if max_token > np.iinfo(np.int32).max:
        raise OverflowError("model item token exceeds int32 range")
    tokens = np.full(counts.shape, ITEM_TOKENS["unk"], dtype=np.int32)
    tokens[0] = ITEM_TOKENS["pad"]
    seen_rids = np.flatnonzero(counts > 0)
    seen_rids = seen_rids[seen_rids > 0]
    tokens[seen_rids] = seen_rids.astype(np.int32) + 1
    return tokens


def encode_item_rid(
    item_rid: int | None,
    train_count: int | None = None,
    rid_to_token: np.ndarray | None = None,
) -> int:
    """Encode an RID without ever accepting a retrieval_id as a substitute."""

    if item_rid is None:
        return ITEM_TOKENS["unk"]
    rid = int(item_rid)
    if rid <= 0:
        return ITEM_TOKENS["unk"]
    if rid_to_token is not None:
        return (
            int(rid_to_token[rid])
            if rid < len(rid_to_token)
            else ITEM_TOKENS["unk"]
        )
    if train_count is None:
        raise ValueError("train_count or rid_to_token is required for item encoding")
    return rid + 1 if int(train_count) > 0 else ITEM_TOKENS["unk"]
