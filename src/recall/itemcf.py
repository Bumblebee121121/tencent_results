"""Windowed, weighted ItemCF with disk-partitioned pair accumulation."""

from __future__ import annotations

import heapq
import shutil
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ACTION_EXPOSURE = 1
ACTION_CLICK = 2


def action_weights(actions: np.ndarray, exposure: float, click: float, unknown: float = 0.0) -> np.ndarray:
    result = np.full(np.asarray(actions).shape, float(unknown), dtype=np.float64)
    result[np.asarray(actions) == ACTION_EXPOSURE] = float(exposure)
    result[np.asarray(actions) == ACTION_CLICK] = float(click)
    return result


def aggregate_user_strength(item_rids: np.ndarray, weights: np.ndarray) -> dict[int, float]:
    result: dict[int, float] = defaultdict(float)
    for rid, weight in zip(item_rids, weights):
        if weight > 0:
            result[int(rid)] += float(weight)
    return dict(result)


def local_directed_pairs(
    item_rids: Sequence[int], valid: Sequence[bool], candidate_rids: set[int] | np.ndarray, window: int
) -> set[tuple[int, int]]:
    """Return user-deduplicated (arbitrary seed, candidate neighbor) pairs."""

    if window <= 0:
        raise ValueError("window must be positive")
    items = np.asarray(item_rids, dtype=np.int64)
    mask = np.asarray(valid, dtype=np.bool_)
    if isinstance(candidate_rids, np.ndarray):
        candidate_mask = candidate_rids[items]
    else:
        candidate_mask = np.fromiter((int(rid) in candidate_rids for rid in items), dtype=np.bool_, count=items.size)
    result: set[tuple[int, int]] = set()
    for distance in range(1, min(window, items.size - 1) + 1):
        left_items, right_items = items[:-distance], items[distance:]
        eligible = mask[:-distance] & mask[distance:] & (left_items != right_items)
        left_to_right = eligible & candidate_mask[distance:]
        right_to_left = eligible & candidate_mask[:-distance]
        result.update(zip(left_items[left_to_right].tolist(), right_items[left_to_right].tolist()))
        result.update(zip(right_items[right_to_left].tolist(), left_items[right_to_left].tolist()))
    return result


def cosine_pair_scores(
    users: Iterable[tuple[Sequence[int], Sequence[int]]],
    candidate_rids: set[int],
    window: int,
    exposure_weight: float,
    click_weight: float,
    unknown_weight: float = 0.0,
) -> dict[tuple[int, int], float]:
    """Reference implementation used by tests and toy jobs."""

    norms: dict[int, float] = defaultdict(float)
    numerators: dict[tuple[int, int], float] = defaultdict(float)
    for item_values, action_values in users:
        items = np.asarray(item_values, dtype=np.int64)
        weights = action_weights(np.asarray(action_values), exposure_weight, click_weight, unknown_weight)
        strengths = aggregate_user_strength(items, weights)
        for rid, strength in strengths.items():
            norms[rid] += strength * strength
        for seed, neighbor in local_directed_pairs(items, weights > 0, candidate_rids, window):
            numerators[(seed, neighbor)] += strengths[seed] * strengths[neighbor]
    return {
        pair: numerator / np.sqrt(norms[pair[0]] * norms[pair[1]])
        for pair, numerator in numerators.items()
        if norms[pair[0]] > 0 and norms[pair[1]] > 0
    }


def top_neighbors(scores: Mapping[tuple[int, int], float], topn: int) -> dict[int, list[tuple[int, float]]]:
    grouped: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for (seed, neighbor), score in scores.items():
        if seed != neighbor:
            grouped[int(seed)].append((float(score), int(neighbor)))
    return {
        seed: [(neighbor, score) for score, neighbor in sorted(values, key=lambda x: (-x[0], x[1]))[:topn]]
        for seed, values in grouped.items()
    }


class PartitionedPairAccumulator:
    """Bound pair RAM and write hash partitions as sorted NumPy chunks."""

    def __init__(self, root: Path, partitions: int, buffer_size: int):
        self.root = Path(root)
        self.partitions = int(partitions)
        self.buffer_size = int(buffer_size)
        self.buffer: dict[tuple[int, int], list[float]] = {}
        self.chunk_counts = np.zeros(self.partitions, dtype=np.int64)

    def add(self, seed: int, neighbor: int, equal_value: float, click3_value: float) -> None:
        pair = (int(seed), int(neighbor))
        current = self.buffer.get(pair)
        if current is None:
            self.buffer[pair] = [float(equal_value), float(click3_value)]
        else:
            current[0] += float(equal_value)
            current[1] += float(click3_value)
        if len(self.buffer) >= self.buffer_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        grouped: dict[int, list[tuple[int, int, float, float]]] = defaultdict(list)
        for (seed, neighbor), values in self.buffer.items():
            grouped[seed % self.partitions].append((seed, neighbor, values[0], values[1]))
        for partition, rows in grouped.items():
            directory = self.root / f"part_{partition:04d}"
            directory.mkdir(parents=True, exist_ok=True)
            chunk = int(self.chunk_counts[partition])
            array = np.asarray(rows, dtype=np.float64)
            np.save(directory / f"chunk_{chunk:06d}.npy", array, allow_pickle=False)
            self.chunk_counts[partition] += 1
        self.buffer.clear()


NEIGHBOR_SCHEMA = pa.schema(
    [
        ("weight_variant", pa.string()),
        ("item_rid", pa.int64()),
        ("neighbor_item_rid", pa.int64()),
        ("neighbor_item_oid", pa.int64()),
        ("similarity", pa.float32()),
        ("rank", pa.int32()),
    ]
)


def _reduce_partition(directory: Path) -> dict[tuple[int, int], list[float]]:
    result: dict[tuple[int, int], list[float]] = {}
    if not directory.exists():
        return result
    for path in sorted(directory.glob("chunk_*.npy")):
        for seed_value, neighbor_value, equal_value, click3_value in np.load(path, mmap_mode="r"):
            pair = (int(seed_value), int(neighbor_value))
            current = result.get(pair)
            if current is None:
                result[pair] = [float(equal_value), float(click3_value)]
            else:
                current[0] += float(equal_value)
                current[1] += float(click3_value)
    return result


def _partition_topn(
    pairs: Mapping[tuple[int, int], Sequence[float]],
    norms: np.ndarray,
    variant_index: int,
    topn: int,
) -> dict[int, list[tuple[int, float]]]:
    heaps: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for (seed, neighbor), values in pairs.items():
        denominator = float(np.sqrt(norms[variant_index, seed] * norms[variant_index, neighbor]))
        if seed == neighbor or denominator <= 0:
            continue
        entry = (float(values[variant_index]) / denominator, -int(neighbor))
        heap = heaps[seed]
        if len(heap) < topn:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return {
        seed: [(-neg_neighbor, score) for score, neg_neighbor in sorted(heap, reverse=True)]
        for seed, heap in heaps.items()
    }


def build_partitioned_itemcf(
    offsets: np.ndarray,
    seq_items: np.ndarray,
    seq_actions: np.ndarray,
    seq_timestamps: np.ndarray,
    cutoff: int,
    candidate_oid_by_rid: Mapping[int, int],
    output_path: Path,
    shard_root: Path,
    window: int,
    topn: int,
    partitions: int,
    buffer_size: int,
    overwrite: bool,
    max_users: int | None = None,
    logger=None,
) -> dict[str, int]:
    """Build equal/click3 similarities; recent20 reuses click3 similarity at recall time."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists (pass --overwrite): {output_path}")
    candidate_rids = set(map(int, candidate_oid_by_rid))
    users = min(len(offsets) - 1, max_users) if max_users is not None else len(offsets) - 1
    selected_stop = int(offsets[users])
    maximum_rid = max(int(np.max(seq_items[:selected_stop])) if selected_stop else 0, max(candidate_rids, default=0))
    candidate_lookup = np.zeros(maximum_rid + 1, dtype=np.bool_)
    candidate_lookup[np.fromiter(candidate_rids, dtype=np.int64)] = True
    norms = np.zeros((2, maximum_rid + 1), dtype=np.float64)
    temporary = output_path.parent / f"itemcf_pairs_{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    accumulator = PartitionedPairAccumulator(temporary, partitions, buffer_size)
    event_count = 0
    try:
        for user_id in range(users):
            start, stop = int(offsets[user_id]), int(offsets[user_id + 1])
            if stop <= start:
                continue
            timestamps = np.asarray(seq_timestamps[start:stop])
            train_stop = int(np.searchsorted(timestamps, cutoff, side="left"))
            if train_stop <= 0:
                continue
            items = np.asarray(seq_items[start : start + train_stop], dtype=np.int64)
            actions = np.asarray(seq_actions[start : start + train_stop], dtype=np.int8)
            equal_weights = action_weights(actions, 1.0, 1.0, 0.0)
            click3_weights = action_weights(actions, 1.0, 3.0, 0.0)
            equal_strength = aggregate_user_strength(items, equal_weights)
            click3_strength = aggregate_user_strength(items, click3_weights)
            for rid, strength in equal_strength.items():
                norms[0, rid] += strength * strength
            for rid, strength in click3_strength.items():
                norms[1, rid] += strength * strength
            for seed, neighbor in local_directed_pairs(items, equal_weights > 0, candidate_lookup, window):
                accumulator.add(
                    seed, neighbor,
                    equal_strength[seed] * equal_strength[neighbor],
                    click3_strength[seed] * click3_strength[neighbor],
                )
            event_count += train_stop
            if logger is not None and user_id and user_id % 50000 == 0:
                logger.info("itemcf scanned_users=%d train_events=%d buffered_pairs=%d", user_id, event_count, len(accumulator.buffer))
        accumulator.flush()

        if shard_root.exists() and not overwrite:
            raise FileExistsError(f"neighbor shard directory exists (pass --overwrite): {shard_root}")
        shard_root.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output_path, NEIGHBOR_SCHEMA, compression="snappy")
        row_count = 0
        seed_count: set[int] = set()
        try:
            for partition in range(partitions):
                reduced = _reduce_partition(temporary / f"part_{partition:04d}")
                for variant_index, variant_name in enumerate(("equal", "click3")):
                    neighbors = _partition_topn(reduced, norms, variant_index, topn)
                    rows = []
                    for seed in sorted(neighbors):
                        seed_count.add(seed)
                        for rank, (neighbor, score) in enumerate(neighbors[seed], start=1):
                            rows.append(
                                {
                                    "weight_variant": variant_name,
                                    "item_rid": seed,
                                    "neighbor_item_rid": neighbor,
                                    "neighbor_item_oid": int(candidate_oid_by_rid[neighbor]),
                                    "similarity": float(score),
                                    "rank": rank,
                                }
                            )
                            if len(rows) >= 100000:
                                writer.write_table(pa.Table.from_pylist(rows, schema=NEIGHBOR_SCHEMA))
                                row_count += len(rows)
                                rows.clear()
                    if rows:
                        writer.write_table(pa.Table.from_pylist(rows, schema=NEIGHBOR_SCHEMA))
                        row_count += len(rows)
                    _write_neighbor_shard(shard_root, variant_name, partition, neighbors, partitions)
                del reduced
        finally:
            writer.close()
        return {"processed_users": users, "train_event_count": event_count, "neighbor_rows": row_count, "seed_count": len(seed_count)}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _write_neighbor_shard(
    root: Path, variant: str, partition: int, neighbors: Mapping[int, Sequence[tuple[int, float]]], partitions: int
) -> None:
    directory = root / variant
    directory.mkdir(parents=True, exist_ok=True)
    max_local = max((seed // partitions for seed in neighbors), default=-1)
    offsets = np.zeros(max_local + 2, dtype=np.int64)
    total = sum(len(values) for values in neighbors.values())
    flat_rids = np.empty(total, dtype=np.int32)
    flat_scores = np.empty(total, dtype=np.float32)
    cursor = 0
    for local in range(max_local + 1):
        seed = local * partitions + partition
        for neighbor, score in neighbors.get(seed, ()):
            flat_rids[cursor] = int(neighbor)
            flat_scores[cursor] = float(score)
            cursor += 1
        offsets[local + 1] = cursor
    prefix = directory / f"part_{partition:04d}"
    np.save(str(prefix) + "_offsets.npy", offsets, allow_pickle=False)
    np.save(str(prefix) + "_rids.npy", flat_rids, allow_pickle=False)
    np.save(str(prefix) + "_scores.npy", flat_scores, allow_pickle=False)


class NeighborShardLookup:
    def __init__(self, root: Path, variant: str, partitions: int):
        self.root = Path(root) / variant
        self.partitions = int(partitions)
        self.cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def neighbors(self, seed_rid: int) -> tuple[np.ndarray, np.ndarray]:
        seed = int(seed_rid)
        partition = seed % self.partitions
        if partition not in self.cache:
            prefix = self.root / f"part_{partition:04d}"
            self.cache[partition] = (
                np.load(str(prefix) + "_offsets.npy", mmap_mode="r", allow_pickle=False),
                np.load(str(prefix) + "_rids.npy", mmap_mode="r", allow_pickle=False),
                np.load(str(prefix) + "_scores.npy", mmap_mode="r", allow_pickle=False),
            )
        offsets, rids, scores = self.cache[partition]
        local = seed // self.partitions
        if local + 1 >= offsets.size:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        start, stop = int(offsets[local]), int(offsets[local + 1])
        return np.asarray(rids[start:stop]), np.asarray(scores[start:stop])


def retrieve_itemcf(
    history_rids: Sequence[int],
    history_actions: Sequence[int],
    lookup: NeighborShardLookup,
    max_k: int,
    exposure_weight: float,
    click_weight: float,
    unknown_weight: float,
    history_limit: int | None,
) -> list[int]:
    items = np.asarray(history_rids, dtype=np.int64)
    actions = np.asarray(history_actions, dtype=np.int8)
    if history_limit is not None:
        items, actions = items[-history_limit:], actions[-history_limit:]
    weights = action_weights(actions, exposure_weight, click_weight, unknown_weight)
    interacted = set(map(int, history_rids))
    scores: dict[int, float] = defaultdict(float)
    for seed, weight in zip(items, weights):
        if weight <= 0:
            continue
        neighbor_rids, similarities = lookup.neighbors(int(seed))
        for neighbor, similarity in zip(neighbor_rids, similarities):
            rid = int(neighbor)
            if rid not in interacted:
                scores[rid] += float(weight) * float(similarity)
    return [rid for rid, _ in sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:max_k]]
