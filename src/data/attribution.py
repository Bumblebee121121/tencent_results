"""Click-to-exposure attribution for the Stage 3 next-click protocol."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence


OidLookup = Callable[[int], int]


@dataclass
class AttributionStats:
    """Exact audit counters produced while attributing click events."""

    click_count: int = 0
    attributed_count: int = 0
    failure_count: int = 0
    multiple_preceding_exposure_count: int = 0
    same_timestamp_count: int = 0
    gaps: list[int] = field(default_factory=list)

    def merge(self, other: "AttributionStats") -> None:
        self.click_count += other.click_count
        self.attributed_count += other.attributed_count
        self.failure_count += other.failure_count
        self.multiple_preceding_exposure_count += other.multiple_preceding_exposure_count
        self.same_timestamp_count += other.same_timestamp_count
        self.gaps.extend(other.gaps)


def _required_int(event: Mapping[str, Any], field_name: str, position: int) -> int:
    value = event.get(field_name)
    if value is None:
        raise ValueError(f"event[{position}].{field_name} must not be null")
    return int(value)


def attribute_sequence(
    user_id: int,
    events: Sequence[Mapping[str, Any]],
    oid_lookup: OidLookup,
) -> tuple[list[dict[str, int]], AttributionStats]:
    """Attribute each click to its nearest preceding same-item exposure.

    Positions are zero-based. ``history_end_position`` is an exclusive slice
    boundary, so ``events[:history_end_position]`` contains timestamps strictly
    earlier than the target exposure timestamp.
    """

    timestamps = [_required_int(event, "timestamp", pos) for pos, event in enumerate(events)]
    if any(later < earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise ValueError(f"user {user_id} sequence timestamps are not non-decreasing")

    last_exposure: dict[int, tuple[int, int]] = {}
    exposure_counts: dict[int, int] = {}
    samples: list[dict[str, int]] = []
    stats = AttributionStats()

    for position, event in enumerate(events):
        item_rid = _required_int(event, "item_id", position)
        action = event.get("action_type")
        timestamp = timestamps[position]

        if action == 0:
            last_exposure[item_rid] = (position, timestamp)
            exposure_counts[item_rid] = exposure_counts.get(item_rid, 0) + 1
            continue
        if action != 1:
            # Null actions remain in history, but cannot be targets or exposures.
            continue

        stats.click_count += 1
        attribution = last_exposure.get(item_rid)
        if attribution is None:
            stats.failure_count += 1
            continue

        exposure_position, exposure_timestamp = attribution
        if exposure_position >= position or exposure_timestamp > timestamp:
            raise AssertionError("attribution must precede click and cannot come from the future")

        history_end = bisect_left(timestamps, exposure_timestamp, 0, position + 1)
        if history_end > exposure_position:
            raise AssertionError("target exposure leaked into the history slice")

        gap = timestamp - exposure_timestamp
        target_oid = int(oid_lookup(item_rid))
        samples.append(
            {
                "user_id": int(user_id),
                "target_item_rid": item_rid,
                "target_item_oid": target_oid,
                "target_exposure_timestamp": exposure_timestamp,
                "target_click_timestamp": timestamp,
                "target_exposure_position": exposure_position,
                "target_click_position": position,
                "history_end_position": history_end,
                "history_length": history_end,
                "attribution_gap": gap,
            }
        )
        stats.attributed_count += 1
        stats.gaps.append(gap)
        if exposure_counts[item_rid] > 1:
            stats.multiple_preceding_exposure_count += 1
        if gap == 0:
            stats.same_timestamp_count += 1

    if stats.click_count != stats.attributed_count + stats.failure_count:
        raise AssertionError("click attribution counters do not close")
    return samples, stats
