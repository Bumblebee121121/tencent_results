"""Click-labeled pseudo-target construction for the Stage 3 proxy protocol."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass
class ClickTargetStats:
    """Exact counters collected while inspecting complete user sequences."""

    total_event_count: int = 0
    click_target_count: int = 0
    unknown_action_count: int = 0
    empty_history_target_count: int = 0
    same_timestamp_prefix_excluded_count: int = 0
    click_targets_per_user: list[int] = field(default_factory=list)
    history_lengths: list[int] = field(default_factory=list)

    def merge(self, other: "ClickTargetStats", retain_distributions: bool = True) -> None:
        self.total_event_count += other.total_event_count
        self.click_target_count += other.click_target_count
        self.unknown_action_count += other.unknown_action_count
        self.empty_history_target_count += other.empty_history_target_count
        self.same_timestamp_prefix_excluded_count += (
            other.same_timestamp_prefix_excluded_count
        )
        if retain_distributions:
            self.click_targets_per_user.extend(other.click_targets_per_user)
            self.history_lengths.extend(other.history_lengths)


def _required_int(event: Mapping[str, Any], field_name: str, position: int) -> int:
    value = event.get(field_name)
    if value is None:
        raise ValueError(f"event[{position}].{field_name} must not be null")
    return int(value)


def find_click_targets(
    events: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, int]], ClickTargetStats]:
    """Return every click-labeled interaction with its strict prefix boundary.

    ``history_end_position`` is exclusive. Therefore
    ``events[:history_end_position]`` contains only events whose timestamps are
    strictly less than ``target_timestamp``. Empty-history targets are returned
    here for auditing; the sample builder is responsible for skipping them.
    """

    timestamps = [_required_int(event, "timestamp", pos) for pos, event in enumerate(events)]
    if any(later < earlier for earlier, later in zip(timestamps, timestamps[1:])):
        raise ValueError("sequence timestamps are not non-decreasing")

    targets: list[dict[str, int]] = []
    stats = ClickTargetStats(total_event_count=len(events))
    user_click_count = 0
    for position, event in enumerate(events):
        item_rid = _required_int(event, "item_id", position)
        action = event.get("action_type")
        if action is None:
            stats.unknown_action_count += 1
            continue
        if action == 0:
            continue
        if action != 1:
            raise ValueError(f"event[{position}].action_type must be 0, 1 or null")

        target_timestamp = timestamps[position]
        history_end = bisect_left(timestamps, target_timestamp, 0, position + 1)
        history_length = history_end
        if history_end and timestamps[history_end - 1] >= target_timestamp:
            raise AssertionError("history contains target-time or future events")

        targets.append(
            {
                "target_item_rid": item_rid,
                "target_timestamp": target_timestamp,
                "target_position": position,
                "history_end_position": history_end,
                "history_length": history_length,
                "target_action_type": 1,
            }
        )
        stats.click_target_count += 1
        user_click_count += 1
        stats.history_lengths.append(history_length)
        if history_length == 0:
            stats.empty_history_target_count += 1
        if history_end < position:
            # At least one earlier sequence element shares the target timestamp
            # and is deliberately excluded from the strict historical prefix.
            stats.same_timestamp_prefix_excluded_count += 1

    stats.click_targets_per_user.append(user_click_count)
    return targets, stats
