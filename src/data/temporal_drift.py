"""Pure helpers for the Stage 3.7 temporal item-drift audit."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Mapping, Sequence

import numpy as np


STRENGTH_GROUPS = ("Head", "Mid", "Tail", "Unseen")
SPLITS = ("Validation", "Test")
_UNITS_PER_SECOND = {
    "seconds": 1,
    "milliseconds": 1_000,
    "microseconds": 1_000_000,
    "nanoseconds": 1_000_000_000,
}


def units_per_day(timestamp_unit: str) -> int:
    """Resolve an explicit timestamp unit; never infer it from value magnitude."""

    try:
        return 86_400 * _UNITS_PER_SECOND[timestamp_unit]
    except KeyError as error:
        supported = ", ".join(_UNITS_PER_SECOND)
        raise ValueError(
            f"unsupported timestamp_unit {timestamp_unit!r}; choose one of: {supported}"
        ) from error


def encode_strength_groups(groups: Sequence[str]) -> np.ndarray:
    mapping = {name: index for index, name in enumerate(STRENGTH_GROUPS)}
    codes = np.empty(len(groups), dtype=np.int8)
    for index, group in enumerate(groups):
        if group not in mapping:
            raise ValueError(f"unknown item strength group: {group!r}")
        codes[index] = mapping[group]
    return codes


def validate_fixed_strength_groups(
    train_event_counts: np.ndarray,
    strength_codes: np.ndarray,
    p50_train: float,
    p90_train: float,
) -> None:
    """Validate labels using supplied Stage 3.5 thresholds without recomputing them."""

    counts = np.asarray(train_event_counts, dtype=np.int64)
    codes = np.asarray(strength_codes, dtype=np.int8)
    if counts.size != codes.size:
        raise ValueError("train_event_counts and strength_codes must have equal length")
    expected = np.full(counts.shape, STRENGTH_GROUPS.index("Head"), dtype=np.int8)
    expected[counts == 0] = STRENGTH_GROUPS.index("Unseen")
    expected[(counts > 0) & (counts <= p50_train)] = STRENGTH_GROUPS.index("Tail")
    expected[(counts > p50_train) & (counts <= p90_train)] = STRENGTH_GROUPS.index("Mid")
    mismatches = np.flatnonzero(expected != codes)
    if mismatches.size:
        first = int(mismatches[0])
        raise ValueError(
            "item strength label disagrees with fixed Stage 3.5 thresholds at "
            f"row {first}: count={int(counts[first])}, "
            f"label={STRENGTH_GROUPS[int(codes[first])]}"
        )


def lookup_strength_codes(
    target_oids: np.ndarray,
    sorted_item_oids: np.ndarray,
    sorted_strength_codes: np.ndarray,
) -> np.ndarray:
    targets = np.asarray(target_oids, dtype=np.int64)
    item_oids = np.asarray(sorted_item_oids, dtype=np.int64)
    codes = np.asarray(sorted_strength_codes, dtype=np.int8)
    positions = np.searchsorted(item_oids, targets)
    found = positions < item_oids.size
    valid = np.flatnonzero(found)
    found[valid] &= item_oids[positions[valid]] == targets[valid]
    if not np.all(found):
        missing = int(targets[np.flatnonzero(~found)[0]])
        raise ValueError(f"target OID {missing} is missing from item_train_counts")
    return codes[positions]


def _calendar_date(timestamp: int, timestamp_unit: str) -> str:
    scale = _UNITS_PER_SECOND[timestamp_unit]
    whole_seconds = int(timestamp) // scale
    return datetime.fromtimestamp(whole_seconds, tz=timezone.utc).date().isoformat()


def aggregate_temporal_buckets(
    target_timestamps: np.ndarray,
    strength_codes: np.ndarray,
    split_codes: np.ndarray,
    train_cutoff: int,
    timestamp_unit: str,
    min_targets_per_bucket: int,
    analysis_scope: str,
) -> list[dict[str, object]]:
    """Aggregate fixed 24-hour buckets anchored at the Train cutoff."""

    timestamps = np.asarray(target_timestamps, dtype=np.int64)
    groups = np.asarray(strength_codes, dtype=np.int8)
    splits = np.asarray(split_codes, dtype=np.int8)
    if not (timestamps.size == groups.size == splits.size):
        raise ValueError("timestamps, strength codes and split codes must have equal length")
    if timestamps.size == 0:
        raise ValueError("temporal drift analysis requires at least one target")
    if min_targets_per_bucket <= 0:
        raise ValueError("min_targets_per_bucket must be positive")
    if np.any(timestamps < train_cutoff):
        raise ValueError("days_from_train_cutoff must be nonnegative")
    if np.any((groups < 0) | (groups >= len(STRENGTH_GROUPS))):
        raise ValueError("strength code is out of range")
    if np.any((splits < 0) | (splits >= len(SPLITS))):
        raise ValueError("split code is out of range")

    day_size = units_per_day(timestamp_unit)
    day_indices = (timestamps - int(train_cutoff)) // day_size
    counters: dict[tuple[int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(len(STRENGTH_GROUPS), dtype=np.int64)
    )
    combined_totals: dict[int, int] = defaultdict(int)
    for day, split_code, group_code in zip(day_indices, splits, groups):
        key = (int(day), int(split_code))
        counters[key][int(group_code)] += 1
        combined_totals[int(day)] += 1

    rows: list[dict[str, object]] = []
    for (day, split_code), counts in sorted(counters.items()):
        target_count = int(counts.sum())
        ratios = counts.astype(np.float64) / target_count
        if not np.isclose(float(ratios.sum()), 1.0, atol=1e-12):
            raise AssertionError("daily strength-group ratios do not sum to one")
        bucket_start = int(train_cutoff) + day * day_size
        row: dict[str, object] = {
            "analysis_scope": analysis_scope,
            "date": _calendar_date(bucket_start, timestamp_unit),
            "calendar_date": _calendar_date(bucket_start, timestamp_unit),
            "bucket_start_timestamp": bucket_start,
            "days_from_train_cutoff": day,
            "split": SPLITS[split_code],
            "target_count": target_count,
        }
        for index, group in enumerate(STRENGTH_GROUPS):
            key = group.lower()
            row[f"{key}_count"] = int(counts[index])
            row[f"{key}_ratio"] = float(ratios[index])
        row["used_for_trend"] = combined_totals[day] >= min_targets_per_bucket
        rows.append(row)
    return rows


def build_split_summary(
    bucket_rows: Sequence[Mapping[str, object]],
    analysis_scope: str = "primary",
) -> list[dict[str, object]]:
    counters = {split: np.zeros(len(STRENGTH_GROUPS), dtype=np.int64) for split in SPLITS}
    for row in bucket_rows:
        split = str(row["split"])
        if split not in counters:
            raise ValueError(f"unknown split in bucket row: {split!r}")
        for index, group in enumerate(STRENGTH_GROUPS):
            counters[split][index] += int(row[f"{group.lower()}_count"])

    summaries: list[dict[str, object]] = []
    for split in SPLITS:
        counts = counters[split]
        target_count = int(counts.sum())
        if target_count == 0:
            raise ValueError(f"{split} contains no targets")
        ratios = counts.astype(np.float64) / target_count
        summary: dict[str, object] = {
            "analysis_scope": analysis_scope,
            "split": split,
            "target_count": target_count,
        }
        for index, group in enumerate(STRENGTH_GROUPS):
            key = group.lower()
            summary[f"{key}_count"] = int(counts[index])
            summary[f"{key}_ratio"] = float(ratios[index])
        summaries.append(summary)
    return summaries


def verify_split_summary(
    actual_rows: Sequence[Mapping[str, object]],
    expected: Mapping[str, Mapping[str, tuple[int, float]]],
    ratio_tolerance: float = 1e-12,
) -> None:
    actual_by_split = {str(row["split"]): row for row in actual_rows}
    for split in SPLITS:
        if split not in expected or split not in actual_by_split:
            raise ValueError(f"missing Stage 3.5 consistency data for split {split}")
        actual = actual_by_split[split]
        for group in STRENGTH_GROUPS:
            expected_count, expected_ratio = expected[split][group]
            actual_count = int(actual[f"{group.lower()}_count"])
            actual_ratio = float(actual[f"{group.lower()}_ratio"])
            if actual_count != expected_count or not np.isclose(
                actual_ratio, expected_ratio, atol=ratio_tolerance, rtol=0.0
            ):
                raise ValueError(
                    "Stage 3.7 split summary disagrees with Stage 3.5: "
                    f"split={split}, group={group}, actual=({actual_count}, {actual_ratio}), "
                    f"expected=({expected_count}, {expected_ratio})"
                )


def quantify_unseen_drift(
    bucket_rows: Sequence[Mapping[str, object]],
    min_targets_per_bucket: int,
) -> dict[str, object]:
    """Summarize raw daily drift and the sufficiently populated trend subset."""

    if min_targets_per_bucket <= 0:
        raise ValueError("min_targets_per_bucket must be positive")
    by_day: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=np.int64))
    for row in bucket_rows:
        day = int(row["days_from_train_cutoff"])
        by_day[day][0] += int(row["target_count"])
        by_day[day][1] += int(row["unseen_count"])
    if not by_day:
        raise ValueError("temporal drift quantification requires at least one bucket")

    ordered_days = np.asarray(sorted(by_day), dtype=np.float64)
    target_counts = np.asarray([by_day[int(day)][0] for day in ordered_days], dtype=np.int64)
    unseen_counts = np.asarray([by_day[int(day)][1] for day in ordered_days], dtype=np.int64)
    unseen_ratios = unseen_counts / target_counts
    eligible = target_counts >= min_targets_per_bucket
    eligible_days = ordered_days[eligible]
    eligible_target_counts = target_counts[eligible]
    eligible_ratios = unseen_ratios[eligible]

    slope: float | None = None
    if eligible_days.size >= 2 and np.ptp(eligible_days) > 0:
        centered_days = eligible_days - eligible_days.mean()
        slope = float(
            np.sum(centered_days * (eligible_ratios - eligible_ratios.mean()))
            / np.sum(centered_days**2)
        )

    eligible_summary: dict[str, object]
    if eligible_days.size:
        eligible_summary = {
            "first_eligible_day": int(eligible_days[0]),
            "last_eligible_day": int(eligible_days[-1]),
            "first_eligible_day_target_count": int(eligible_target_counts[0]),
            "last_eligible_day_target_count": int(eligible_target_counts[-1]),
            "first_eligible_day_unseen_ratio": float(eligible_ratios[0]),
            "last_eligible_day_unseen_ratio": float(eligible_ratios[-1]),
            "eligible_unseen_ratio_change_first_to_last": float(
                eligible_ratios[-1] - eligible_ratios[0]
            ),
        }
    else:
        # A small debug sample can legitimately have no day reaching the formal
        # threshold. Keep the raw audit usable while making the absent trend explicit.
        eligible_summary = {
            "first_eligible_day": None,
            "last_eligible_day": None,
            "first_eligible_day_target_count": None,
            "last_eligible_day_target_count": None,
            "first_eligible_day_unseen_ratio": None,
            "last_eligible_day_unseen_ratio": None,
            "eligible_unseen_ratio_change_first_to_last": None,
        }

    return {
        "first_day": int(ordered_days[0]),
        "last_day": int(ordered_days[-1]),
        "first_day_unseen_ratio": float(unseen_ratios[0]),
        "last_day_unseen_ratio": float(unseen_ratios[-1]),
        "raw_unseen_ratio_change_first_to_last": float(
            unseen_ratios[-1] - unseen_ratios[0]
        ),
        "min_daily_unseen_ratio": float(unseen_ratios.min()),
        "max_daily_unseen_ratio": float(unseen_ratios.max()),
        "weighted_mean_unseen_ratio": float(unseen_counts.sum() / target_counts.sum()),
        "daily_bucket_count": int(ordered_days.size),
        "trend_bucket_count": int(np.count_nonzero(eligible)),
        **eligible_summary,
        "linear_slope_unseen_ratio_per_day": slope,
    }
