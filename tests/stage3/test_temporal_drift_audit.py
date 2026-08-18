from __future__ import annotations

import unittest

import numpy as np

from src.data.temporal_drift import (
    STRENGTH_GROUPS,
    aggregate_temporal_buckets,
    build_split_summary,
    encode_strength_groups,
    quantify_unseen_drift,
    verify_split_summary,
    validate_fixed_strength_groups,
)


class TemporalDriftAuditTest(unittest.TestCase):
    def test_reuses_train_only_strength_thresholds(self) -> None:
        counts = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        groups = encode_strength_groups(
            ["Unseen", "Tail", "Tail", "Mid", "Mid", "Head"]
        )
        validate_fixed_strength_groups(counts, groups, p50_train=2.0, p90_train=4.0)
        corrupted = groups.copy()
        corrupted[1] = STRENGTH_GROUPS.index("Head")
        with self.assertRaises(ValueError):
            validate_fixed_strength_groups(
                counts, corrupted, p50_train=2.0, p90_train=4.0
            )

    def test_daily_group_ratios_sum_to_one(self) -> None:
        rows = aggregate_temporal_buckets(
            target_timestamps=np.array([10, 20, 86_410]),
            strength_codes=encode_strength_groups(["Head", "Unseen", "Unseen"]),
            split_codes=np.array([0, 0, 1], dtype=np.int8),
            train_cutoff=0,
            timestamp_unit="seconds",
            min_targets_per_bucket=1,
            analysis_scope="primary",
        )
        for row in rows:
            total = sum(float(row[f"{group.lower()}_ratio"]) for group in STRENGTH_GROUPS)
            self.assertAlmostEqual(total, 1.0)
        day_zero = next(row for row in rows if row["days_from_train_cutoff"] == 0)
        self.assertEqual(day_zero["target_count"], 2)
        self.assertEqual(day_zero["unseen_ratio"], 0.5)

    def test_split_summary_matches_stage3_5(self) -> None:
        rows = aggregate_temporal_buckets(
            target_timestamps=np.array([1, 2, 86_401, 86_402]),
            strength_codes=encode_strength_groups(["Head", "Unseen", "Tail", "Unseen"]),
            split_codes=np.array([0, 0, 1, 1], dtype=np.int8),
            train_cutoff=0,
            timestamp_unit="seconds",
            min_targets_per_bucket=1,
            analysis_scope="primary",
        )
        summary = build_split_summary(rows)
        expected = {
            "Validation": {
                "Head": (1, 0.5),
                "Mid": (0, 0.0),
                "Tail": (0, 0.0),
                "Unseen": (1, 0.5),
            },
            "Test": {
                "Head": (0, 0.0),
                "Mid": (0, 0.0),
                "Tail": (1, 0.5),
                "Unseen": (1, 0.5),
            },
        }
        verify_split_summary(summary, expected)
        expected["Test"]["Unseen"] = (2, 1.0)
        with self.assertRaises(ValueError):
            verify_split_summary(summary, expected)

    def test_days_from_train_cutoff_is_nonnegative(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_temporal_buckets(
                target_timestamps=np.array([99]),
                strength_codes=encode_strength_groups(["Unseen"]),
                split_codes=np.array([0], dtype=np.int8),
                train_cutoff=100,
                timestamp_unit="seconds",
                min_targets_per_bucket=1,
                analysis_scope="primary",
            )

    def test_primary_and_all_target_outputs_are_separate(self) -> None:
        arguments = dict(
            target_timestamps=np.array([1]),
            strength_codes=encode_strength_groups(["Head"]),
            split_codes=np.array([0], dtype=np.int8),
            train_cutoff=0,
            timestamp_unit="seconds",
            min_targets_per_bucket=1,
        )
        primary = aggregate_temporal_buckets(**arguments, analysis_scope="primary")
        auxiliary = aggregate_temporal_buckets(**arguments, analysis_scope="all_targets")
        self.assertEqual({row["analysis_scope"] for row in primary}, {"primary"})
        self.assertEqual(
            {row["analysis_scope"] for row in auxiliary}, {"all_targets"}
        )

    def test_sparse_tail_buckets_are_excluded_from_trend(self) -> None:
        day_zero_count = 10_000
        day_one_count = 5_000
        day_two_count = 3
        timestamps = np.concatenate(
            [
                np.zeros(day_zero_count, dtype=np.int64),
                np.full(day_one_count, 86_400, dtype=np.int64),
                np.full(day_two_count, 2 * 86_400, dtype=np.int64),
            ]
        )
        groups = np.concatenate(
            [
                np.full(day_zero_count, STRENGTH_GROUPS.index("Head"), dtype=np.int8),
                np.full(day_one_count, STRENGTH_GROUPS.index("Head"), dtype=np.int8),
                np.full(day_two_count, STRENGTH_GROUPS.index("Unseen"), dtype=np.int8),
            ]
        )
        groups[:1_000] = STRENGTH_GROUPS.index("Unseen")
        groups[day_zero_count : day_zero_count + 1_000] = STRENGTH_GROUPS.index(
            "Unseen"
        )
        rows = aggregate_temporal_buckets(
            target_timestamps=timestamps,
            strength_codes=groups,
            split_codes=np.zeros(timestamps.size, dtype=np.int8),
            train_cutoff=0,
            timestamp_unit="seconds",
            min_targets_per_bucket=1_000,
            analysis_scope="primary",
        )
        drift = quantify_unseen_drift(rows, min_targets_per_bucket=1_000)

        day_zero = next(row for row in rows if row["days_from_train_cutoff"] == 0)
        day_one = next(row for row in rows if row["days_from_train_cutoff"] == 1)
        day_two = next(row for row in rows if row["days_from_train_cutoff"] == 2)
        self.assertTrue(day_zero["used_for_trend"])
        self.assertTrue(day_one["used_for_trend"])
        self.assertFalse(day_two["used_for_trend"])
        self.assertEqual(drift["trend_bucket_count"], 2)
        self.assertEqual(drift["first_eligible_day"], 0)
        self.assertEqual(drift["last_eligible_day"], 1)
        self.assertEqual(drift["first_eligible_day_target_count"], 10_000)
        self.assertEqual(drift["last_eligible_day_target_count"], 5_000)
        self.assertAlmostEqual(drift["first_eligible_day_unseen_ratio"], 0.1)
        self.assertAlmostEqual(drift["last_eligible_day_unseen_ratio"], 0.2)
        self.assertAlmostEqual(
            drift["eligible_unseen_ratio_change_first_to_last"], 0.1
        )
        self.assertAlmostEqual(drift["linear_slope_unseen_ratio_per_day"], 0.1)
        self.assertAlmostEqual(drift["raw_unseen_ratio_change_first_to_last"], 0.9)


if __name__ == "__main__":
    unittest.main()
