from __future__ import annotations

import unittest

import numpy as np

from src.data.temporal_drift import (
    STRENGTH_GROUPS,
    aggregate_temporal_buckets,
    build_split_summary,
    encode_strength_groups,
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


if __name__ == "__main__":
    unittest.main()
