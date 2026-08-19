from __future__ import annotations

import unittest

import numpy as np

from src.features.feature_contract import (
    ADDED_TARGET_SOURCE,
    OFFICIAL_SOURCE,
    FeatureComparison,
    select_item_side_source,
)
from src.features.multimodal_store import (
    candidate_row_index,
    optional_mm_vector,
    validate_mm_vector,
)


class SourceAndMultimodalContractsTest(unittest.TestCase):
    def test_candidate_item_normalization_and_mismatch_audit(self) -> None:
        audit = FeatureComparison()
        audit.update("42", 42)
        audit.update("43", 44)
        audit.update(None, 5)
        row = audit.as_row("102")
        self.assertEqual(row["both_non_null_count"], 2)
        self.assertEqual(row["equal_count"], 1)
        self.assertEqual(row["mismatch_count"], 1)
        self.assertEqual(row["candidate_null_item_non_null"], 1)

    def test_source_precedence_is_explicit(self) -> None:
        self.assertEqual(select_item_side_source("official"), OFFICIAL_SOURCE)
        self.assertEqual(select_item_side_source("validation_target"), ADDED_TARGET_SOURCE)
        self.assertEqual(select_item_side_source("test_target"), ADDED_TARGET_SOURCE)
        with self.assertRaises(ValueError):
            select_item_side_source("unknown")

    def test_multimodal_valid_missing_and_fail_fast(self) -> None:
        vector = validate_mm_vector(np.arange(32), 32)
        self.assertEqual(vector.dtype, np.float32)
        self.assertEqual(vector.shape, (32,))
        missing, valid = optional_mm_vector(None, 32)
        self.assertFalse(valid)
        self.assertTrue(np.all(missing == 0))
        with self.assertRaises(ValueError):
            validate_mm_vector(np.zeros(31), 32)
        invalid = np.zeros(32)
        invalid[0] = np.nan
        with self.assertRaises(ValueError):
            validate_mm_vector(invalid, 32)

    def test_candidate_mm_alignment_uses_row_order_not_retrieval_id(self) -> None:
        index = candidate_row_index(
            item_oids=[2003, 2001, 2002],
            retrieval_ids=[2, 0, 1],
        )
        self.assertEqual(index, {2003: 0, 2001: 1, 2002: 2})
        with self.assertRaises(ValueError):
            candidate_row_index([2001, 2002], [7, 7])


if __name__ == "__main__":
    unittest.main()
