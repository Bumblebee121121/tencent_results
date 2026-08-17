from __future__ import annotations

import unittest

from src.data.evaluation_candidates import ordered_candidate_additions


class CandidateCoverageTest(unittest.TestCase):
    def test_union_covers_validation_and_test_targets(self) -> None:
        official = {1, 2}
        validation = {2, 3}
        test = {3, 4}
        val_additions, test_additions = ordered_candidate_additions(official, validation, test)
        final = official | set(val_additions) | set(test_additions)
        self.assertEqual(val_additions, [3])
        self.assertEqual(test_additions, [4])
        self.assertTrue(validation.issubset(final))
        self.assertTrue(test.issubset(final))


if __name__ == "__main__":
    unittest.main()
