from __future__ import annotations

import unittest

from src.data.sample_builder import make_rid_to_oid, oid_lookup_from_array
from src.data.evaluation_candidates import candidate_history_rid


class IdMappingTest(unittest.TestCase):
    def test_oid_rid_oid_round_trip(self) -> None:
        mapping = {20001: 1, 20002: 2}
        reverse = make_rid_to_oid(mapping)
        lookup = oid_lookup_from_array(reverse)
        for oid, rid in mapping.items():
            self.assertEqual(lookup(rid), oid)

    def test_missing_history_rid_is_not_fabricated(self) -> None:
        reverse = make_rid_to_oid({20001: 1, 20003: 3})
        lookup = oid_lookup_from_array(reverse)
        with self.assertRaises(KeyError):
            lookup(2)

    def test_history_unseen_candidate_keeps_null_rid(self) -> None:
        self.assertIsNone(candidate_history_rid(99999, {20001: 1}))


if __name__ == "__main__":
    unittest.main()
