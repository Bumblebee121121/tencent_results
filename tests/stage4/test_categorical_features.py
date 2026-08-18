from __future__ import annotations

import unittest

from src.features.categorical_encoder import CategoricalVocabulary, normalize_integer_value
from src.features.id_semantics import CATEGORICAL_TOKENS


class CategoricalFeaturesTest(unittest.TestCase):
    def test_known_oov_and_missing_are_distinct(self) -> None:
        vocabulary = CategoricalVocabulary.fit([10, 20])
        known = vocabulary.encode(10)
        oov = vocabulary.encode(30)
        missing = vocabulary.encode(None)
        self.assertGreaterEqual(known.token, 3)
        self.assertEqual(oov.token, CATEGORICAL_TOKENS["oov"])
        self.assertEqual(missing.token, CATEGORICAL_TOKENS["missing"])
        self.assertTrue(oov.oov)
        self.assertTrue(missing.missing)

    def test_candidate_decimal_string_normalizes_to_item_integer(self) -> None:
        self.assertEqual(normalize_integer_value(" 42 "), 42)
        self.assertEqual(normalize_integer_value(42), 42)
        with self.assertRaises(ValueError):
            normalize_integer_value("not-an-integer")

    def test_list_order_is_preserved_and_null_is_distinct_from_empty(self) -> None:
        vocabulary = CategoricalVocabulary.fit([1, 2, 3])
        expected = [vocabulary.encode(3).token, vocabulary.encode(1).token, vocabulary.encode(3).token]
        tokens, missing, oov = vocabulary.encode_list([3, 1, 3])
        self.assertEqual(tokens, expected)
        self.assertFalse(missing)
        self.assertEqual(oov, [False, False, False])
        null_tokens, null_missing, _ = vocabulary.encode_list(None)
        empty_tokens, empty_missing, _ = vocabulary.encode_list([])
        self.assertEqual(null_tokens, empty_tokens)
        self.assertTrue(null_missing)
        self.assertFalse(empty_missing)


if __name__ == "__main__":
    unittest.main()
