from __future__ import annotations

import unittest

import numpy as np

from src.features.id_semantics import (
    ACTION_TOKENS,
    ITEM_TOKENS,
    build_rid_to_model_item_token,
    encode_action,
    encode_item_rid,
)


class IdAndActionSemanticsTest(unittest.TestCase):
    def test_train_seen_gets_stable_token_and_unseen_gets_shared_unk(self) -> None:
        counts = np.array([0, 4, 0, 2], dtype=np.int32)
        tokens = build_rid_to_model_item_token(counts)
        self.assertEqual(tokens[0], ITEM_TOKENS["pad"])
        self.assertNotEqual(encode_item_rid(1, rid_to_token=tokens), ITEM_TOKENS["unk"])
        self.assertEqual(encode_item_rid(2, rid_to_token=tokens), ITEM_TOKENS["unk"])
        self.assertEqual(encode_item_rid(None, rid_to_token=tokens), ITEM_TOKENS["unk"])

    def test_retrieval_id_is_not_an_item_encoding_argument(self) -> None:
        with self.assertRaises(TypeError):
            encode_item_rid(None, retrieval_id=123)  # type: ignore[call-arg]

    def test_action_tokens_are_four_distinct_states(self) -> None:
        values = {
            encode_action(0),
            encode_action(1),
            encode_action(None),
            encode_action(None, padding=True),
        }
        self.assertEqual(len(values), 4)
        self.assertEqual(encode_action(0), ACTION_TOKENS["exposure"])
        self.assertEqual(encode_action(1), ACTION_TOKENS["click"])
        self.assertEqual(encode_action(None), ACTION_TOKENS["unknown"])
        self.assertEqual(encode_action(None, padding=True), ACTION_TOKENS["pad"])


if __name__ == "__main__":
    unittest.main()
