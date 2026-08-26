import unittest
from pathlib import Path

from src.recall.stage6_runtime import stage6_paths


class Stage6ContractTest(unittest.TestCase):
    def test_stage5_output_is_forbidden(self):
        with self.assertRaisesRegex(ValueError, "Stage 5"):
            stage6_paths({"stage5_root": "artifacts/stage5", "output_root": "artifacts/stage5"}, False)

    def test_debug_output_is_separate(self):
        paths = stage6_paths({"output_root": "artifacts/stage6", "log_root": "logs/stage6"}, True)
        self.assertEqual(paths["output_root"].name, "stage6_debug")

