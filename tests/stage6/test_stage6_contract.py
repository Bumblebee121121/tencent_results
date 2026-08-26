import unittest
import json
from pathlib import Path

from src.recall.stage6_runtime import configured_session, load_session_gap_selection, stage6_paths
from src.recall.stage6_workflow import _selected_parent_checkpoint
from tests.stage6._temp import workspace_tempdir


def formal_config():
    return {
        "stage6_protocol_version": "stage6_eda_driven_recall_v1",
        "user_tower": {
            "short_session": {
                "session_gap_seconds": None,
                "candidate_gap_seconds": [600, 1800, 3600],
                "max_events": 32,
            },
            "long_history": {"max_events": 64},
        },
    }


class Stage6ContractTest(unittest.TestCase):
    def test_stage5_output_is_forbidden(self):
        with self.assertRaisesRegex(ValueError, "Stage 5"):
            stage6_paths({"stage5_root": "artifacts/stage5", "output_root": "artifacts/stage5"}, False)

    def test_debug_output_is_separate(self):
        paths = stage6_paths({"output_root": "artifacts/stage6", "log_root": "logs/stage6"}, True)
        self.assertEqual(paths["output_root"].name, "stage6_debug")

    def test_formal_session_cannot_bypass_selection_audit(self):
        with workspace_tempdir() as directory:
            config = formal_config()
            config["user_tower"]["short_session"]["session_gap_seconds"] = 1800
            with self.assertRaisesRegex(FileNotFoundError, "stage6_1b"):
                configured_session(config, False, Path(directory))

    def test_formal_session_uses_frozen_validation_selection(self):
        with workspace_tempdir() as directory:
            root = Path(directory); (root / "audits").mkdir()
            selection = {
                "protocol_version": "stage6_eda_driven_recall_v1",
                "selection_split": "validation", "selection_metric": "Overall Recall@100",
                "candidate_gap_seconds": [600, 1800, 3600],
                "selected_session_gap_seconds": 1800,
                "test_used_for_selection": False, "frozen": True,
            }
            (root / "audits" / "session_gap_selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            self.assertEqual(configured_session(formal_config(), False, root), (1800, 32, 64))

    def test_formal_parent_requires_checkpoint_selection(self):
        with workspace_tempdir() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "progression blocked"):
                _selected_parent_checkpoint(Path(directory), "U1", False)

    def test_selected_parent_owner_can_be_gap_run(self):
        with workspace_tempdir() as directory:
            root = Path(directory); (root / "manifests").mkdir()
            selection = {
                "selection_split": "validation", "selection_metric": "Overall Recall@100",
                "test_used_for_selection": False, "selected_checkpoint_label": "final",
                "checkpoint_owner": "U1_gap1800",
            }
            (root / "manifests" / "u1_checkpoint_selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            expected = root / "checkpoints" / "U1_gap1800" / "final.pt"
            self.assertEqual(_selected_parent_checkpoint(root, "U1", False), expected)
