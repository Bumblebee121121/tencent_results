import argparse
import unittest
from pathlib import Path

from scripts.stage6.run_stage6_debug import build_steps, select_steps


def args(**overrides):
    values = {
        "config": Path("configs/stage6.yaml"),
        "device": "cuda",
        "overwrite": True,
        "skip_tests": False,
        "start_at": None,
        "stop_after": None,
        "dry_run": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class DebugRunnerTest(unittest.TestCase):
    def test_complete_dependency_order(self):
        steps = build_steps(args())
        names = [step.name for step in steps]
        self.assertLess(names.index("u1_evaluate"), names.index("u2_train"))
        self.assertLess(names.index("u2_evaluate"), names.index("u3_train"))
        self.assertLess(names.index("u3_evaluate"), names.index("i1_train"))
        self.assertLess(names.index("i3_evaluate"), names.index("e1_train"))
        self.assertEqual(names[-3:], ["ablation", "complementarity", "fusion"])
        self.assertNotIn("stage6_1b_select_session_gap", " ".join(names))

    def test_every_stage_script_is_debug_scoped(self):
        for step in build_steps(args(skip_tests=True)):
            self.assertIn("--debug", step.command)
            self.assertIn("--overwrite", step.command)

    def test_resume_slice_is_inclusive(self):
        steps = build_steps(args())
        selected = select_steps(steps, "i2_index", "i3_train")
        self.assertEqual(selected[0].name, "i2_index")
        self.assertEqual(selected[-1].name, "i3_train")

    def test_invalid_resume_step_fails(self):
        with self.assertRaisesRegex(ValueError, "unknown --start-at"):
            select_steps(build_steps(args()), "missing", None)

