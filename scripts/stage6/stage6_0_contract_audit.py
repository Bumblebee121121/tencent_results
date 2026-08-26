"""Stage 6.0: audit immutable upstream contracts and freeze Stage 5 baselines."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from src.recall.stage6_runtime import add_common_arguments, guard_outputs, load_config, require_contracts, save_json, stage6_paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__); add_common_arguments(parser); return parser.parse_args()


def _recall100(metrics, variant=None):
    block = metrics["metrics"]
    if variant is not None: block = block[variant]
    return float(block["validation"]["Overall"]["Recall@100"])


def main():
    args = parse_args(); config = load_config(args.config); paths = stage6_paths(config, args.debug)
    contracts = require_contracts(paths, config)
    audit_path = paths["output_root"] / "audits" / "contract.json"
    freeze_path = paths["output_root"] / "manifests" / "baseline_freeze.json"
    guard_outputs([audit_path, freeze_path], args.overwrite)
    best = contracts["stage5"].get("best_itemcf_selected_on_validation", "click3_recent20")
    freeze = {
        "stage6_protocol_version": config["stage6_protocol_version"], "stage5_read_only": True,
        "itemcf_variant": best, "itemcf_validation_recall100": _recall100(contracts["itemcf"], best),
        "two_tower_validation_recall100": _recall100(contracts["two_tower"]),
    }
    audit = {
        "passed": True, "stage3_protocol": config["stage3_protocol_version"],
        "stage4_protocol": config["stage4_protocol_version"], "stage5_protocol": config["stage5_recall_protocol_version"],
        "train_cutoff_exclusive": contracts["splits"]["train_raw_event_cutoff_exclusive"],
        "validation_primary_count": contracts["splits"]["primary_counts"]["validation"],
        "test_primary_count": contracts["splits"]["primary_counts"]["test"],
        "eval_candidate_count": contracts["candidates"]["final_candidate_count"],
        "strength_thresholds": {"p50_train": contracts["stage4"]["p50_train"], "p90_train": contracts["stage4"]["p90_train"]},
        "stage5_output_writes_allowed": False,
    }
    save_json(freeze, freeze_path, args.overwrite); save_json(audit, audit_path, args.overwrite)


if __name__ == "__main__": main()

