"""Select 600/1800/3600-second U1 session gap using Validation Recall@100 only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recall.stage6_retrieval import build_variant_index, evaluate_variant
from src.recall.stage6_runtime import (
    add_common_arguments, guard_outputs, load_config, require_contracts,
    load_json, require_paths, save_json, session_gap_candidates, stage6_paths,
)
from src.recall.stage6_workflow import train_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.debug:
        raise ValueError(
            "Debug uses the provisional best_loss fast path; Session Gap selection is Formal-only"
        )
    config = load_config(args.config)
    paths = stage6_paths(config, False)
    require_contracts(paths, config)
    root = paths["output_root"]
    require_paths([
        root / "audits" / "time_normalization.json",
        root / "audits" / "session_gap_candidate_statistics.json",
    ])
    time_audit = load_json(root / "audits" / "time_normalization.json")
    session_statistics = load_json(root / "audits" / "session_gap_candidate_statistics.json")
    if not bool(time_audit.get("full_train_scan")) or not bool(session_statistics.get("full_train_scan")):
        raise ValueError("Formal Session Gap selection requires complete Train-only statistics")
    selection_path = root / "audits" / "session_gap_selection.json"
    definition_path = root / "audits" / "session_definition.json"
    u1_selection_path = root / "manifests" / "u1_checkpoint_selection.json"
    guard_outputs([selection_path, definition_path, u1_selection_path], args.overwrite)

    candidate_results = []
    for gap in session_gap_candidates(config):
        owner = f"U1_gap{gap}"
        train_variant(
            "U1", config, paths, False, args.overwrite, args.device,
            artifact_name=owner, session_gap_override=gap,
        )
        checkpoint_results = {}
        for checkpoint_label in ("best_loss", "final"):
            build_variant_index(
                "U1", config, paths, False, args.overwrite, args.device,
                checkpoint_label=checkpoint_label, artifact_name=owner,
            )
            checkpoint_results[checkpoint_label] = evaluate_variant(
                "U1", config, paths, False, args.overwrite, args.device,
                checkpoint_label=checkpoint_label, selection_candidate=True,
                artifact_name=owner, session_gap_override=gap, include_test=False,
            )
        selected_checkpoint = max(
            checkpoint_results,
            key=lambda label: float(
                checkpoint_results[label]["metrics"]["validation"]["Overall"]["Recall@100"]
            ),
        )
        candidate_results.append({
            "session_gap_seconds": gap,
            "checkpoint_owner": owner,
            "selected_checkpoint_label": selected_checkpoint,
            "validation_recall100": float(
                checkpoint_results[selected_checkpoint]["metrics"]["validation"]["Overall"]["Recall@100"]
            ),
            "checkpoint_candidate_recall100": {
                label: float(result["metrics"]["validation"]["Overall"]["Recall@100"])
                for label, result in checkpoint_results.items()
            },
        })

    selected = max(
        candidate_results,
        key=lambda row: (float(row["validation_recall100"]), -int(row["session_gap_seconds"])),
    )
    # Materialize canonical U1 Validation output from the winning gap/checkpoint.
    evaluate_variant(
        "U1", config, paths, False, args.overwrite, args.device,
        checkpoint_label=str(selected["selected_checkpoint_label"]),
        selection_candidate=False, artifact_name=str(selected["checkpoint_owner"]),
        session_gap_override=int(selected["session_gap_seconds"]), include_test=False,
    )

    selection = {
        "stage": "6.1b",
        "protocol_version": config["stage6_protocol_version"],
        "selection_split": "validation",
        "selection_metric": "Overall Recall@100",
        "candidate_gap_seconds": list(session_gap_candidates(config)),
        "candidate_results": candidate_results,
        "selected_session_gap_seconds": int(selected["session_gap_seconds"]),
        "selected_checkpoint_owner": str(selected["checkpoint_owner"]),
        "selected_checkpoint_label": str(selected["selected_checkpoint_label"]),
        "test_used_for_selection": False,
        "frozen": True,
        "tie_breaker": "higher Validation Recall@100, then smaller gap",
    }
    save_json(selection, selection_path, args.overwrite)
    save_json({
        "selected_session_gap_seconds": int(selected["session_gap_seconds"]),
        "source": "formal_validation_recall100_selection",
        "selection_audit": str(selection_path.relative_to(PROJECT_ROOT)),
        "frozen": True,
    }, definition_path, args.overwrite)
    save_json({
        "variant": "U1",
        "checkpoint_owner": str(selected["checkpoint_owner"]),
        "selected_checkpoint_label": str(selected["selected_checkpoint_label"]),
        "session_gap_seconds": int(selected["session_gap_seconds"]),
        "selection_split": "validation",
        "selection_metric": "Overall Recall@100",
        "candidate_recall100": {
            str(row["session_gap_seconds"]): float(row["validation_recall100"])
            for row in candidate_results
        },
        "test_used_for_selection": False,
    }, u1_selection_path, args.overwrite)


if __name__ == "__main__":
    main()
