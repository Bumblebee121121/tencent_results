"""Stage 5.6: compare recall baselines and quantify hit complementarity."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recall.evaluation import complementarity_from_ranks
from src.recall.runtime import (
    Timer, add_common_arguments, configure_logging, guard_outputs, load_config, load_json,
    require_contracts, require_paths, save_csv, save_json, stage5_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--run-unit-tests", action="store_true")
    return parser.parse_args()


def rank_map(path: Path) -> dict[int, int | None]:
    table = ds.dataset(path, format="parquet").to_table(columns=["sample_id", "target_rank"])
    return {
        int(sample_id): (None if rank is None else int(rank))
        for sample_id, rank in zip(table.column("sample_id").to_pylist(), table.column("target_rank").to_pylist())
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, stage3_root, stage4_root, output_root, log_root = stage5_paths(config, args.debug)
    logger = configure_logging(log_root, "stage5_6_compare_baselines")
    require_contracts(stage3_root, stage4_root, config)
    timer = Timer()
    itemcf_metrics_path = output_root / "itemcf" / "metrics.json"
    two_tower_metrics_path = output_root / "two_tower" / "metrics.json"
    report_path = output_root / "reports" / "recall_comparison.csv"
    complementarity_path = output_root / "reports" / "channel_complementarity.json"
    manifest_path = output_root / "manifests" / "stage5_manifest.json"
    repeated_audit_path = output_root / "audits" / "repeated_target_audit.json"
    hnsw_audit_path = output_root / "audits" / "hnsw_accuracy_audit.json"
    require_paths([itemcf_metrics_path, two_tower_metrics_path, repeated_audit_path, hnsw_audit_path])
    guard_outputs([report_path, complementarity_path, manifest_path], args.overwrite)
    itemcf = load_json(itemcf_metrics_path)
    two_tower = load_json(two_tower_metrics_path)
    repeated_audit = load_json(repeated_audit_path)
    hnsw_audit = load_json(hnsw_audit_path)
    if itemcf.get("recall_protocol_version") != config["recall_protocol_version"] or two_tower.get("recall_protocol_version") != config["recall_protocol_version"]:
        raise ValueError("baseline metrics protocol mismatch")
    if repeated_audit.get("recall_protocol_version") != config["recall_protocol_version"]:
        raise ValueError("repeated-target audit protocol mismatch")
    if hnsw_audit.get("recall_protocol_version") != config["recall_protocol_version"]:
        raise ValueError("HNSW accuracy audit protocol mismatch")
    if not bool(hnsw_audit.get("passed")):
        raise ValueError("HNSW accuracy audit did not pass configured retrieval-recall thresholds")

    rows = []
    metric_sets = {f"itemcf:{name}": splits for name, splits in itemcf["metrics"].items()}
    metric_sets["two_tower"] = two_tower["metrics"]
    for model, splits in metric_sets.items():
        for split, groups in splits.items():
            for group, metrics in groups.items():
                for k in config["recall_ks"]:
                    rows.append(
                        {"model": model, "split": split, "strength_group": group, "k": int(k),
                         "sample_count": int(metrics["count"]), "recall": float(metrics[f"Recall@{int(k)}"])}
                    )
    save_csv(rows, ["model", "split", "strength_group", "k", "sample_count", "recall"], report_path, args.overwrite)

    # Selection is validation-only; test is used once for the complementarity report.
    best_itemcf = max(
        itemcf["metrics"],
        key=lambda name: float(itemcf["metrics"][name]["validation"]["Overall"]["Recall@100"]),
    )
    first = rank_map(output_root / "itemcf" / "ranks" / f"{best_itemcf}_test.parquet")
    second = rank_map(output_root / "two_tower" / "ranks" / "test.parquet")
    if set(first) != set(second):
        raise ValueError("ItemCF and Two-Tower test rank sample IDs do not align")
    sample_ids = sorted(first)
    complementarity = complementarity_from_ranks(
        [first[sample_id] for sample_id in sample_ids],
        [second[sample_id] for sample_id in sample_ids],
        config["recall_ks"],
    )
    save_json(
        {"selection_rule": "highest validation Overall Recall@100", "first_channel": f"itemcf:{best_itemcf}",
         "second_channel": "two_tower", "evaluation_split": "test", "metrics": complementarity},
        complementarity_path, args.overwrite,
    )

    tests_passed = False
    if args.run_unit_tests:
        completed = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", "tests/stage5", "-v"],
            cwd=PROJECT_ROOT, text=True, capture_output=True, encoding="utf-8", check=False,
        )
        logger.info("Stage 5 unit tests:\n%s%s", completed.stdout, completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError("Stage 5 unit tests failed")
        tests_passed = True
    save_json(
        {
            "schema_version": 1, "recall_protocol_version": config["recall_protocol_version"],
            "stage3_protocol_version": config["stage3_protocol_version"],
            "stage4_protocol_version": config["stage4_protocol_version"], "debug": bool(args.debug),
            "frameworks": {"two_tower": "PyTorch", "ann": "FAISS CPU HNSW-IP", "itemcf": "NumPy/PyArrow"},
            "best_itemcf_selected_on_validation": best_itemcf,
            "exclude_history_items": bool(config["retrieval"]["exclude_history_items"]),
            "hnsw_accuracy_audit_passed": True,
            "all_stage5_tests_passed": tests_passed,
            "outputs": {
                "itemcf_metrics": str(itemcf_metrics_path.relative_to(PROJECT_ROOT)),
                "two_tower_metrics": str(two_tower_metrics_path.relative_to(PROJECT_ROOT)),
                "recall_comparison": str(report_path.relative_to(PROJECT_ROOT)),
                "channel_complementarity": str(complementarity_path.relative_to(PROJECT_ROOT)),
                "repeated_target_audit": str(repeated_audit_path.relative_to(PROJECT_ROOT)),
                "hnsw_accuracy_audit": str(hnsw_audit_path.relative_to(PROJECT_ROOT)),
            },
            "elapsed_seconds": timer.elapsed_seconds,
        },
        manifest_path, args.overwrite,
    )
    logger.info("comparison complete selected_itemcf=%s elapsed_seconds=%.2f", best_itemcf, timer.elapsed_seconds)


if __name__ == "__main__":
    main()
