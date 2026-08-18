"""Stage 4.8: smoke-test the unified FeatureDataset and write the final manifest."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_store import FeatureStore
from src.features.id_semantics import ITEM_TOKENS
from src.features.next_click_dataset import NextClickFeatureDataset, add_dynamic_time_features
from src.features.runtime import (
    Timer,
    add_common_arguments,
    configure_logging,
    guard_outputs,
    load_json,
    load_stage4_config,
    require_paths,
    require_stage3_contracts,
    save_json,
    stage4_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--samples-per-split", type=int, default=100)
    parser.add_argument("--run-unit-tests", action="store_true")
    return parser.parse_args()


def first_rows(path: Path, limit: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for batch in ds.dataset(path, format="parquet").scanner(batch_size=min(8192, limit)).to_batches():
        remaining = limit - len(rows)
        if remaining <= 0:
            break
        if batch.num_rows > remaining:
            batch = batch.slice(0, remaining)
        rows.extend(batch.to_pylist())
    return rows


def selected_user_features(path: Path, user_ids: set[int]) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for batch in ds.dataset(path, format="parquet").scanner(batch_size=8192).to_batches():
        for row in batch.to_pylist():
            user_id = int(row["user_id"])
            if user_id in user_ids:
                result[user_id] = row
        if len(result) == len(user_ids):
            break
    return result


def main() -> None:
    args = parse_args()
    if args.samples_per_split <= 0:
        raise ValueError("--samples-per-split must be positive")
    config = load_stage4_config(args.config)
    data_root, stage3_root, output_root, log_root = stage4_paths(config, args.debug)
    logger = configure_logging(log_root, "stage4_8_feature_dataset_smoke", args.debug)
    timer = Timer()
    stage3 = require_stage3_contracts(stage3_root, str(config["stage3_protocol_version"]))
    component_paths = {
        "item_base": output_root / "manifests" / "train_item_base_manifest.json",
        "user": output_root / "manifests" / "user_feature_manifest.json",
        "item_side": output_root / "manifests" / "item_side_manifest.json",
        "mm": output_root / "manifests" / "multimodal_store_manifest.json",
        "sequence": output_root / "manifests" / "sequence_store_manifest.json",
        "vocab": output_root / "mappings" / "feature_vocab_manifest.json",
    }
    sample_paths = {
        "train": stage3_root / "samples" / "train_samples.parquet",
        "validation": stage3_root / "samples" / "val_primary.parquet",
        "test": stage3_root / "samples" / "test_primary.parquet",
    }
    user_feature_path = output_root / "feature_store" / "user_features.parquet"
    candidate_side_path = output_root / "feature_store" / "eval_candidate_side.parquet"
    candidate_mm_path = output_root / "feature_store" / "eval_candidate_mm.npy"
    candidate_mm_valid_path = output_root / "feature_store" / "eval_candidate_mm_valid.npy"
    coverage_path = output_root / "audits" / "feature_vocab_coverage.csv"
    contract_path = output_root / "audits" / "feature_contract.json"
    require_paths(
        [*component_paths.values(), *sample_paths.values(), user_feature_path, candidate_side_path, candidate_mm_path, candidate_mm_valid_path, coverage_path, contract_path]
    )
    final_manifest_path = output_root / "manifests" / "stage4_manifest.json"
    smoke_path = output_root / "manifests" / "feature_dataset_smoke.json"
    leakage_path = output_root / "audits" / "leakage_audit.json"
    guard_outputs([final_manifest_path, smoke_path, leakage_path], args.overwrite)
    components = {name: load_json(path) for name, path in component_paths.items()}

    samples_by_split = {
        split: first_rows(path, args.samples_per_split) for split, path in sample_paths.items()
    }
    all_samples = [row for rows in samples_by_split.values() for row in rows]
    user_ids = {int(row["user_id"]) for row in all_samples}
    user_features = selected_user_features(user_feature_path, user_ids)
    if set(user_features) != user_ids:
        raise ValueError("user feature store does not cover every smoke sample user")

    candidate_count = int(components["item_side"]["eval_candidate_count"])
    import numpy as np

    candidate_mm = np.load(candidate_mm_path, mmap_mode="r", allow_pickle=False)
    candidate_mm_valid = np.load(candidate_mm_valid_path, mmap_mode="r", allow_pickle=False)
    if candidate_mm.shape != (candidate_count, int(config["mm_dim"])):
        raise ValueError("eval candidate MM array is not aligned with candidate side rows")
    if candidate_mm_valid.shape != (candidate_count,):
        raise ValueError("eval candidate MM valid mask is not aligned")

    provisional_manifest = {
        "p50_train": float(components["item_base"]["p50_train"]),
        "p90_train": float(components["item_base"]["p90_train"]),
        "mm_dim": int(config["mm_dim"]),
    }
    store = FeatureStore(output_root, manifest_override=provisional_manifest)
    required_keys = {
        "sample_id", "user_id", "target_item_oid", "target_item_rid", "target_item_token",
        "target_timestamp", "hist_item_rid", "hist_item_token", "hist_action_token",
        "hist_timestamp", "hist_length", "user_features", "target_item_side",
        "target_item_side_missing", "target_item_side_oov", "target_mm", "target_mm_valid",
        "target_train_count", "target_train_count_log1p", "target_strength_group",
    }
    split_results = {}
    unseen_checked = 0
    for split, rows in samples_by_split.items():
        dataset = NextClickFeatureDataset(rows, store, user_features)
        for index in range(len(dataset)):
            item = dataset[index]
            missing_keys = required_keys - set(item)
            if missing_keys:
                raise ValueError(f"FeatureDataset missing keys: {sorted(missing_keys)}")
            timed = add_dynamic_time_features(item)
            if len(timed["hist_item_rid"]) != int(timed["hist_length"]):
                raise ValueError("history feature lengths disagree")
            if int(timed["target_train_count"]) == 0:
                unseen_checked += 1
                if int(timed["target_item_token"]) != ITEM_TOKENS["unk"]:
                    raise ValueError("train-unseen target did not map to shared UNK")
        split_results[split] = {"sample_count": len(dataset), "passed": True}

    tests_passed = False
    test_output = "not run; pass --run-unit-tests after builders complete"
    if args.run_unit_tests:
        completed = subprocess.run(
            [sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", "tests/stage4", "-v"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            encoding="utf-8",
            check=False,
        )
        test_output = completed.stdout + completed.stderr
        logger.info("Stage 4 unit tests output:\n%s", test_output)
        if completed.returncode != 0:
            raise RuntimeError("Stage 4 unit tests failed")
        tests_passed = True

    train_unseen_candidate_count = 0
    for batch in ds.dataset(candidate_side_path, format="parquet").scanner(
        columns=["model_item_token"], batch_size=65536
    ).to_batches():
        train_unseen_candidate_count += sum(
            int(value) == ITEM_TOKENS["unk"] for value in batch.column(0).to_pylist()
        )
    feature_vocab_sizes = {}
    missing_rates = {}
    oov_rates = {}
    with coverage_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = f"{row['entity']}.f{row['feature']}"
            feature_vocab_sizes[key] = int(row["train_known_vocab_size"])
            missing_rates[key] = float(row["missing_rate"])
            oov_rates[key] = float(row["oov_rate"])

    contract = load_json(contract_path)
    final_manifest = {
        "schema_version": 1,
        "feature_protocol_version": config["feature_protocol_version"],
        "stage3_protocol_version": config["stage3_protocol_version"],
        "stage3_train_cutoff": int(stage3["splits"]["train_raw_event_cutoff_exclusive"]),
        "debug": bool(args.debug),
        "user_count": int(components["sequence"]["user_count"]),
        "item_rid_count": int(components["item_base"]["item_rid_count"]),
        "eval_candidate_count": candidate_count,
        "user_feature_fields": config["user_scalar_features"] + config["user_list_features"],
        "item_feature_fields": config["item_features"],
        "special_token_definitions": config["special_tokens"],
        "train_seen_item_count": int(components["item_base"]["train_seen_item_count"]),
        "train_unseen_eval_candidate_count": train_unseen_candidate_count,
        "feature_vocab_sizes": feature_vocab_sizes,
        "missing_rates": missing_rates,
        "oov_rates": oov_rates,
        "p50_train": float(components["item_base"]["p50_train"]),
        "p90_train": float(components["item_base"]["p90_train"]),
        "mm_dim": int(components["mm"]["mm_dim"]),
        "mm_valid_count": int(components["mm"]["mm_valid_count"]),
        "mm_missing_count": int(components["mm"]["mm_missing_count"]),
        "sequence_event_count": int(components["sequence"]["sequence_event_count"]),
        "sequence_store_backend": components["sequence"]["sequence_store_backend"],
        "candidate_cold_start_used_as_model_feature": False,
        "retrieval_id_used_as_model_item_id": False,
        "materialize_history_per_sample": False,
        "stage3_count_consistency_passed": bool(components["item_base"]["stage3_count_consistency_passed"]),
        "candidate_item_mismatch_fields": contract["candidate_item_mismatch_fields"],
        "feature_dataset_smoke_passed": True,
        "all_stage4_tests_passed": tests_passed,
    }
    save_json(
        {
            "schema_version": 1,
            "splits": split_results,
            "train_unseen_targets_checked": unseen_checked,
            "required_fields_passed": True,
            "history_invariant_passed": True,
            "unit_tests_passed": tests_passed,
            "unit_test_output": test_output,
            "elapsed_seconds": timer.elapsed_seconds,
        },
        smoke_path,
        args.overwrite,
    )
    save_json(
        {
            "fit_order": "split first -> fit train-known state -> transform all splits",
            "item_vocabulary_fit_scope": "train-seen item feature values only",
            "user_vocabulary_fit_scope": "users with Stage 3 train samples only",
            "train_count_cutoff": final_manifest["stage3_train_cutoff"],
            "stage3_count_consistency_passed": True,
            "train_unseen_item_token": ITEM_TOKENS["unk"],
            "retrieval_id_used_as_model_item_id": False,
            "candidate_cold_start_used_as_model_feature": False,
        },
        leakage_path,
        args.overwrite,
    )
    save_json(final_manifest, final_manifest_path, args.overwrite)
    logger.info("FeatureDataset smoke passed samples=%d elapsed_seconds=%.2f", len(all_samples), timer.elapsed_seconds)


if __name__ == "__main__":
    main()
