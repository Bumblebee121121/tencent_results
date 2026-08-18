"""Stage 4.4: build train-fitted RID and evaluation-candidate item side features."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.stage3_runtime import ParquetSink
from src.features.categorical_encoder import CategoricalVocabulary
from src.features.feature_contract import select_item_side_source
from src.features.id_semantics import encode_item_rid
from src.features.item_feature_store import item_strength_features
from src.features.runtime import (
    Timer,
    add_common_arguments,
    configure_logging,
    guard_outputs,
    load_json,
    load_stage4_config,
    require_paths,
    require_stage3_contracts,
    save_csv,
    save_json,
    stage4_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--max-candidates", type=int)
    return parser.parse_args()


def candidate_schema(features: list[str]) -> pa.Schema:
    fields: list[tuple[str, pa.DataType]] = [
        ("retrieval_id", pa.int64()),
        ("item_oid", pa.int64()),
        ("item_rid", pa.int64()),
        ("model_item_token", pa.int32()),
    ]
    for feature in features:
        fields.append((f"f{feature}_token", pa.int32()))
    for feature in features:
        fields.append((f"f{feature}_missing", pa.bool_()))
    for feature in features:
        fields.append((f"f{feature}_oov", pa.bool_()))
    fields.extend(
        [
            ("item_train_count", pa.int64()),
            ("item_train_count_log1p", pa.float32()),
            ("strength_group", pa.string()),
            ("source_kind", pa.string()),
        ]
    )
    return pa.schema(fields)


def main() -> None:
    args = parse_args()
    config = load_stage4_config(args.config)
    data_root, stage3_root, output_root, log_root = stage4_paths(config, args.debug)
    logger = configure_logging(log_root, "stage4_4_build_item_side_features", args.debug)
    timer = Timer()
    manifests = require_stage3_contracts(stage3_root, str(config["stage3_protocol_version"]))
    features = [str(value) for value in config["item_features"]]
    p50 = float(manifests["strength"]["p50_train"])
    p90 = float(manifests["strength"]["p90_train"])
    item_feat_path = data_root / "item_feat"
    official_candidate_path = data_root / "candidate"
    eval_candidate_path = stage3_root / "candidates" / "eval_candidates.parquet"
    count_path = output_root / "mappings" / "train_item_count_by_rid.npy"
    token_path = output_root / "mappings" / "rid_to_model_item_token.npy"
    user_manifest_path = output_root / "manifests" / "user_feature_manifest.json"
    user_coverage_path = output_root / "audits" / "user_feature_vocab_coverage.csv"
    require_paths(
        [item_feat_path, official_candidate_path, eval_candidate_path, count_path, token_path, user_manifest_path, user_coverage_path]
    )
    item_dataset = ds.dataset(item_feat_path, format="parquet")
    required = {"item_id", *features}
    if not required.issubset(item_dataset.schema.names):
        raise ValueError(f"item_feat missing fields: {sorted(required - set(item_dataset.schema.names))}")
    for feature in features:
        if not pa.types.is_int64(item_dataset.schema.field(feature).type):
            raise TypeError(f"item feature {feature} must be int64")

    feature_dir = output_root / "feature_store"
    side_path = feature_dir / "item_side_tokens_by_rid.npy"
    missing_path = feature_dir / "item_side_missing_by_rid.npy"
    oov_path = feature_dir / "item_side_oov_by_rid.npy"
    candidate_side_path = feature_dir / "eval_candidate_side.parquet"
    manifest_path = output_root / "manifests" / "item_side_manifest.json"
    vocab_manifest_path = output_root / "mappings" / "feature_vocab_manifest.json"
    coverage_path = output_root / "audits" / "feature_vocab_coverage.csv"
    vocab_dir = output_root / "mappings" / "vocab"
    vocab_paths = {feature: vocab_dir / f"item_f{feature}.npy" for feature in features}
    guard_outputs(
        [side_path, missing_path, oov_path, candidate_side_path, manifest_path, vocab_manifest_path, coverage_path, *vocab_paths.values()],
        args.overwrite,
    )
    counts = np.load(count_path, mmap_mode="r", allow_pickle=False)
    rid_tokens = np.load(token_path, mmap_mode="r", allow_pickle=False)
    max_rid = counts.size - 1

    known_sets = {feature: set() for feature in features}
    processed_item_rows = 0
    scanner = item_dataset.scanner(columns=["item_id", *features], batch_size=8192)
    for batch in scanner.to_batches():
        if args.max_items is not None:
            remaining = args.max_items - processed_item_rows
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
        rids = [int(value) for value in batch.column(0).to_pylist()]
        columns = {feature: batch.column(index + 1).to_pylist() for index, feature in enumerate(features)}
        for row_index, rid in enumerate(rids):
            if rid <= 0 or rid > max_rid:
                raise ValueError(f"item_feat RID outside count store: {rid}")
            if int(counts[rid]) <= 0:
                continue
            for feature in features:
                value = columns[feature][row_index]
                if value is not None:
                    known_sets[feature].add(int(value))
        processed_item_rows += batch.num_rows
    vocabularies = {feature: CategoricalVocabulary(sorted(known_sets[feature])) for feature in features}
    for feature, vocabulary in vocabularies.items():
        vocabulary.save(vocab_paths[feature])

    shape = (counts.size, len(features))
    side_store = np.lib.format.open_memmap(side_path, mode="w+", dtype=np.int32, shape=shape)
    missing_store = np.lib.format.open_memmap(missing_path, mode="w+", dtype=np.bool_, shape=shape)
    oov_store = np.lib.format.open_memmap(oov_path, mode="w+", dtype=np.bool_, shape=shape)
    side_store[:] = 1
    missing_store[:] = True
    oov_store[:] = False
    seen_item_rows = np.zeros(counts.size, dtype=np.bool_)
    item_missing = np.zeros(len(features), dtype=np.int64)
    item_oov = np.zeros(len(features), dtype=np.int64)
    processed_item_rows = 0
    scanner = item_dataset.scanner(columns=["item_id", *features], batch_size=8192)
    for batch in scanner.to_batches():
        if args.max_items is not None:
            remaining = args.max_items - processed_item_rows
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
        rids = np.asarray(batch.column(0).to_numpy(), dtype=np.int64)
        if np.any((rids <= 0) | (rids > max_rid)):
            raise ValueError("item_feat RID outside count store")
        if np.any(seen_item_rows[rids]):
            raise ValueError("item_feat contains duplicate item_id rows")
        seen_item_rows[rids] = True
        for index, feature in enumerate(features):
            tokens, missing, oov = vocabularies[feature].encode_many(batch.column(index + 1).to_pylist())
            side_store[rids, index] = tokens
            missing_store[rids, index] = missing
            oov_store[rids, index] = oov
            item_missing[index] += int(np.count_nonzero(missing))
            item_oov[index] += int(np.count_nonzero(oov))
        processed_item_rows += batch.num_rows
    side_store.flush()
    missing_store.flush()
    oov_store.flush()

    official_dataset = ds.dataset(official_candidate_path, format="parquet")
    official_required = {"item_id", "retrieval_id", *features}
    if not official_required.issubset(official_dataset.schema.names):
        raise ValueError(f"official candidate missing fields: {sorted(official_required - set(official_dataset.schema.names))}")
    official_count = official_dataset.count_rows()
    official_oid = np.full(official_count, -1, dtype=np.int64)
    official_tokens = np.full((official_count, len(features)), 1, dtype=np.int32)
    official_missing = np.ones((official_count, len(features)), dtype=np.bool_)
    official_oov = np.zeros((official_count, len(features)), dtype=np.bool_)
    seen_retrieval = np.zeros(official_count, dtype=np.bool_)
    for batch in official_dataset.scanner(columns=["item_id", "retrieval_id", *features], batch_size=8192).to_batches():
        oids = np.asarray(batch.column(0).to_numpy(), dtype=np.int64)
        retrieval_ids = np.asarray(batch.column(1).to_numpy(), dtype=np.int64)
        if np.any((retrieval_ids < 0) | (retrieval_ids >= official_count)):
            raise ValueError("official retrieval_id must be a dense zero-based range")
        if np.any(seen_retrieval[retrieval_ids]):
            raise ValueError("official candidate contains duplicate retrieval_id")
        seen_retrieval[retrieval_ids] = True
        official_oid[retrieval_ids] = oids
        for index, feature in enumerate(features):
            struct_values = batch.column(index + 2)
            raw_values = struct_values.field("feature_value").to_pylist()
            tokens, missing, oov = vocabularies[feature].encode_many(raw_values)
            official_tokens[retrieval_ids, index] = tokens
            official_missing[retrieval_ids, index] = missing
            official_oov[retrieval_ids, index] = oov
    if not np.all(seen_retrieval):
        raise ValueError("official retrieval_id range contains gaps")

    candidate_missing = np.zeros(len(features), dtype=np.int64)
    candidate_oov = np.zeros(len(features), dtype=np.int64)
    candidate_rows = 0
    eval_dataset = ds.dataset(eval_candidate_path, format="parquet")
    with ParquetSink(candidate_side_path, candidate_schema(features), args.overwrite) as sink:
        scanner = eval_dataset.scanner(
            columns=["item_oid", "item_rid", "retrieval_id", "source"], batch_size=8192
        )
        for batch in scanner.to_batches():
            if args.max_candidates is not None:
                remaining = args.max_candidates - candidate_rows
                if remaining <= 0:
                    break
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
            rows = []
            for oid_value, rid_value, retrieval_value, source_value in zip(
                batch.column(0).to_pylist(), batch.column(1).to_pylist(), batch.column(2).to_pylist(), batch.column(3).to_pylist()
            ):
                oid = int(oid_value)
                rid = None if rid_value is None else int(rid_value)
                retrieval_id = int(retrieval_value)
                source = str(source_value)
                source_kind = select_item_side_source(source)
                if source == "official":
                    if retrieval_id < 0 or retrieval_id >= official_count or int(official_oid[retrieval_id]) != oid:
                        raise ValueError("official candidate/eval candidate retrieval alignment mismatch")
                    row_tokens = official_tokens[retrieval_id]
                    row_missing = official_missing[retrieval_id]
                    row_oov = official_oov[retrieval_id]
                else:
                    if rid is None or rid <= 0 or rid > max_rid:
                        raise ValueError("added target must have a valid historical RID")
                    row_tokens = side_store[rid]
                    row_missing = missing_store[rid]
                    row_oov = oov_store[rid]
                count = 0 if rid is None or rid > max_rid else int(counts[rid])
                strength = item_strength_features(count, p50, p90)
                row: dict[str, object] = {
                    "retrieval_id": retrieval_id,
                    "item_oid": oid,
                    "item_rid": rid,
                    "model_item_token": encode_item_rid(rid, rid_to_token=rid_tokens),
                }
                for index, feature in enumerate(features):
                    row[f"f{feature}_token"] = int(row_tokens[index])
                    row[f"f{feature}_missing"] = bool(row_missing[index])
                    row[f"f{feature}_oov"] = bool(row_oov[index])
                    candidate_missing[index] += int(row_missing[index])
                    candidate_oov[index] += int(row_oov[index])
                row.update(strength)
                row["source_kind"] = source_kind
                rows.append(row)
            sink.write_rows(rows)
            candidate_rows += batch.num_rows

    coverage_rows: list[dict[str, object]] = []
    with user_coverage_path.open("r", encoding="utf-8-sig", newline="") as handle:
        coverage_rows.extend(dict(row) for row in csv.DictReader(handle))
    for index, feature in enumerate(features):
        coverage_rows.extend(
            [
                {
                    "entity": "item_rid",
                    "feature": feature,
                    "field_type": "scalar",
                    "train_known_vocab_size": vocabularies[feature].known_size,
                    "row_count": processed_item_rows,
                    "value_count": processed_item_rows,
                    "missing_count": int(item_missing[index]),
                    "missing_rate": float(item_missing[index] / processed_item_rows) if processed_item_rows else 0.0,
                    "oov_count": int(item_oov[index]),
                    "oov_rate": float(item_oov[index] / processed_item_rows) if processed_item_rows else 0.0,
                },
                {
                    "entity": "eval_candidate",
                    "feature": feature,
                    "field_type": "scalar",
                    "train_known_vocab_size": vocabularies[feature].known_size,
                    "row_count": candidate_rows,
                    "value_count": candidate_rows,
                    "missing_count": int(candidate_missing[index]),
                    "missing_rate": float(candidate_missing[index] / candidate_rows) if candidate_rows else 0.0,
                    "oov_count": int(candidate_oov[index]),
                    "oov_rate": float(candidate_oov[index] / candidate_rows) if candidate_rows else 0.0,
                },
            ]
        )
    coverage_fields = ["entity", "feature", "field_type", "train_known_vocab_size", "row_count", "value_count", "missing_count", "missing_rate", "oov_count", "oov_rate"]
    save_csv(coverage_rows, coverage_fields, coverage_path, args.overwrite)

    user_manifest = load_json(user_manifest_path)
    item_vocab_manifest = {
        feature: {
            "train_known_vocab_size": vocabularies[feature].known_size,
            "vocab_path": str(vocab_paths[feature].relative_to(output_root)),
        }
        for feature in features
    }
    save_json(
        {
            "schema_version": 1,
            "fit_scope": {
                "user": "users with Stage 3 train samples",
                "item": "item feature values on items with train_event_count > 0",
            },
            "special_tokens": config["special_tokens"]["categorical"],
            "user": user_manifest["fields"],
            "item": item_vocab_manifest,
        },
        vocab_manifest_path,
        args.overwrite,
    )
    manifest = {
        "stage": "4.4",
        "schema_version": 1,
        "feature_protocol_version": config["feature_protocol_version"],
        "debug": bool(args.debug),
        "item_feature_fields": features,
        "item_rid_array_shape": list(shape),
        "processed_item_row_count": processed_item_rows,
        "eval_candidate_count": candidate_rows,
        "official_candidate_count": official_count,
        "source_precedence": {
            "official": "candidate.feature_value",
            "added_validation_or_test_target": "item_feat by RID",
        },
        "candidate_cold_start_used_as_model_feature": False,
        "retrieval_id_used_as_model_item_id": False,
        "elapsed_seconds": timer.elapsed_seconds,
    }
    save_json(manifest, manifest_path, args.overwrite)
    logger.info("wrote item rows=%d candidates=%d elapsed_seconds=%.2f", processed_item_rows, candidate_rows, timer.elapsed_seconds)


if __name__ == "__main__":
    main()
