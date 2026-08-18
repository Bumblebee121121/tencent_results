"""Stage 4.1: audit schemas, ID semantics and cross-source feature consistency."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_contract import FeatureComparison
from src.features.multimodal_store import validate_mm_vector
from src.features.runtime import (
    Timer,
    add_common_arguments,
    configure_logging,
    guard_outputs,
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


def require_fields(dataset: ds.Dataset, path: Path, fields: set[str]) -> None:
    missing = fields - set(dataset.schema.names)
    if missing:
        raise ValueError(f"schema fields missing from {path}: {sorted(missing)}")


def main() -> None:
    args = parse_args()
    config = load_stage4_config(args.config)
    data_root, stage3_root, output_root, log_root = stage4_paths(config, args.debug)
    logger = configure_logging(log_root, "stage4_1_feature_contract_audit", args.debug)
    timer = Timer()
    stage3 = require_stage3_contracts(stage3_root, str(config["stage3_protocol_version"]))
    user_scalar = [str(value) for value in config["user_scalar_features"]]
    user_list = [str(value) for value in config["user_list_features"]]
    item_features = [str(value) for value in config["item_features"]]
    paths = {
        "seq": data_root / "seq",
        "user_feat": data_root / "user_feat",
        "item_feat": data_root / "item_feat",
        "candidate": data_root / "candidate",
        "mm": data_root / "mm_emb" / "emb_81_32_parquet",
        "indexer": data_root / "indexer.pkl",
        "eval_candidates": stage3_root / "candidates" / "eval_candidates.parquet",
    }
    require_paths(paths.values())
    audit_dir = output_root / "audits"
    contract_path = audit_dir / "feature_contract.json"
    consistency_path = audit_dir / "candidate_item_feature_consistency.csv"
    mm_audit_path = audit_dir / "multimodal_source_audit.json"
    guard_outputs([contract_path, consistency_path, mm_audit_path], args.overwrite)
    datasets = {
        name: ds.dataset(path, format="parquet")
        for name, path in paths.items()
        if name != "indexer"
    }
    require_fields(datasets["seq"], paths["seq"], {"user_id", "seq"})
    require_fields(datasets["user_feat"], paths["user_feat"], {"user_id", *user_scalar, *user_list})
    require_fields(datasets["item_feat"], paths["item_feat"], {"item_id", *item_features})
    require_fields(datasets["candidate"], paths["candidate"], {"item_id", "retrieval_id", *item_features})
    require_fields(datasets["mm"], paths["mm"], {"anonymous_cid", "emb"})
    require_fields(datasets["eval_candidates"], paths["eval_candidates"], {"item_oid", "item_rid", "retrieval_id", "source"})
    seq_type = datasets["seq"].schema.field("seq").type
    if not pa.types.is_list(seq_type):
        raise TypeError("seq must be a list")
    seq_struct = seq_type.value_type
    if not pa.types.is_struct(seq_struct) or set(seq_struct.names) != {"item_id", "action_type", "timestamp"}:
        raise TypeError("seq element must contain item_id/action_type/timestamp")
    for field in user_scalar:
        if not pa.types.is_int64(datasets["user_feat"].schema.field(field).type):
            raise TypeError(f"user scalar {field} must be int64")
    for field in user_list:
        if not pa.types.is_list(datasets["user_feat"].schema.field(field).type):
            raise TypeError(f"user list {field} must be list")
    for field in item_features:
        if not pa.types.is_int64(datasets["item_feat"].schema.field(field).type):
            raise TypeError(f"item feature {field} must be int64")
        candidate_type = datasets["candidate"].schema.field(field).type
        if not pa.types.is_struct(candidate_type) or set(candidate_type.names) != {"cold_start", "feature_value"}:
            raise TypeError(f"candidate feature {field} must contain cold_start/feature_value")

    logger.info("loading indexer=%s", paths["indexer"])
    with paths["indexer"].open("rb") as handle:
        indexer = pickle.load(handle)
    if not isinstance(indexer, dict) or not {"i", "u", "a", "f"}.issubset(indexer):
        raise ValueError("indexer must contain i/u/a/f mappings")
    oid_to_rid = indexer["i"]
    rid_values = np.fromiter((int(value) for value in oid_to_rid.values()), dtype=np.int64)
    if np.unique(rid_values).size != rid_values.size:
        raise ValueError("item OID->RID mapping is not one-to-one")
    rid_min = int(rid_values.min())
    rid_max = int(rid_values.max())
    if rid_min <= 0:
        raise ValueError("historical item RID must be positive")

    namespace_audit = {}
    for field in user_scalar + user_list + item_features:
        namespace = indexer["f"].get(field)
        if not isinstance(namespace, dict):
            raise ValueError(f"indexer feature namespace missing or invalid: {field}")
        sample_items = list(namespace.items())[:100]
        namespace_audit[field] = {
            "size": len(namespace),
            "key_types": sorted({type(key).__name__ for key, _ in sample_items}),
            "value_types": sorted({type(value).__name__ for _, value in sample_items}),
            "candidate_feature_value_normalization": "decimal string -> int64",
        }

    official_oids: list[int] = []
    official_rids: list[int] = []
    official_retrieval: list[int] = []
    official_dataset = datasets["candidate"]
    processed_candidates = 0
    for batch in official_dataset.scanner(columns=["item_id", "retrieval_id"], batch_size=65536).to_batches():
        if args.max_candidates is not None:
            remaining = args.max_candidates - processed_candidates
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
        for oid_value, retrieval_value in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            oid = int(oid_value)
            official_oids.append(oid)
            rid = oid_to_rid.get(oid)
            if rid is not None:
                official_rids.append(int(rid))
            official_retrieval.append(int(retrieval_value))
        processed_candidates += batch.num_rows
    if len(set(official_oids)) != len(official_oids):
        raise ValueError("official candidate OIDs are not unique")
    if len(set(official_retrieval)) != len(official_retrieval):
        raise ValueError("official retrieval_ids are not unique")

    overlap_rids = set(official_rids)
    item_values_by_rid: dict[int, tuple[object | None, ...]] = {}
    processed_items = 0
    for batch in datasets["item_feat"].scanner(columns=["item_id", *item_features], batch_size=8192).to_batches():
        columns = [batch.column(index + 1).to_pylist() for index in range(len(item_features))]
        for row_index, rid_value in enumerate(batch.column(0).to_pylist()):
            rid = int(rid_value)
            if rid in overlap_rids:
                if rid in item_values_by_rid:
                    raise ValueError(f"duplicate item_feat RID: {rid}")
                item_values_by_rid[rid] = tuple(column[row_index] for column in columns)
        processed_items += batch.num_rows
        if args.max_items is not None and len(item_values_by_rid) >= args.max_items:
            break

    comparisons = {field: FeatureComparison() for field in item_features}
    compared_items = 0
    processed_candidates = 0
    for batch in official_dataset.scanner(columns=["item_id", *item_features], batch_size=8192).to_batches():
        if args.max_candidates is not None:
            remaining = args.max_candidates - processed_candidates
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
        for row_index, oid_value in enumerate(batch.column(0).to_pylist()):
            rid = oid_to_rid.get(int(oid_value))
            if rid is None:
                continue
            item_values = item_values_by_rid.get(int(rid))
            if item_values is None:
                continue
            compared_items += 1
            for feature_index, field in enumerate(item_features):
                candidate_value = batch.column(feature_index + 1)[row_index].as_py()
                feature_value = None if candidate_value is None else candidate_value.get("feature_value")
                comparisons[field].update(feature_value, item_values[feature_index])
        processed_candidates += batch.num_rows
    consistency_rows = [comparisons[field].as_row(field) for field in item_features]

    eval_null_rid = 0
    eval_count = 0
    eval_retrieval: set[int] = set()
    for batch in datasets["eval_candidates"].scanner(columns=["item_rid", "retrieval_id"], batch_size=65536).to_batches():
        eval_null_rid += batch.column(0).null_count
        values = [int(value) for value in batch.column(1).to_pylist()]
        if len(set(values)) != len(values) or any(value in eval_retrieval for value in values):
            raise ValueError("eval candidate retrieval_id is not unique")
        eval_retrieval.update(values)
        eval_count += batch.num_rows

    mm_rows = 0
    mm_null_embedding_rows = 0
    mm_oids: set[int] = set()
    for batch in datasets["mm"].scanner(columns=["anonymous_cid", "emb"], batch_size=8192).to_batches():
        if args.max_items is not None:
            remaining = args.max_items - mm_rows
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
        for oid_text, vector in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            oid = int(str(oid_text), 10)
            if oid in mm_oids:
                raise ValueError(f"duplicate MM OID: {oid}")
            mm_oids.add(oid)
            if vector is None:
                mm_null_embedding_rows += 1
            else:
                validate_mm_vector(vector, int(config["mm_dim"]))
            mm_rows += 1

    save_csv(consistency_rows, list(consistency_rows[0]), consistency_path, args.overwrite)
    save_json(
        {
            "expected_dimension": int(config["mm_dim"]),
            "processed_row_count": mm_rows,
            "null_embedding_row_count": mm_null_embedding_rows,
            "null_embedding_policy": "missing MM; zeros in store with valid mask=0",
            "wrong_dimension_count": 0,
            "nan_or_inf_count": 0,
            "duplicate_oid_count": 0,
            "audit_complete": args.max_items is None,
        },
        mm_audit_path,
        args.overwrite,
    )
    contract = {
        "stage": "4.1",
        "schema_version": 1,
        "feature_protocol_version": config["feature_protocol_version"],
        "stage3_protocol_version": config["stage3_protocol_version"],
        "stage3_train_cutoff": int(stage3["splits"]["train_raw_event_cutoff_exclusive"]),
        "debug": bool(args.debug),
        "schema_validation_passed": True,
        "configured_user_scalar_features": user_scalar,
        "configured_user_list_features": user_list,
        "configured_item_features": item_features,
        "item_rid_min": rid_min,
        "item_rid_max": rid_max,
        "item_rid_unique_count": int(rid_values.size),
        "official_candidate_count_audited": len(official_oids),
        "candidate_oid_to_rid_coverage_count": len(official_rids),
        "candidate_oid_to_rid_coverage_ratio": len(official_rids) / len(official_oids) if official_oids else None,
        "retrieval_id_unique": True,
        "retrieval_id_min": min(official_retrieval) if official_retrieval else None,
        "retrieval_id_max": max(official_retrieval) if official_retrieval else None,
        "eval_candidate_count": eval_count,
        "eval_candidate_rid_null_count": eval_null_rid,
        "candidate_item_compared_item_count": compared_items,
        "item_feat_rows_scanned_for_overlap": processed_items,
        "candidate_item_mismatch_fields": [row["feature"] for row in consistency_rows if int(row["mismatch_count"]) > 0],
        "feature_namespace_audit": namespace_audit,
        "source_precedence": {
            "official_candidate": "candidate.feature_value",
            "added_validation_or_test_target": "item_feat by historical RID",
        },
        "candidate_cold_start_used_as_model_feature": False,
        "retrieval_id_is_model_embedding_id": False,
        "retrieval_id_used_as_model_item_id": False,
        "elapsed_seconds": timer.elapsed_seconds,
    }
    save_json(contract, contract_path, args.overwrite)
    logger.info("audit complete compared_items=%d mm_rows=%d elapsed_seconds=%.2f", compared_items, mm_rows, timer.elapsed_seconds)


if __name__ == "__main__":
    main()
