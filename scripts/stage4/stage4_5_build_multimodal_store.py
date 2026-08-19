"""Stage 4.5: build float32 multimodal stores with explicit validity masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.stage3_runtime import load_item_mapping
from src.features.multimodal_store import candidate_row_index, validate_mm_vector
from src.features.runtime import (
    Timer,
    add_common_arguments,
    configure_logging,
    guard_outputs,
    load_stage4_config,
    require_paths,
    require_stage3_contracts,
    save_json,
    stage4_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--max-items", type=int, help="limit MM rows for smoke runs")
    parser.add_argument("--max-candidates", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_stage4_config(args.config)
    data_root, stage3_root, output_root, log_root = stage4_paths(config, args.debug)
    logger = configure_logging(log_root, "stage4_5_build_multimodal_store", args.debug)
    timer = Timer()
    require_stage3_contracts(stage3_root, str(config["stage3_protocol_version"]))
    mm_dim = int(config["mm_dim"])
    if str(config["mm_dtype"]) != "float32":
        raise ValueError("Stage 4 MM store currently requires mm_dtype: float32")
    mm_path = data_root / "mm_emb" / "emb_81_32_parquet"
    indexer_path = data_root / "indexer.pkl"
    candidate_path = stage3_root / "candidates" / "eval_candidates.parquet"
    require_paths([mm_path, indexer_path, candidate_path])
    output_dir = output_root / "feature_store"
    mm_rid_path = output_dir / "mm_by_rid.npy"
    mm_rid_valid_path = output_dir / "mm_valid_by_rid.npy"
    candidate_mm_path = output_dir / "eval_candidate_mm.npy"
    candidate_valid_path = output_dir / "eval_candidate_mm_valid.npy"
    manifest_path = output_root / "manifests" / "multimodal_store_manifest.json"
    audit_path = output_root / "audits" / "multimodal_alignment.json"
    guard_outputs(
        [mm_rid_path, mm_rid_valid_path, candidate_mm_path, candidate_valid_path, manifest_path, audit_path],
        args.overwrite,
    )

    oid_to_rid = load_item_mapping(indexer_path)
    max_rid = max(int(value) for value in oid_to_rid.values())
    candidate_dataset = ds.dataset(candidate_path, format="parquet")
    candidate_count_total = candidate_dataset.count_rows()
    candidate_limit = candidate_count_total
    if args.max_candidates is not None:
        if args.max_candidates <= 0:
            raise ValueError("--max-candidates must be positive")
        candidate_limit = min(candidate_limit, args.max_candidates)
    candidate_oids = np.empty(candidate_limit, dtype=np.int64)
    candidate_retrieval_ids = np.empty(candidate_limit, dtype=np.int64)
    loaded_candidates = 0
    for batch in candidate_dataset.scanner(columns=["item_oid", "retrieval_id"], batch_size=65536).to_batches():
        remaining = candidate_limit - loaded_candidates
        if remaining <= 0:
            break
        if batch.num_rows > remaining:
            batch = batch.slice(0, remaining)
        for oid_value, retrieval_value in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            oid = int(oid_value)
            candidate_oids[loaded_candidates] = oid
            candidate_retrieval_ids[loaded_candidates] = int(retrieval_value)
            loaded_candidates += 1
    if loaded_candidates != candidate_limit:
        raise AssertionError("did not load the requested number of evaluation candidates")
    candidate_index = candidate_row_index(candidate_oids, candidate_retrieval_ids)

    mm_by_rid = np.lib.format.open_memmap(
        mm_rid_path, mode="w+", dtype=np.float32, shape=(max_rid + 1, mm_dim)
    )
    mm_valid_by_rid = np.lib.format.open_memmap(
        mm_rid_valid_path, mode="w+", dtype=np.bool_, shape=(max_rid + 1,)
    )
    candidate_mm = np.lib.format.open_memmap(
        candidate_mm_path, mode="w+", dtype=np.float32, shape=(candidate_limit, mm_dim)
    )
    candidate_valid = np.lib.format.open_memmap(
        candidate_valid_path, mode="w+", dtype=np.bool_, shape=(candidate_limit,)
    )
    mm_by_rid[:] = 0.0
    mm_valid_by_rid[:] = False
    candidate_mm[:] = 0.0
    candidate_valid[:] = False

    mm_dataset = ds.dataset(mm_path, format="parquet")
    if not {"anonymous_cid", "emb"}.issubset(mm_dataset.schema.names):
        raise ValueError("MM dataset must contain anonymous_cid and emb")
    processed_mm = 0
    null_embedding_rows = 0
    rid_aligned = 0
    candidate_aligned = 0
    seen_oids: set[int] = set()
    for batch in mm_dataset.scanner(columns=["anonymous_cid", "emb"], batch_size=8192).to_batches():
        if args.max_items is not None:
            remaining = args.max_items - processed_mm
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
        for oid_text, vector_value in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist()):
            try:
                oid = int(str(oid_text), 10)
            except ValueError as error:
                raise ValueError(f"MM anonymous_cid is not a decimal OID: {oid_text!r}") from error
            if oid in seen_oids:
                raise ValueError(f"duplicate MM OID: {oid}")
            seen_oids.add(oid)
            if vector_value is None:
                null_embedding_rows += 1
                processed_mm += 1
                continue
            vector = validate_mm_vector(vector_value, mm_dim)
            rid = oid_to_rid.get(oid)
            if rid is not None:
                rid_value = int(rid)
                mm_by_rid[rid_value] = vector
                mm_valid_by_rid[rid_value] = True
                rid_aligned += 1
            candidate_row = candidate_index.get(oid)
            if candidate_row is not None:
                candidate_mm[candidate_row] = vector
                candidate_valid[candidate_row] = True
                candidate_aligned += 1
            processed_mm += 1
        if processed_mm and processed_mm % 100000 == 0:
            logger.info("processed_mm_rows=%d", processed_mm)
    mm_by_rid.flush()
    mm_valid_by_rid.flush()
    candidate_mm.flush()
    candidate_valid.flush()
    rid_valid_count = int(np.count_nonzero(mm_valid_by_rid))
    candidate_valid_count = int(np.count_nonzero(candidate_valid))
    audit = {
        "schema_version": 1,
        "expected_dimension": mm_dim,
        "processed_mm_row_count": processed_mm,
        "null_embedding_row_count": null_embedding_rows,
        "null_embedding_policy": "missing MM; zeros in store with valid mask=0",
        "wrong_dimension_count": 0,
        "nan_or_inf_count": 0,
        "duplicate_oid_count": 0,
        "audit_complete": args.max_items is None,
        "history_rid_aligned_count": rid_aligned,
        "eval_candidate_aligned_count": candidate_aligned,
        "eval_candidate_missing_count": candidate_limit - candidate_valid_count,
        "oid_normalization": "decimal string -> int64",
    }
    save_json(audit, audit_path, args.overwrite)
    manifest = {
        "stage": "4.5",
        "schema_version": 1,
        "feature_protocol_version": config["feature_protocol_version"],
        "debug": bool(args.debug),
        "mm_dim": mm_dim,
        "mm_dtype": "float32",
        "mm_valid_count": rid_valid_count,
        "mm_missing_count": max_rid - rid_valid_count,
        "eval_candidate_count": candidate_limit,
        "eval_candidate_mm_valid_count": candidate_valid_count,
        "eval_candidate_mm_missing_count": candidate_limit - candidate_valid_count,
        "eval_candidate_array_alignment": "physical row order of eval_candidates.parquet",
        "retrieval_id_used_as_array_row": False,
        "missing_vector_storage": "zeros with separate valid mask",
        "l2_normalized": False,
        "source_scan_complete": args.max_items is None,
        "elapsed_seconds": timer.elapsed_seconds,
    }
    save_json(manifest, manifest_path, args.overwrite)
    logger.info("wrote MM rows=%d candidate_valid=%d elapsed_seconds=%.2f", processed_mm, candidate_valid_count, timer.elapsed_seconds)


if __name__ == "__main__":
    main()
