"""Stage 3.4: build the official-plus-target evaluation candidate pool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.evaluation_candidates import candidate_history_rid, ordered_candidate_additions
from src.data.stage3_runtime import (
    ParquetSink,
    common_parser_arguments,
    guard_outputs,
    load_config,
    load_item_mapping,
    require_paths,
    require_protocol_manifest,
    runtime_paths,
    save_json,
)


CANDIDATE_SCHEMA = pa.schema(
    [
        ("item_oid", pa.int64()),
        ("item_rid", pa.int64()),
        ("retrieval_id", pa.int64()),
        ("source", pa.string()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    common_parser_arguments(parser)
    return parser.parse_args()


def unique_targets(path: Path) -> set[int]:
    dataset = ds.dataset(path, format="parquet")
    if "target_item_oid" not in dataset.schema.names:
        raise ValueError(f"target_item_oid missing from {path}")
    values: set[int] = set()
    for batch in dataset.scanner(columns=["target_item_oid"], batch_size=65536).to_batches():
        values.update(int(value) for value in batch.column(0).to_pylist())
    return values


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root, output_root = runtime_paths(config, args.data_root, args.output_root, args.debug)
    val_path = output_root / "samples" / "val_all_targets.parquet"
    test_path = output_root / "samples" / "test_all_targets.parquet"
    official_dir = data_root / "candidate"
    split_manifest_path = output_root / "splits" / "split_manifest.json"
    require_paths(
        [val_path, test_path, split_manifest_path, official_dir, data_root / "indexer.pkl"]
    )
    protocol_version = str(config.get("protocol_version", "click_target_prefix_v2"))
    require_protocol_manifest(split_manifest_path, protocol_version)

    output_path = output_root / "candidates" / "eval_candidates.parquet"
    manifest_path = output_root / "candidates" / "eval_candidate_manifest.json"
    guard_outputs([output_path, manifest_path], args.overwrite)

    candidate_dataset = ds.dataset(official_dir, format="parquet")
    required = {"item_id", "retrieval_id"}
    if not required.issubset(candidate_dataset.schema.names):
        raise ValueError(f"official candidate missing columns: {sorted(required - set(candidate_dataset.schema.names))}")
    official_oids: list[int] = []
    official_retrieval_ids: list[int] = []
    for batch in candidate_dataset.scanner(
        columns=["item_id", "retrieval_id"], batch_size=65536
    ).to_batches():
        official_oids.extend(int(value) for value in batch.column(0).to_pylist())
        official_retrieval_ids.extend(int(value) for value in batch.column(1).to_pylist())
    if len(set(official_oids)) != len(official_oids):
        raise ValueError("official candidate contains duplicate OIDs")
    if len(set(official_retrieval_ids)) != len(official_retrieval_ids):
        raise ValueError("official candidate contains duplicate retrieval_ids")

    val_targets = unique_targets(val_path)
    test_targets = unique_targets(test_path)
    official_set = set(official_oids)
    val_additions, test_additions = ordered_candidate_additions(
        official_set, val_targets, test_targets
    )
    mapping = load_item_mapping(data_root / "indexer.pkl")
    next_retrieval_id = max(official_retrieval_ids, default=-1) + 1

    with ParquetSink(output_path, CANDIDATE_SCHEMA, args.overwrite) as sink:
        for start in range(0, len(official_oids), 65536):
            stop = start + 65536
            rows = [
                {
                    "item_oid": oid,
                    "item_rid": candidate_history_rid(oid, mapping),
                    "retrieval_id": retrieval_id,
                    "source": "official",
                }
                for oid, retrieval_id in zip(
                    official_oids[start:stop], official_retrieval_ids[start:stop]
                )
            ]
            sink.write_rows(rows)
        for source, additions in (
            ("validation_target", val_additions),
            ("test_target", test_additions),
        ):
            for start in range(0, len(additions), 65536):
                chunk = additions[start : start + 65536]
                rows = []
                for offset, oid in enumerate(chunk):
                    rid = candidate_history_rid(oid, mapping)
                    if rid is None:
                        raise ValueError(f"pseudo target OID {oid} has no historical RID")
                    rows.append(
                        {
                            "item_oid": oid,
                            "item_rid": rid,
                            "retrieval_id": next_retrieval_id + offset,
                            "source": source,
                        }
                    )
                sink.write_rows(rows)
                next_retrieval_id += len(chunk)
        final_count = sink.row_count

    final_set = official_set | val_targets | test_targets
    if final_count != len(final_set):
        raise AssertionError("written evaluation candidate count does not match the union")
    if not val_targets.issubset(final_set) or not test_targets.issubset(final_set):
        raise AssertionError("evaluation target coverage is not 100%")
    manifest = {
        "stage": "3.4",
        "schema_version": 2,
        "protocol_version": protocol_version,
        "debug": bool(args.debug),
        "union_definition": "official candidates union validation targets union test targets",
        "official_candidate_count": len(official_oids),
        "unique_validation_target_count": len(val_targets),
        "unique_test_target_count": len(test_targets),
        "validation_target_official_coverage_count": len(val_targets & official_set),
        "validation_target_official_coverage_ratio": (
            len(val_targets & official_set) / len(val_targets) if val_targets else None
        ),
        "test_target_official_coverage_count": len(test_targets & official_set),
        "test_target_official_coverage_ratio": (
            len(test_targets & official_set) / len(test_targets) if test_targets else None
        ),
        "added_validation_target_count": len(val_additions),
        "added_test_target_count": len(test_additions),
        "final_candidate_count": final_count,
        "validation_target_final_coverage_ratio": 1.0,
        "test_target_final_coverage_ratio": 1.0,
        "id_semantics": {
            "item_oid": "original anonymous item ID",
            "item_rid": "historical remapped ID; null is legal for history-unseen official candidates",
            "retrieval_id": "candidate-local retrieval ID",
        },
    }
    save_json(manifest, manifest_path, args.overwrite)
    print(f"wrote {final_count:,} evaluation candidates")


if __name__ == "__main__":
    main()
