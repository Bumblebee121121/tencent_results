"""Stage 4.3: build train-fitted user scalar/list categorical features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.stage3_runtime import ParquetSink
from src.features.categorical_encoder import CategoricalVocabulary
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
    parser.add_argument("--max-users", type=int)
    return parser.parse_args()


def unique_user_ids(path: Path) -> set[int]:
    result: set[int] = set()
    for batch in ds.dataset(path, format="parquet").scanner(
        columns=["user_id"], batch_size=65536
    ).to_batches():
        result.update(int(value) for value in batch.column(0).to_pylist())
    return result


def output_schema(scalar_fields: list[str], list_fields: list[str]) -> pa.Schema:
    fields: list[tuple[str, pa.DataType]] = [("user_id", pa.int64())]
    for name in scalar_fields:
        fields.extend(
            [
                (f"f{name}_token", pa.int32()),
                (f"f{name}_missing", pa.bool_()),
                (f"f{name}_oov", pa.bool_()),
            ]
        )
    for name in list_fields:
        fields.extend(
            [
                (f"f{name}_tokens", pa.list_(pa.int32())),
                (f"f{name}_missing", pa.bool_()),
                (f"f{name}_oov", pa.list_(pa.bool_())),
            ]
        )
    return pa.schema(fields)


def main() -> None:
    args = parse_args()
    config = load_stage4_config(args.config)
    data_root, stage3_root, output_root, log_root = stage4_paths(config, args.debug)
    logger = configure_logging(log_root, "stage4_3_build_user_features", args.debug)
    timer = Timer()
    require_stage3_contracts(stage3_root, str(config["stage3_protocol_version"]))
    scalar_fields = [str(value) for value in config["user_scalar_features"]]
    list_fields = [str(value) for value in config["user_list_features"]]
    expected_fields = scalar_fields + list_fields
    user_feat_path = data_root / "user_feat"
    sample_paths = {
        "train": stage3_root / "samples" / "train_samples.parquet",
        "validation": stage3_root / "samples" / "val_primary.parquet",
        "test": stage3_root / "samples" / "test_primary.parquet",
    }
    require_paths([user_feat_path, *sample_paths.values()])
    dataset = ds.dataset(user_feat_path, format="parquet")
    required = {"user_id", *expected_fields}
    if not required.issubset(dataset.schema.names):
        raise ValueError(f"user_feat missing fields: {sorted(required - set(dataset.schema.names))}")
    for field in scalar_fields:
        if not pa.types.is_int64(dataset.schema.field(field).type):
            raise TypeError(f"user scalar feature {field} must be int64")
    for field in list_fields:
        if not pa.types.is_list(dataset.schema.field(field).type):
            raise TypeError(f"user list feature {field} must be list")

    train_users = unique_user_ids(sample_paths["train"])
    relevant_users = set(train_users)
    relevant_users.update(unique_user_ids(sample_paths["validation"]))
    relevant_users.update(unique_user_ids(sample_paths["test"]))
    if args.max_users is not None:
        if args.max_users <= 0:
            raise ValueError("--max-users must be positive")
        relevant_users = set(sorted(relevant_users)[: args.max_users])
        train_users &= relevant_users

    feature_path = output_root / "feature_store" / "user_features.parquet"
    manifest_path = output_root / "manifests" / "user_feature_manifest.json"
    coverage_path = output_root / "audits" / "user_feature_vocab_coverage.csv"
    vocab_dir = output_root / "mappings" / "vocab"
    vocab_paths = {field: vocab_dir / f"user_f{field}.npy" for field in expected_fields}
    guard_outputs(
        [feature_path, manifest_path, coverage_path, *vocab_paths.values()],
        args.overwrite,
    )

    known_sets = {field: set() for field in expected_fields}
    scanner = dataset.scanner(columns=["user_id", *expected_fields], batch_size=8192)
    for batch in scanner.to_batches():
        columns = {name: batch.column(index + 1).to_pylist() for index, name in enumerate(expected_fields)}
        for row_index, user_value in enumerate(batch.column(0).to_pylist()):
            if int(user_value) not in train_users:
                continue
            for field in scalar_fields:
                value = columns[field][row_index]
                if value is not None:
                    known_sets[field].add(int(value))
            for field in list_fields:
                values = columns[field][row_index]
                if values is not None:
                    known_sets[field].update(int(value) for value in values if value is not None)
    vocabularies = {
        field: CategoricalVocabulary(sorted(known_sets[field])) for field in expected_fields
    }
    for field, vocab in vocabularies.items():
        vocab.save(vocab_paths[field])

    counters = {
        field: {"row_count": 0, "value_count": 0, "missing_count": 0, "oov_count": 0, "list_lengths": []}
        for field in expected_fields
    }
    found_users: set[int] = set()
    with ParquetSink(feature_path, output_schema(scalar_fields, list_fields), args.overwrite) as sink:
        scanner = dataset.scanner(columns=["user_id", *expected_fields], batch_size=8192)
        for batch in scanner.to_batches():
            columns = {name: batch.column(index + 1).to_pylist() for index, name in enumerate(expected_fields)}
            rows = []
            for row_index, user_value in enumerate(batch.column(0).to_pylist()):
                user_id = int(user_value)
                if user_id not in relevant_users:
                    continue
                found_users.add(user_id)
                row: dict[str, object] = {"user_id": user_id}
                for field in scalar_fields:
                    encoded = vocabularies[field].encode(columns[field][row_index])
                    row[f"f{field}_token"] = encoded.token
                    row[f"f{field}_missing"] = encoded.missing
                    row[f"f{field}_oov"] = encoded.oov
                    counters[field]["value_count"] += 1
                    counters[field]["row_count"] += 1
                    counters[field]["missing_count"] += int(encoded.missing)
                    counters[field]["oov_count"] += int(encoded.oov)
                for field in list_fields:
                    tokens, missing, oov = vocabularies[field].encode_list(columns[field][row_index])
                    row[f"f{field}_tokens"] = tokens
                    row[f"f{field}_missing"] = missing
                    row[f"f{field}_oov"] = oov
                    counters[field]["value_count"] += len(tokens)
                    counters[field]["row_count"] += 1
                    counters[field]["missing_count"] += int(missing)
                    counters[field]["oov_count"] += sum(oov)
                    counters[field]["list_lengths"].append(len(tokens))
                rows.append(row)
            sink.write_rows(rows)
        missing_users = sorted(relevant_users - found_users)
        for start in range(0, len(missing_users), 8192):
            rows = []
            for user_id in missing_users[start : start + 8192]:
                row = {"user_id": user_id}
                for field in scalar_fields:
                    row.update({f"f{field}_token": 1, f"f{field}_missing": True, f"f{field}_oov": False})
                    counters[field]["value_count"] += 1
                    counters[field]["row_count"] += 1
                    counters[field]["missing_count"] += 1
                for field in list_fields:
                    row.update({f"f{field}_tokens": [], f"f{field}_missing": True, f"f{field}_oov": []})
                    counters[field]["row_count"] += 1
                    counters[field]["missing_count"] += 1
                    counters[field]["list_lengths"].append(0)
                rows.append(row)
            sink.write_rows(rows)
        row_count = sink.row_count

    coverage_rows = []
    field_manifest = {}
    for field in expected_fields:
        counter = counters[field]
        value_count = int(counter["value_count"])
        missing_count = int(counter["missing_count"])
        oov_count = int(counter["oov_count"])
        coverage_rows.append(
            {
                "entity": "user",
                "feature": field,
                "field_type": "scalar" if field in scalar_fields else "list",
                "train_known_vocab_size": vocabularies[field].known_size,
                "row_count": int(counter["row_count"]),
                "value_count": value_count,
                "missing_count": missing_count,
                "missing_rate": missing_count / int(counter["row_count"]) if int(counter["row_count"]) else 0.0,
                "oov_count": oov_count,
                "oov_rate": oov_count / value_count if value_count else 0.0,
            }
        )
        lengths = counter["list_lengths"]
        field_manifest[field] = {
            "field_type": "scalar" if field in scalar_fields else "list",
            "train_known_vocab_size": vocabularies[field].known_size,
            "missing_count": missing_count,
            "oov_count": oov_count,
            "list_length_min": min(lengths) if lengths else None,
            "list_length_max": max(lengths) if lengths else None,
            "list_length_mean": sum(lengths) / len(lengths) if lengths else None,
            "vocab_path": str(vocab_paths[field].relative_to(output_root)),
        }
    save_csv(coverage_rows, list(coverage_rows[0]), coverage_path, args.overwrite)
    manifest = {
        "stage": "4.3",
        "schema_version": 1,
        "feature_protocol_version": config["feature_protocol_version"],
        "debug": bool(args.debug),
        "row_count": row_count,
        "train_scope_user_count": len(train_users),
        "relevant_user_count": len(relevant_users),
        "missing_user_feature_row_count": len(relevant_users - found_users),
        "fields": field_manifest,
        "elapsed_seconds": timer.elapsed_seconds,
    }
    save_json(manifest, manifest_path, args.overwrite)
    logger.info("wrote user rows=%d missing_rows=%d elapsed_seconds=%.2f", row_count, len(relevant_users - found_users), timer.elapsed_seconds)


if __name__ == "__main__":
    main()
