"""Configuration, mapping and output helpers for Stage 3 entry points."""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage3.yaml"


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"config does not exist: {config_path.resolve()}")
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Stage 3 config must be a YAML mapping")
    return data


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def runtime_paths(
    config: Mapping[str, Any],
    data_root_override: str | None,
    output_root_override: str | None,
    debug: bool,
) -> tuple[Path, Path]:
    data_root = resolve_path(data_root_override or config.get("data_root", "data/TencentGR-1M"))
    output_root = resolve_path(output_root_override or config.get("output_root", "artifacts/stage3"))
    if debug:
        output_root = output_root.parent / "stage3_debug"
    return data_root, output_root


def require_paths(paths: Iterable[Path]) -> None:
    missing = [str(path.resolve()) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n- " + "\n- ".join(missing))


def guard_output(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"output already exists (pass --overwrite): {path.resolve()}")
    path.parent.mkdir(parents=True, exist_ok=True)


def guard_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    paths = list(paths)
    existing = [str(path.resolve()) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "outputs already exist (pass --overwrite):\n- " + "\n- ".join(existing)
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def save_json(data: Mapping[str, Any], path: Path, overwrite: bool) -> None:
    guard_output(path, overwrite)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def save_csv(
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
    path: Path,
    overwrite: bool,
) -> None:
    guard_output(path, overwrite)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_item_mapping(indexer_path: Path) -> dict[int, int]:
    require_paths([indexer_path])
    with indexer_path.open("rb") as handle:
        indexer = pickle.load(handle)
    if not isinstance(indexer, Mapping) or "i" not in indexer:
        raise ValueError("indexer.pkl does not contain indexer['i']")
    mapping = indexer["i"]
    return mapping if isinstance(mapping, dict) else dict(mapping)


def iter_sequence_rows(seq_dir: Path, batch_size: int, max_users: int | None):
    """Yield complete user rows without ever splitting a user sequence."""

    require_paths([seq_dir])
    dataset = ds.dataset(seq_dir, format="parquet")
    required = {"user_id", "seq"}
    if not required.issubset(dataset.schema.names):
        raise ValueError(f"seq dataset is missing columns: {sorted(required - set(dataset.schema.names))}")
    processed = 0
    scanner = dataset.scanner(columns=["user_id", "seq"], batch_size=batch_size)
    for batch in scanner.to_batches():
        if max_users is not None:
            remaining = max_users - processed
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
        user_ids = batch.column(0).to_pylist()
        sequences = batch.column(1).to_pylist()
        processed += batch.num_rows
        yield user_ids, sequences


class ParquetSink:
    """Small streaming Parquet writer with explicit overwrite protection."""

    def __init__(self, path: Path, schema: pa.Schema, overwrite: bool):
        guard_output(path, overwrite)
        self.path = path
        self.schema = schema
        self.writer = pq.ParquetWriter(path, schema, compression="snappy")
        self.row_count = 0

    def write_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if not rows:
            return
        table = pa.Table.from_pylist(list(rows), schema=self.schema)
        self.write_table(table)

    def write_table(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        if table.schema != self.schema:
            table = table.cast(self.schema)
        self.writer.write_table(table)
        self.row_count += table.num_rows

    def close(self) -> None:
        self.writer.close()

    def __enter__(self) -> "ParquetSink":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def common_parser_arguments(parser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", help="override config data_root")
    parser.add_argument("--output-root", help="override config output_root")
    parser.add_argument("--debug", action="store_true", help="write under artifacts/stage3_debug")
    parser.add_argument("--overwrite", action="store_true")
