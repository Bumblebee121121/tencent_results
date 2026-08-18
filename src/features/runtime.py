"""Shared Stage 4 configuration, paths, logging and manifest helpers."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from src.data.stage3_runtime import guard_outputs, require_paths, save_csv, save_json

from .id_semantics import validate_special_tokens


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage4.yaml"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_stage4_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Stage 4 config must be a YAML mapping")
    validate_special_tokens(config.get("special_tokens", {}))
    if config.get("use_candidate_cold_start_as_model_feature") is not False:
        raise ValueError("candidate cold_start must remain audit-only")
    if config.get("materialize_history_per_sample") is not False:
        raise ValueError("Stage 4 must not materialize history per sample")
    return config


def stage4_paths(config: Mapping[str, Any], debug: bool) -> tuple[Path, Path, Path, Path]:
    data_root = resolve_path(str(config["data_root"]))
    stage3_root = resolve_path(str(config["stage3_root"]))
    output_root = resolve_path(str(config["output_root"]))
    log_root = resolve_path(str(config["log_root"]))
    if debug:
        stage3_root = stage3_root.parent / f"{stage3_root.name}_debug"
        output_root = output_root.parent / f"{output_root.name}_debug"
    return data_root, stage3_root, output_root, log_root


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def configure_logging(log_root: Path, stage_name: str, debug: bool) -> logging.Logger:
    log_root.mkdir(parents=True, exist_ok=True)
    suffix = "_debug" if debug else ""
    log_path = log_root / f"{stage_name}{suffix}.log"
    logger = logging.getLogger(stage_name + suffix)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8", mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.info("log_path=%s", log_path.resolve())
    return logger


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON manifest must be an object: {path}")
    return value


def require_stage3_contracts(stage3_root: Path, protocol_version: str) -> dict[str, dict[str, Any]]:
    paths = {
        "samples": stage3_root / "samples" / "sample_manifest.json",
        "splits": stage3_root / "splits" / "split_manifest.json",
        "candidates": stage3_root / "candidates" / "eval_candidate_manifest.json",
        "strength": stage3_root / "item_strength" / "item_strength_thresholds.json",
        "evaluation": stage3_root / "evaluation" / "evaluation_protocol.json",
    }
    require_paths(paths.values())
    manifests = {name: load_json(path) for name, path in paths.items()}
    for name, manifest in manifests.items():
        if manifest.get("protocol_version") != protocol_version:
            raise ValueError(
                f"Stage 3 {name} protocol mismatch: "
                f"expected {protocol_version!r}, found {manifest.get('protocol_version')!r}"
            )
    cutoff = int(manifests["splits"]["train_raw_event_cutoff_exclusive"])
    if int(manifests["strength"]["train_raw_event_cutoff_exclusive"]) != cutoff:
        raise ValueError("Stage 3 split and item-strength cutoffs disagree")
    return manifests


class Timer:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_CONFIG",
    "Timer",
    "add_common_arguments",
    "configure_logging",
    "guard_outputs",
    "load_json",
    "load_stage4_config",
    "require_paths",
    "require_stage3_contracts",
    "save_csv",
    "save_json",
    "stage4_paths",
]
