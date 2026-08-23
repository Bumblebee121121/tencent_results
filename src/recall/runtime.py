"""Configuration and artifact contracts shared by Stage 5 entry points."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.data.stage3_runtime import guard_outputs, require_paths, save_csv, save_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage5.yaml"


class Timer:
    def __init__(self) -> None:
        self.started = time.perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        return round(time.perf_counter() - self.started, 3)


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Stage 5 config must be a YAML mapping")
    return config


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def stage5_paths(config: Mapping[str, Any], debug: bool) -> tuple[Path, Path, Path, Path, Path]:
    data_root = resolve_path(config.get("data_root", "data/TencentGR-1M"))
    stage3_root = resolve_path(config.get("stage3_root", "artifacts/stage3"))
    stage4_root = resolve_path(config.get("stage4_root", "artifacts/stage4"))
    output_root = resolve_path(config.get("output_root", "artifacts/stage5"))
    log_root = resolve_path(config.get("log_root", "logs/stage5"))
    if debug:
        output_root = output_root.parent / "stage5_debug"
        log_root = log_root.parent / "stage5_debug"
    return data_root, stage3_root, stage4_root, output_root, log_root


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return value


def require_contracts(stage3_root: Path, stage4_root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    paths = {
        "samples": stage3_root / "samples" / "sample_manifest.json",
        "splits": stage3_root / "splits" / "split_manifest.json",
        "candidates": stage3_root / "candidates" / "eval_candidate_manifest.json",
        "stage4": stage4_root / "manifests" / "stage4_manifest.json",
    }
    require_paths(paths.values())
    values = {name: load_json(path) for name, path in paths.items()}
    expected3 = str(config["stage3_protocol_version"])
    for name in ("samples", "splits", "candidates"):
        if values[name].get("protocol_version") != expected3:
            raise ValueError(f"Stage 3 {name} protocol mismatch")
    stage4 = values["stage4"]
    if stage4.get("feature_protocol_version") != str(config["stage4_protocol_version"]):
        raise ValueError("Stage 4 feature protocol mismatch")
    if stage4.get("stage3_protocol_version") != expected3:
        raise ValueError("Stage 4 was built from a different Stage 3 protocol")
    if int(stage4["stage3_train_cutoff"]) != int(values["splits"]["train_raw_event_cutoff_exclusive"]):
        raise ValueError("Stage 3/4 train cutoff mismatch")
    if not bool(stage4.get("feature_dataset_smoke_passed")):
        raise ValueError("Stage 4 feature dataset smoke was not completed successfully")
    return values


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def configure_logging(log_root: Path, name: str) -> logging.Logger:
    log_root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_root / f"{name}.log", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def select_device(requested: str | None = None):
    import torch

    if requested is not None:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


__all__ = [
    "PROJECT_ROOT", "Timer", "add_common_arguments", "configure_logging", "guard_outputs",
    "load_config", "load_json", "require_contracts", "require_paths", "save_csv", "save_json",
    "seed_everything", "select_device", "stage5_paths",
]
