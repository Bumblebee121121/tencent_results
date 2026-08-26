"""Configuration and immutable upstream contracts for Stage 6 entry points."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.data.stage3_runtime import guard_outputs, require_paths, save_csv, save_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "stage6.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Stage 6 config must be a YAML mapping")
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def stage6_paths(config: Mapping[str, Any], debug: bool) -> dict[str, Path]:
    paths = {name: resolve_path(config.get(name, default)) for name, default in {
        "data_root": "data/TencentGR-1M", "stage3_root": "artifacts/stage3",
        "stage4_root": "artifacts/stage4", "stage5_root": "artifacts/stage5",
        "output_root": "artifacts/stage6", "log_root": "logs/stage6",
    }.items()}
    if debug:
        paths["output_root"] = paths["output_root"].parent / "stage6_debug"
        paths["log_root"] = paths["log_root"].parent / "stage6_debug"
    stage5 = paths["stage5_root"].resolve()
    output = paths["output_root"].resolve()
    if output == stage5 or stage5 in output.parents:
        raise ValueError("Stage 6 output_root must never be Stage 5 or a child of it")
    return paths


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def require_contracts(paths: Mapping[str, Path], config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    files = {
        "samples": paths["stage3_root"] / "samples" / "sample_manifest.json",
        "splits": paths["stage3_root"] / "splits" / "split_manifest.json",
        "candidates": paths["stage3_root"] / "candidates" / "eval_candidate_manifest.json",
        "stage4": paths["stage4_root"] / "manifests" / "stage4_manifest.json",
        "side": paths["stage4_root"] / "manifests" / "item_side_manifest.json",
        "mm": paths["stage4_root"] / "manifests" / "multimodal_store_manifest.json",
        "stage5": paths["stage5_root"] / "manifests" / "stage5_manifest.json",
        "itemcf": paths["stage5_root"] / "itemcf" / "metrics.json",
        "two_tower": paths["stage5_root"] / "two_tower" / "metrics.json",
        "complementarity": paths["stage5_root"] / "reports" / "channel_complementarity.json",
    }
    require_paths(files.values())
    values = {name: load_json(path) for name, path in files.items()}
    expected3 = str(config["stage3_protocol_version"])
    for name in ("samples", "splits", "candidates"):
        if values[name].get("protocol_version") != expected3:
            raise ValueError(f"Stage 3 {name} protocol mismatch")
    stage4 = values["stage4"]
    if stage4.get("feature_protocol_version") != config["stage4_protocol_version"]:
        raise ValueError("Stage 4 protocol mismatch")
    if stage4.get("stage3_protocol_version") != expected3:
        raise ValueError("Stage 4 was built from another Stage 3 protocol")
    cutoff = int(values["splits"]["train_raw_event_cutoff_exclusive"])
    if int(stage4["stage3_train_cutoff"]) != cutoff:
        raise ValueError("Stage 3/4 train cutoff mismatch")
    candidate_count = int(values["candidates"]["final_candidate_count"])
    if int(values["side"]["eval_candidate_count"]) != candidate_count or int(values["mm"]["eval_candidate_count"]) != candidate_count:
        raise ValueError("Stage 3/4 candidate counts do not align")
    if values["stage5"].get("recall_protocol_version") != config["stage5_recall_protocol_version"]:
        raise ValueError("Stage 5 recall protocol mismatch")
    if not bool(stage4.get("feature_dataset_smoke_passed")):
        raise ValueError("Stage 4 feature smoke did not pass")
    return values


def configured_session(config: Mapping[str, Any], debug: bool) -> tuple[int, int | None, int | None]:
    if debug:
        section = config["debug"]
        return int(section["session_gap_seconds"]), int(section["short_max_events"]), int(section["long_max_events"])
    short = config["user_tower"]["short_session"]
    gap = short.get("session_gap_seconds")
    if gap is None:
        raise ValueError("formal run blocked: select session_gap_seconds on Validation and freeze configs/stage6.yaml")
    return int(gap), None if short.get("max_events") is None else int(short["max_events"]), None if config["user_tower"]["long_history"].get("max_events") is None else int(config["user_tower"]["long_history"]["max_events"])


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def configure_logging(root: Path, name: str) -> logging.Logger:
    root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name); logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(root / f"{name}.log", encoding="utf-8")):
        handler.setFormatter(formatter); logger.addHandler(handler)
    logger.propagate = False
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def select_device(value: str | None):
    import torch
    return torch.device(value or ("cuda" if torch.cuda.is_available() else "cpu"))


__all__ = [
    "PROJECT_ROOT", "DEFAULT_CONFIG", "add_common_arguments", "configured_session",
    "configure_logging", "guard_outputs", "load_config", "load_json", "require_contracts",
    "require_paths", "save_csv", "save_json", "seed_everything", "select_device", "stage6_paths",
]

