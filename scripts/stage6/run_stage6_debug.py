"""Run the complete Stage 6 Debug pipeline in dependency-safe order.

This runner intentionally excludes Formal-only Session Gap selection and Stage 6
finalization. Debug uses the provisional 1800-second gap, one epoch, best-loss
checkpoints, and limited Train/Validation/Test samples from ``configs/stage6.yaml``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recall.stage6_runtime import load_config, stage6_paths


@dataclass(frozen=True)
class DebugStep:
    name: str
    command: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "stage6.yaml")
    parser.add_argument("--device", help="PyTorch device, for example cuda or cpu")
    parser.add_argument("--overwrite", action="store_true", help="rebuild outputs owned by selected steps")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--start-at", help="resume from this named step, inclusive")
    parser.add_argument("--stop-after", help="stop after this named step, inclusive")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    return parser.parse_args()


def build_steps(args: argparse.Namespace) -> list[DebugStep]:
    python = sys.executable
    config = str(args.config.resolve())

    def script_step(name: str, filename: str, *extra: str, device: bool = False) -> DebugStep:
        command = [
            python, "-X", "utf8", "-u",
            str(PROJECT_ROOT / "scripts" / "stage6" / filename),
            "--config", config, "--debug",
        ]
        if args.overwrite:
            command.append("--overwrite")
        if device and args.device:
            command.extend(["--device", args.device])
        command.extend(extra)
        return DebugStep(name, tuple(command))

    steps = []
    if not args.skip_tests:
        steps.append(DebugStep("unit_tests", (
            python, "-X", "utf8", "-m", "unittest", "discover",
            "-s", "tests/stage6", "-v",
        )))
    steps.extend([
        script_step("contract_audit", "stage6_0_contract_audit.py"),
        script_step("sequence_smoke", "stage6_1_sequence_adapter_smoke.py"),
    ])
    for variant in ("U1", "U2", "U3"):
        lower = variant.lower()
        steps.extend([
            script_step(f"{lower}_train", "stage6_2_train_user_variants.py", "--variant", variant, device=True),
            script_step(f"{lower}_index", "stage6_5_build_indexes.py", "--variant", variant, device=True),
            script_step(f"{lower}_evaluate", "stage6_6_evaluate_variants.py", "--variant", variant, device=True),
        ])
    for variant in ("I1", "I2", "I3"):
        lower = variant.lower()
        steps.extend([
            script_step(f"{lower}_train", "stage6_3_train_item_variants.py", "--variant", variant, device=True),
            script_step(f"{lower}_index", "stage6_5_build_indexes.py", "--variant", variant, device=True),
            script_step(f"{lower}_evaluate", "stage6_6_evaluate_variants.py", "--variant", variant, device=True),
        ])
    steps.extend([
        script_step("e1_train", "stage6_4_train_enhanced_two_tower.py", device=True),
        script_step("e1_index", "stage6_5_build_indexes.py", "--variant", "E1", device=True),
        script_step("e1_evaluate", "stage6_6_evaluate_variants.py", "--variant", "E1", device=True),
        script_step("ablation", "stage6_7_compare_ablation.py"),
        script_step("complementarity", "stage6_8_channel_complementarity.py"),
        script_step("fusion", "stage6_9_fuse_recall.py"),
    ])
    return steps


def select_steps(steps: Sequence[DebugStep], start_at: str | None, stop_after: str | None) -> list[DebugStep]:
    names = [step.name for step in steps]
    if start_at is not None and start_at not in names:
        raise ValueError(f"unknown --start-at={start_at!r}; available: {', '.join(names)}")
    if stop_after is not None and stop_after not in names:
        raise ValueError(f"unknown --stop-after={stop_after!r}; available: {', '.join(names)}")
    start = names.index(start_at) if start_at is not None else 0
    stop = names.index(stop_after) + 1 if stop_after is not None else len(steps)
    if start >= stop:
        raise ValueError("--start-at must not occur after --stop-after")
    return list(steps[start:stop])


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = stage6_paths(config, True)
    all_steps = build_steps(args)
    steps = select_steps(all_steps, args.start_at, args.stop_after)
    if args.dry_run:
        for index, step in enumerate(steps, start=1):
            print(f"[{index:02d}/{len(steps):02d}] {step.name}")
            print(subprocess.list2cmdline(step.command))
        return

    manifest_path = paths["output_root"] / "manifests" / "debug_run_manifest.json"
    manifest: dict[str, object] = {
        "stage": "6.debug.full",
        "protocol_version": config["stage6_protocol_version"],
        "debug": True,
        "debug_results_must_not_be_used_for_conclusions": True,
        "python": sys.executable,
        "config": str(args.config.resolve()),
        "device_override": args.device,
        "overwrite": bool(args.overwrite),
        "started_at": datetime.now().astimezone().isoformat(),
        "selected_steps": [step.name for step in steps],
        "steps": [],
        "status": "running",
    }
    write_manifest(manifest_path, manifest)
    pipeline_started = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    for index, step in enumerate(steps, start=1):
        print(f"\n[{index:02d}/{len(steps):02d}] START {step.name}", flush=True)
        print(subprocess.list2cmdline(step.command), flush=True)
        started = time.perf_counter()
        completed = subprocess.run(
            step.command,
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )
        elapsed = round(time.perf_counter() - started, 3)
        step_result = {
            "name": step.name,
            "returncode": int(completed.returncode),
            "elapsed_seconds": elapsed,
            "status": "completed" if completed.returncode == 0 else "failed",
        }
        manifest["steps"].append(step_result)
        manifest["elapsed_seconds"] = round(time.perf_counter() - pipeline_started, 3)
        if completed.returncode != 0:
            manifest["status"] = "failed"
            manifest["failed_step"] = step.name
            manifest["finished_at"] = datetime.now().astimezone().isoformat()
            write_manifest(manifest_path, manifest)
            raise SystemExit(completed.returncode)
        write_manifest(manifest_path, manifest)
        print(f"[{index:02d}/{len(steps):02d}] DONE  {step.name} ({elapsed:.1f}s)", flush=True)

    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    manifest["elapsed_seconds"] = round(time.perf_counter() - pipeline_started, 3)
    write_manifest(manifest_path, manifest)
    print(f"\nStage 6 full Debug completed in {manifest['elapsed_seconds']}s", flush=True)
    print(f"Manifest: {manifest_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
