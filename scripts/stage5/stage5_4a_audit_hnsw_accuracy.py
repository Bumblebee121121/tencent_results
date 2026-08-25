"""Stage 5.4a: compare HNSW Top-K with exact Inner Product Top-K on sampled users."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import faiss
import numpy as np
import pyarrow.dataset as ds
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.vanilla_two_tower import VanillaTwoTower
from src.recall.audits import reservoir_sample_rows
from src.recall.checkpoint import load_model_checkpoint
from src.recall.data import Stage5SequenceStore, TwoTowerCollator
from src.recall.faiss_utils import hnsw_retrieval_recall
from src.recall.runtime import (
    Timer, add_common_arguments, configure_logging, guard_outputs, load_config,
    require_contracts, require_paths, save_json, select_device, stage5_paths,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--device")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--fail-on-low-recall", action="store_true")
    return parser.parse_args()


def encode_sampled_users(rows, store, model, device, batch_size):
    collator = TwoTowerCollator(store, negative_sampler=None, require_seen_target=False)
    vectors = []
    retained_sample_ids = []
    zero_vector_count = 0
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = collator(rows[start : start + batch_size])
            if not batch["rows"]:
                continue
            query = model.encode_user(
                batch["history_tokens"].to(device), batch["history_offsets"].to(device),
            ).cpu().numpy().astype(np.float32)
            valid = np.linalg.norm(query, axis=1) > 0
            zero_vector_count += int(np.count_nonzero(~valid))
            if np.any(valid):
                vectors.append(query[valid])
                retained_sample_ids.extend(
                    int(row["sample_id"]) for row, keep in zip(batch["rows"], valid) if keep
                )
    if not vectors:
        raise ValueError("all sampled user vectors are zero after ignoring PAD and UNK")
    return np.ascontiguousarray(np.concatenate(vectors), dtype=np.float32), retained_sample_ids, zero_vector_count


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, stage3_root, stage4_root, output_root, log_root = stage5_paths(config, args.debug)
    logger = configure_logging(log_root, "stage5_4a_audit_hnsw_accuracy")
    require_contracts(stage3_root, stage4_root, config)
    timer = Timer()
    root = output_root / "two_tower"
    checkpoint_path = root / "checkpoints" / "best.pt"
    index_path = root / "faiss.index"
    embeddings_path = root / "item_embeddings.npy"
    indexed_path = root / "indexed_candidates.parquet"
    output_path = output_root / "audits" / "hnsw_accuracy_audit.json"
    require_paths([checkpoint_path, index_path, embeddings_path, indexed_path])
    guard_outputs([output_path], args.overwrite)

    section = config["hnsw_accuracy_audit"]
    sample_size = args.sample_size or int(section["sample_size"])
    if args.debug and args.sample_size is None:
        sample_size = int(config["debug"]["hnsw_audit_samples"])
    split_name = str(section["split"])
    split_paths = {
        "validation_primary": stage3_root / "samples" / "val_primary.parquet",
        "test_primary": stage3_root / "samples" / "test_primary.parquet",
    }
    if split_name not in split_paths:
        raise ValueError("HNSW audit split must be validation_primary or test_primary")
    sampled_rows, source_count = reservoir_sample_rows(
        split_paths[split_name], sample_size, int(config["seed"]), int(config["scan_batch_size"])
    )

    store = Stage5SequenceStore(stage4_root)
    model_section = config["two_tower"]
    model = VanillaTwoTower(store.rid_to_token.size + 1, int(model_section["embedding_dim"]))
    checkpoint = load_model_checkpoint(checkpoint_path, model, map_location="cpu")
    if checkpoint["protocols"].get("recall") != config["recall_protocol_version"]:
        raise ValueError("checkpoint recall protocol mismatch")
    device = select_device(args.device)
    model.to(device).eval()
    queries, sample_ids, zero_count = encode_sampled_users(
        sampled_rows, store, model, device, int(config["faiss"]["query_batch_size"])
    )

    embeddings = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    indexed_count = ds.dataset(indexed_path, format="parquet").count_rows()
    if embeddings.shape[0] != indexed_count:
        raise ValueError("item embedding rows do not align with indexed candidates")
    hnsw = faiss.read_index(str(index_path))
    if int(hnsw.ntotal) != indexed_count:
        raise ValueError("HNSW rows do not align with indexed candidates")
    hnsw.hnsw.efSearch = int(config["faiss"]["efSearch"])
    exact = faiss.IndexFlatIP(int(embeddings.shape[1]))
    exact.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    ks = sorted(set(map(int, section["ks"])))
    max_k = max(ks)
    if max_k > indexed_count:
        raise ValueError("HNSW audit K exceeds indexed candidate count")
    _, approximate_rows = hnsw.search(queries, max_k)
    _, exact_rows = exact.search(queries, max_k)
    metrics = hnsw_retrieval_recall(approximate_rows, exact_rows, ks)
    thresholds = {int(k): float(value) for k, value in section["minimum_mean_recall"].items()}
    checks = {
        f"@{k}": {
            "minimum_required_mean_recall": thresholds[k],
            "passed": float(metrics[f"@{k}"]["mean_recall"]) >= thresholds[k],
        }
        for k in ks
    }
    passed = all(value["passed"] for value in checks.values())
    report = {
        "stage": "5.4a", "schema_version": 1,
        "recall_protocol_version": config["recall_protocol_version"],
        "debug": bool(args.debug), "split": split_name, "seed": int(config["seed"]),
        "source_sample_count": source_count, "requested_sample_size": sample_size,
        "sampled_row_count": len(sampled_rows), "valid_query_count": len(sample_ids),
        "zero_user_vector_count": zero_count,
        "zero_vector_policy": "exclude from ANN-vs-exact audit and report count",
        "comparison": "raw HNSW Top-K rows vs exact IndexFlatIP Top-K rows; no history filtering",
        "configured_efSearch": int(config["faiss"]["efSearch"]),
        "indexed_candidate_count": int(indexed_count), "metrics": metrics,
        "threshold_checks": checks, "passed": passed, "elapsed_seconds": timer.elapsed_seconds,
    }
    save_json(report, output_path, args.overwrite)
    logger.info("HNSW accuracy audit passed=%s valid_queries=%d elapsed_seconds=%.2f", passed, len(sample_ids), timer.elapsed_seconds)
    if args.fail_on_low_recall and not passed:
        raise RuntimeError(f"HNSW retrieval recall is below configured threshold; inspect {output_path}")


if __name__ == "__main__":
    main()
