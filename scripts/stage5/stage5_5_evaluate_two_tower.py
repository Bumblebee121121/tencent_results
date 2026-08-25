"""Stage 5.5: evaluate Two-Tower with FAISS while retaining unseen targets in the denominator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.item_strength import classify_strength
from src.models.vanilla_two_tower import VanillaTwoTower
from src.recall.checkpoint import load_model_checkpoint
from src.recall.data import ParquetSampleIterableDataset, Stage5SequenceStore, TwoTowerCollator
from src.recall.evaluation import metrics_from_ranks
from src.recall.faiss_utils import filter_history_from_faiss_rows, search_nonzero_queries
from src.recall.runtime import (
    Timer, add_common_arguments, configure_logging, guard_outputs, load_config,
    load_json, require_contracts, require_paths, save_json, select_device, stage5_paths,
)


RANK_SCHEMA = pa.schema(
    [
        ("sample_id", pa.int64()), ("target_item_oid", pa.int64()),
        ("target_item_rid", pa.int64()), ("target_strength_group", pa.string()),
        ("target_rank", pa.int32()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--device")
    parser.add_argument("--max-samples", type=int)
    return parser.parse_args()


def evaluate_split(
    path, output_path, model, index, indexed_rids, store, config, device,
    max_samples, p50, p90, exclude_history_items,
):
    dataset = ParquetSampleIterableDataset(path, max_rows=max_samples)
    collator = TwoTowerCollator(store, negative_sampler=None, require_seen_target=False)
    loader = DataLoader(dataset, batch_size=int(config["faiss"]["query_batch_size"]), collate_fn=collator, num_workers=0)
    writer = pq.ParquetWriter(output_path, RANK_SCHEMA, compression="snappy")
    ranks: list[int | None] = []
    groups: list[str] = []
    zero_user_vector_count = 0
    max_k = max(map(int, config["recall_ks"]))
    try:
        with torch.no_grad():
            for batch in loader:
                if not batch["rows"]:
                    continue
                history_tokens = batch["history_tokens"].to(device)
                history_offsets = batch["history_offsets"].to(device)
                query = np.ascontiguousarray(
                    model.encode_user(history_tokens, history_offsets).cpu().numpy(), dtype=np.float32,
                )
                histories = [store.history(row).item_rid for row in batch["rows"]]
                largest_history = (
                    max((len(set(map(int, values))) for values in histories), default=0)
                    if exclude_history_items else 0
                )
                search_k = min(int(index.ntotal), max_k + largest_history)
                retrieved, nonzero_queries = search_nonzero_queries(index, query, search_k)
                zero_user_vector_count += int(np.count_nonzero(~nonzero_queries))
                output_rows = []
                for row, history, faiss_rows, query_is_nonzero in zip(
                    batch["rows"], histories, retrieved, nonzero_queries
                ):
                    target_rid = int(row["target_item_rid"])
                    count = int(store.train_counts[target_rid]) if 0 < target_rid < store.train_counts.size else 0
                    group = classify_strength(count, p50, p90)
                    if not query_is_nonzero or group == "Unseen":
                        rank = None
                    else:
                        ranking = filter_history_from_faiss_rows(
                            faiss_rows, indexed_rids, history, max_k,
                            exclude_history_items=exclude_history_items,
                        )
                        rank = ranking.index(target_rid) + 1 if target_rid in ranking else None
                    ranks.append(rank)
                    groups.append(group)
                    output_rows.append(
                        {"sample_id": int(row["sample_id"]), "target_item_oid": int(row["target_item_oid"]),
                         "target_item_rid": target_rid, "target_strength_group": group, "target_rank": rank}
                    )
                writer.write_table(pa.Table.from_pylist(output_rows, schema=RANK_SCHEMA))
    finally:
        writer.close()
    return (
        metrics_from_ranks(ranks, groups, config["recall_ks"], config["ndcg_ks"]),
        {
            "sample_count": len(ranks),
            "zero_user_vector_count": zero_user_vector_count,
            "zero_user_vector_ratio": zero_user_vector_count / len(ranks) if ranks else 0.0,
        },
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, stage3_root, stage4_root, output_root, log_root = stage5_paths(config, args.debug)
    logger = configure_logging(log_root, "stage5_5_evaluate_two_tower")
    contracts = require_contracts(stage3_root, stage4_root, config)
    timer = Timer()
    root = output_root / "two_tower"
    repeated_audit_path = output_root / "audits" / "repeated_target_audit.json"
    hnsw_audit_path = output_root / "audits" / "hnsw_accuracy_audit.json"
    required = [
        root / "checkpoints" / "best.pt", root / "faiss.index",
        root / "indexed_candidates.parquet", root / "index_manifest.json",
        repeated_audit_path, hnsw_audit_path,
    ]
    require_paths(required)
    repeated_audit = load_json(repeated_audit_path)
    if repeated_audit.get("recall_protocol_version") != config["recall_protocol_version"]:
        raise ValueError("repeated-target audit protocol mismatch")
    if bool(repeated_audit["retrieval_policy"]["exclude_history_items"]) != bool(config["retrieval"]["exclude_history_items"]):
        raise ValueError("repeated-target audit retrieval policy mismatch")
    hnsw_audit = load_json(hnsw_audit_path)
    if hnsw_audit.get("recall_protocol_version") != config["recall_protocol_version"]:
        raise ValueError("HNSW accuracy audit protocol mismatch")
    if not bool(hnsw_audit.get("passed")):
        raise ValueError("HNSW accuracy audit did not pass configured retrieval-recall thresholds")
    metrics_path = root / "metrics.json"
    rank_paths = {split: root / "ranks" / f"{split}.parquet" for split in ("validation", "test")}
    guard_outputs([metrics_path, *rank_paths.values()], args.overwrite)
    store = Stage5SequenceStore(stage4_root)
    section = config["two_tower"]
    model = VanillaTwoTower(store.rid_to_token.size + 1, int(section["embedding_dim"]))
    checkpoint = load_model_checkpoint(required[0], model, map_location="cpu")
    if checkpoint["protocols"].get("recall") != config["recall_protocol_version"]:
        raise ValueError("checkpoint recall protocol mismatch")
    device = select_device(args.device)
    model.to(device).eval()
    index = faiss.read_index(str(root / "faiss.index"))
    index.hnsw.efSearch = int(config["faiss"]["efSearch"])
    indexed = ds.dataset(root / "indexed_candidates.parquet", format="parquet").to_table(columns=["faiss_row", "item_rid"])
    faiss_rows = np.asarray(indexed.column("faiss_row").to_numpy(), dtype=np.int64)
    if not np.array_equal(faiss_rows, np.arange(len(faiss_rows))):
        raise ValueError("indexed candidate physical rows are not dense FAISS rows")
    indexed_rids = np.asarray(indexed.column("item_rid").to_numpy(), dtype=np.int64)
    if int(index.ntotal) != len(indexed_rids):
        raise ValueError("FAISS/indexed candidate row count mismatch")
    max_samples = args.max_samples
    if args.debug and max_samples is None:
        max_samples = int(config["debug"]["max_eval_samples"])
    p50, p90 = float(contracts["stage4"]["p50_train"]), float(contracts["stage4"]["p90_train"])
    exclude_history_items = bool(config["retrieval"]["exclude_history_items"])
    metrics = {}
    zero_user_vectors = {}
    for split, filename in (("validation", "val_primary.parquet"), ("test", "test_primary.parquet")):
        logger.info("evaluating split=%s", split)
        metrics[split], zero_user_vectors[split] = evaluate_split(
            stage3_root / "samples" / filename, rank_paths[split], model, index, indexed_rids,
            store, config, device, max_samples, p50, p90, exclude_history_items,
        )
    save_json(
        {"stage": "5.5", "schema_version": 1, "recall_protocol_version": config["recall_protocol_version"],
         "debug": bool(args.debug), "unseen_policy": "rank is null and sample remains in denominator",
         "exclude_history_items": exclude_history_items,
         "repeated_target_audit": str(repeated_audit_path.relative_to(PROJECT_ROOT)),
         "hnsw_accuracy_audit": str(hnsw_audit_path.relative_to(PROJECT_ROOT)),
         "zero_user_vector_policy": "target_rank is null; sample remains in evaluation denominator; query is not sent to FAISS",
         "zero_user_vectors": zero_user_vectors,
         "metrics": metrics, "elapsed_seconds": timer.elapsed_seconds},
        metrics_path, args.overwrite,
    )
    logger.info("Two-Tower evaluation complete elapsed_seconds=%.2f", timer.elapsed_seconds)


if __name__ == "__main__":
    main()
