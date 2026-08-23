"""Stage 5.4: export train-seen candidate embeddings and build CPU FAISS HNSW-IP."""

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.item_strength import classify_strength
from src.models.vanilla_two_tower import VanillaTwoTower
from src.recall.checkpoint import load_checkpoint
from src.recall.data import Stage5SequenceStore
from src.recall.faiss_utils import build_hnsw_ip, train_seen_candidate_rows
from src.recall.runtime import (
    Timer, add_common_arguments, configure_logging, guard_outputs, load_config,
    require_contracts, require_paths, save_json, select_device, stage5_paths,
)


INDEXED_SCHEMA = pa.schema(
    [
        ("faiss_row", pa.int64()), ("item_oid", pa.int64()), ("item_rid", pa.int64()),
        ("retrieval_id", pa.int64()), ("model_item_token", pa.int64()),
        ("strength_group", pa.string()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _, stage3_root, stage4_root, output_root, log_root = stage5_paths(config, args.debug)
    logger = configure_logging(log_root, "stage5_4_build_faiss_index")
    contracts = require_contracts(stage3_root, stage4_root, config)
    timer = Timer()
    root = output_root / "two_tower"
    checkpoint_path = root / "checkpoints" / "best.pt"
    embeddings_path = root / "item_embeddings.npy"
    candidates_path = root / "indexed_candidates.parquet"
    index_path = root / "faiss.index"
    manifest_path = root / "index_manifest.json"
    require_paths([checkpoint_path])
    guard_outputs([embeddings_path, candidates_path, index_path, manifest_path], args.overwrite)

    store = Stage5SequenceStore(stage4_root)
    raw_rows = ds.dataset(stage3_root / "candidates" / "eval_candidates.parquet", format="parquet").to_table().to_pylist()
    rows = train_seen_candidate_rows(raw_rows, store.rid_to_token)
    p50, p90 = float(contracts["stage4"]["p50_train"]), float(contracts["stage4"]["p90_train"])
    for row in rows:
        row["strength_group"] = classify_strength(int(store.train_counts[int(row["item_rid"])]), p50, p90)
    pq.write_table(pa.Table.from_pylist(rows, schema=INDEXED_SCHEMA), candidates_path, compression="snappy")

    device = select_device(args.device)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    if checkpoint["protocols"].get("recall") != config["recall_protocol_version"]:
        raise ValueError("checkpoint recall protocol mismatch")
    section = checkpoint["config"]["two_tower"]
    model = VanillaTwoTower(store.rid_to_token.size + 1, int(section["embedding_dim"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    batch_size = int(config["faiss"]["embedding_batch_size"])
    embeddings = np.lib.format.open_memmap(
        embeddings_path, mode="w+", dtype=np.float32, shape=(len(rows), int(section["embedding_dim"]))
    )
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            stop = min(len(rows), start + batch_size)
            tokens = torch.tensor([int(row["model_item_token"]) for row in rows[start:stop]], device=device)
            embeddings[start:stop] = model.encode_item(tokens).cpu().numpy()
    embeddings.flush()
    values = np.load(embeddings_path, mmap_mode="r", allow_pickle=False)
    faiss_section = config["faiss"]
    index = build_hnsw_ip(values, int(faiss_section["M"]), int(faiss_section["efConstruction"]), int(faiss_section["efSearch"]))
    faiss.write_index(index, str(index_path))
    save_json(
        {
            "stage": "5.4", "schema_version": 1, "recall_protocol_version": config["recall_protocol_version"],
            "debug": bool(args.debug), "framework": "faiss-cpu", "index_type": "HNSWFlat-IP",
            "M": int(faiss_section["M"]), "efConstruction": int(faiss_section["efConstruction"]),
            "efSearch": int(faiss_section["efSearch"]), "embedding_dim": int(section["embedding_dim"]),
            "full_eval_candidate_count": len(raw_rows), "indexed_train_seen_candidate_count": len(rows),
            "excluded_train_unseen_candidate_count": len(raw_rows) - len(rows),
            "row_alignment": "faiss row equals physical row in indexed_candidates.parquet; retrieval_id is metadata only",
            "faiss_ntotal": int(index.ntotal), "elapsed_seconds": timer.elapsed_seconds,
        },
        manifest_path, args.overwrite,
    )
    logger.info("built FAISS ntotal=%d device_for_export=%s elapsed_seconds=%.2f", index.ntotal, device, timer.elapsed_seconds)


if __name__ == "__main__":
    main()
