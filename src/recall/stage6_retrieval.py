"""Candidate embedding export, FAISS retrieval and rank materialization."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader

from src.data.item_strength import classify_strength
from src.recall.evaluation import metrics_from_ranks
from src.features.feature_store import FeatureStore
from src.recall.faiss_utils import build_hnsw_ip, hnsw_retrieval_recall, search_nonzero_queries

from .stage6_data import Stage6Collator, Stage6ItemStore, Stage6ParquetDataset, TimeNormalization, move_tensor_tree
from .stage6_index import audit_item_embeddings
from .stage6_runtime import configured_session, guard_outputs, load_json, save_json, select_device
from .stage6_training import build_model, load_model_weights


INDEX_SCHEMA = pa.schema([
    ("faiss_row", pa.int64()), ("item_oid", pa.int64()), ("item_rid", pa.int64()),
    ("retrieval_id", pa.int64()), ("model_item_token", pa.int64()), ("strength_group", pa.string()),
])
RANK_SCHEMA = pa.schema([
    ("sample_id", pa.int64()), ("target_item_oid", pa.int64()), ("target_item_rid", pa.int64()),
    ("target_strength_group", pa.string()), ("target_rank", pa.int64()),
    ("top_item_oids", pa.list_(pa.int64())),
])


def _candidate_batches(paths, item_store, config, batch_size, include_unseen):
    fields = list(item_store.side_fields)
    columns = ["retrieval_id", "item_oid", "item_rid", "model_item_token", "item_train_count", "strength_group"] + [f"f{field}_token" for field in fields]
    dataset = ds.dataset(paths["stage4_root"] / "feature_store" / "eval_candidate_side.parquet", format="parquet")
    mm = np.load(paths["stage4_root"] / "feature_store" / "eval_candidate_mm.npy", mmap_mode="r", allow_pickle=False)
    mm_valid = np.load(paths["stage4_root"] / "feature_store" / "eval_candidate_mm_valid.npy", mmap_mode="r", allow_pickle=False)
    physical_offset = 0
    logical_offset = 0
    for batch in dataset.scanner(columns=columns, batch_size=batch_size).to_batches():
        physical_stop = physical_offset + batch.num_rows
        raw_rows = batch.to_pylist()
        selected = [index for index, row in enumerate(raw_rows) if include_unseen or int(row["item_train_count"]) > 0]
        rows = [raw_rows[index] for index in selected]
        if not rows:
            physical_offset = physical_stop
            continue
        item_batch = {
            "item_tokens": torch.tensor([int(row["model_item_token"]) for row in rows]),
            "side_tokens": torch.tensor([[int(row[f"f{field}_token"]) for field in fields] for row in rows]),
            "mm": torch.from_numpy(np.asarray(mm[physical_offset:physical_stop][selected], dtype=np.float32)),
            "mm_valid": torch.from_numpy(np.asarray(mm_valid[physical_offset:physical_stop][selected], dtype=np.bool_)),
            "train_counts": torch.tensor([float(row["item_train_count"]) for row in rows]),
        }
        yield logical_offset, rows, item_batch
        logical_offset += len(rows)
        physical_offset = physical_stop
    if physical_offset != len(mm):
        raise ValueError("candidate Side/MM physical row alignment mismatch")


def build_variant_index(
    variant, config, paths, debug, overwrite, device_name=None,
    checkpoint_label="best_loss", artifact_name=None,
):
    started = time.perf_counter(); root = paths["output_root"]
    owner = artifact_name or variant
    variant_root = root / "indexes" / owner / checkpoint_label
    embedding_path = variant_root / "item_embeddings.npy"; candidate_path = variant_root / "indexed_candidates.parquet"
    index_path = variant_root / "faiss.index"; manifest_path = variant_root / "index_manifest.json"
    gate_path = root / "audits" / f"history_strength_gate_{checkpoint_label}.json" if variant == "E1" else None
    outputs = [embedding_path, candidate_path, index_path, manifest_path] + ([gate_path] if gate_path else [])
    guard_outputs(outputs, overwrite)
    item_store = Stage6ItemStore(paths["stage4_root"], config["item_tower"]["side_fields"])
    model = build_model(variant, config, int(item_store.rid_to_token.size + 1), item_store.side_vocab_sizes)
    checkpoint_path = root / "checkpoints" / owner / f"{checkpoint_label}.pt"
    load_model_weights(checkpoint_path, model)
    device = select_device(device_name); model.to(device).eval()
    include_unseen = variant in {"I1", "I2", "I3", "E1"}
    full_count = int(load_json(paths["stage3_root"] / "candidates" / "eval_candidate_manifest.json")["final_candidate_count"])
    if include_unseen:
        count = full_count
    else:
        side_dataset = ds.dataset(paths["stage4_root"] / "feature_store" / "eval_candidate_side.parquet", format="parquet")
        count = int(side_dataset.count_rows(filter=ds.field("item_train_count") > 0))
    dimension = int(config["model"]["embedding_dim"])
    embeddings = np.lib.format.open_memmap(embedding_path, mode="w+", dtype=np.float32, shape=(count, dimension))
    writer = pq.ParquetWriter(candidate_path, INDEX_SCHEMA, compression="snappy")
    groups=[]; gates_by_group={name: [] for name in ("Head","Mid","Tail","Unseen")}
    try:
        with torch.no_grad():
            for offset, rows, item_batch in _candidate_batches(paths, item_store, config, int(config["faiss"]["embedding_batch_size"]), include_unseen):
                encoded = model.encode_item(move_tensor_tree(item_batch, device), return_gate=variant == "E1")
                if variant == "E1":
                    encoded, gates = encoded
                stop = offset + len(rows); embeddings[offset:stop] = encoded.cpu().numpy()
                output=[]
                for index, row in enumerate(rows):
                    group=str(row["strength_group"]); groups.append(group)
                    output.append({"faiss_row": offset+index, "item_oid": int(row["item_oid"]),
                                   "item_rid": row["item_rid"], "retrieval_id": int(row["retrieval_id"]),
                                   "model_item_token": int(row["model_item_token"]), "strength_group": group})
                    if variant == "E1": gates_by_group[group].append(float(gates[index].cpu()))
                writer.write_table(pa.Table.from_pylist(output, schema=INDEX_SCHEMA))
    finally:
        writer.close(); embeddings.flush()
    values = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
    audit = audit_item_embeddings(values, groups)
    if audit["indexed_candidate_count"] != count or audit["nan_inf_count"] or audit["zero_vector_count"]:
        raise ValueError(f"invalid enhanced candidate vectors: {audit}")
    section=config["faiss"]; index=build_hnsw_ip(values,int(section["M"]),int(section["efConstruction"]),int(section["efSearch"])); faiss.write_index(index,str(index_path))
    if variant == "E1":
        gate_audit={}
        for group, values_ in gates_by_group.items():
            array=np.asarray(values_,dtype=np.float64)
            gate_audit[group]={"count":int(array.size),"mean":float(array.mean()) if array.size else 0.0,
                               **{name:float(np.quantile(array,q)) if array.size else 0.0 for name,q in (("p10",.1),("p50",.5),("p90",.9))}}
        gate_audit["unseen_gate_exactly_zero"] = all(value == 0.0 for value in gates_by_group["Unseen"])
        if not gate_audit["unseen_gate_exactly_zero"]: raise ValueError("E1 unseen ID gate is not zero")
        save_json(gate_audit,gate_path,overwrite)
    manifest={"stage":"6.index","variant":variant,"artifact_name":owner,
              "checkpoint_label":checkpoint_label,"protocol_version":config["stage6_protocol_version"],
              "debug":debug,"all_eval_candidates_included":include_unseen,
              "full_eval_candidate_count":full_count,"excluded_train_unseen_candidate_count":full_count-count,
              "row_alignment":"dense FAISS row follows filtered physical candidate order; retrieval_id metadata only",
              **audit,"faiss_ntotal":int(index.ntotal),"elapsed_seconds":round(time.perf_counter()-started,3)}
    save_json(manifest,manifest_path,overwrite); return manifest


def evaluate_variant(
    variant, config, paths, debug, overwrite, device_name=None,
    checkpoint_label="best_loss", selection_candidate=False,
    artifact_name=None, session_gap_override=None, include_test=True,
):
    root=paths["output_root"]; owner=artifact_name or variant
    variant_root=root/"indexes"/owner/checkpoint_label
    if selection_candidate:
        metrics_path=root/"metrics"/"checkpoint_candidates"/f"{owner}_{checkpoint_label}.json"
        rank_paths={"validation":root/"predictions"/"checkpoint_candidates"/f"{owner}_{checkpoint_label}_validation.parquet"}
        audit_path=root/"audits"/f"{owner.lower()}_{checkpoint_label}_hnsw_accuracy.json"
        split_files=(("validation","val_primary.parquet"),)
    else:
        metrics_path=root/"metrics"/f"{variant}.json"
        selected_splits=("validation","test") if include_test else ("validation",)
        rank_paths={split:root/"predictions"/f"{variant}_{split}.parquet" for split in selected_splits}
        audit_path=root/"audits"/f"{variant.lower()}_hnsw_accuracy.json"
        split_files=tuple(
            (split, "val_primary.parquet" if split == "validation" else "test_primary.parquet")
            for split in selected_splits
        )
    guard_outputs([metrics_path,*rank_paths.values(),audit_path],overwrite)
    store=FeatureStore(paths["stage4_root"]); item_store=Stage6ItemStore(paths["stage4_root"],config["item_tower"]["side_fields"])
    model=build_model(variant,config,int(store.rid_to_token.size+1),item_store.side_vocab_sizes)
    load_model_weights(root/"checkpoints"/owner/f"{checkpoint_label}.pt",model); device=select_device(device_name); model.to(device).eval()
    index=faiss.read_index(str(variant_root/"faiss.index")); index.hnsw.efSearch=int(config["faiss"]["efSearch"])
    indexed=ds.dataset(variant_root/"indexed_candidates.parquet",format="parquet").to_table(columns=["faiss_row","item_oid","item_rid"])
    rows=np.asarray(indexed.column("faiss_row").to_numpy(),dtype=np.int64)
    if not np.array_equal(rows,np.arange(len(rows))) or int(index.ntotal)!=len(rows): raise ValueError("FAISS candidate row misalignment")
    indexed_oids=np.asarray(indexed.column("item_oid").to_numpy(),dtype=np.int64)
    time_stats=TimeNormalization.from_json(load_json(root/"audits"/"time_normalization.json")); gap,short_max,long_max=configured_session(config,debug,root,session_gap_override)
    maximum=int(config["debug"]["max_eval_samples"]) if debug else None; max_k=max(map(int,config["recall_ks"])); all_metrics={}; audit_queries=None
    for split,filename in split_files:
        dataset=Stage6ParquetDataset(paths["stage3_root"]/"samples"/filename,store,maximum,int(config["scan_batch_size"]))
        collator=Stage6Collator(gap,item_store,time_stats,None,short_max,long_max,False)
        loader=DataLoader(dataset,batch_size=int(config["training"]["batch_size"]),collate_fn=collator,num_workers=0)
        ranks=[]; groups=[]; writer=pq.ParquetWriter(rank_paths[split],RANK_SCHEMA,compression="snappy"); query_chunks=[]
        try:
            with torch.no_grad():
                for batch in loader:
                    if not batch.get("rows"): continue
                    queries=model.encode_user(move_tensor_tree(batch["user"],device)).cpu().numpy()
                    retrieved,nonzero=search_nonzero_queries(index,queries,max_k)
                    output=[]
                    for sample,row_ids,valid in zip(batch["rows"],retrieved,nonzero):
                        target_oid=int(sample["target_item_oid"]); group=str(sample["target_strength_group"])
                        ranking=[int(indexed_oids[int(value)]) for value in row_ids if int(value)>=0] if valid else []
                        rank=ranking.index(target_oid)+1 if target_oid in ranking else None
                        ranks.append(rank); groups.append(group)
                        output.append({"sample_id":int(sample["sample_id"]),"target_item_oid":target_oid,
                                       "target_item_rid":sample.get("target_item_rid"),"target_strength_group":group,
                                       "target_rank":rank,"top_item_oids":ranking})
                    writer.write_table(pa.Table.from_pylist(output,schema=RANK_SCHEMA))
                    if split=="validation" and sum(len(x) for x in query_chunks)<int(config["hnsw_accuracy_audit"]["sample_size"]): query_chunks.append(queries[np.linalg.norm(queries,axis=1)>0])
        finally: writer.close()
        all_metrics[split]=metrics_from_ranks(ranks,groups,config["recall_ks"],config["ndcg_ks"])
        if split=="validation": audit_queries=np.concatenate(query_chunks)[:int(config["debug"]["hnsw_audit_samples"] if debug else config["hnsw_accuracy_audit"]["sample_size"])]
    embeddings=np.load(variant_root/"item_embeddings.npy",mmap_mode="r",allow_pickle=False); exact=faiss.IndexFlatIP(embeddings.shape[1]); exact.add(np.ascontiguousarray(embeddings,dtype=np.float32))
    width=max(map(int,config["hnsw_accuracy_audit"]["ks"])); _,approx_rows=index.search(np.ascontiguousarray(audit_queries,dtype=np.float32),width); _,exact_rows=exact.search(np.ascontiguousarray(audit_queries,dtype=np.float32),width)
    audit_metrics=hnsw_retrieval_recall(approx_rows,exact_rows,config["hnsw_accuracy_audit"]["ks"])
    thresholds=config["hnsw_accuracy_audit"]["minimum_mean_recall"]; passed=all(audit_metrics[f"@{int(k)}"]["mean_recall"]>=float(thresholds[int(k)] if int(k) in thresholds else thresholds[str(k)]) for k in config["hnsw_accuracy_audit"]["ks"])
    save_json({"variant":variant,"validation_queries_only":True,"passed":passed,"metrics":audit_metrics},audit_path,overwrite)
    if not passed: raise ValueError("HNSW accuracy audit failed")
    result={"stage":"6.evaluate","variant":variant,"artifact_name":owner,
            "session_gap_seconds":gap,"checkpoint_label":checkpoint_label,
            "selection_candidate_validation_only":selection_candidate,"protocol_version":config["stage6_protocol_version"],"debug":debug,
            "test_evaluated":bool("test" in all_metrics),
            "unseen_targets_remain_in_denominator":True,"hnsw_accuracy_passed":True,"metrics":all_metrics}
    save_json(result,metrics_path,overwrite); return result
