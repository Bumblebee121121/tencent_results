"""Stage 6.10: Validation-selected Reciprocal Rank Fusion of ItemCF and E1."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from src.data.item_strength import classify_strength
from src.recall.data import SAMPLE_COLUMNS,Stage5SequenceStore
from src.recall.evaluation import metrics_from_ranks
from src.recall.fusion import reciprocal_rank_fusion,select_rrf_weight
from src.recall.itemcf import NeighborShardLookup,retrieve_itemcf
from src.recall.stage6_runtime import add_common_arguments,guard_outputs,load_config,require_contracts,require_paths,save_csv,save_json,stage6_paths
SCHEMA=pa.schema([("sample_id",pa.int64()),("target_item_oid",pa.int64()),("target_strength_group",pa.string()),("target_rank",pa.int64()),("top_item_oids",pa.list_(pa.int64()))])
def enhanced_map(path):
 t=ds.dataset(path,format="parquet").to_table(columns=["sample_id","target_item_oid","target_strength_group","top_item_oids"]);return {int(r["sample_id"]):r for r in t.to_pylist()}
def channel_examples(split,path,enhanced,store,lookup,experiment,rid_to_oid,maximum):
 result=[];scanner=ds.dataset(path,format="parquet").scanner(columns=SAMPLE_COLUMNS,batch_size=8192);seen=0
 for batch in scanner.to_batches():
  for row in batch.to_pylist():
   if maximum is not None and seen>=maximum:return result
   sid=int(row["sample_id"])
   if sid not in enhanced:continue
   history=store.history(row);i2i_rids=retrieve_itemcf(history.item_rid,history.action_token,lookup,max(map(int,experiment["recall_ks"])),float(experiment["exposure_weight"]),float(experiment["click_weight"]),float(experiment["unknown_weight"]),experiment["history_limit"],bool(experiment["exclude_history_items"]))
   result.append({"sample_id":sid,"target":int(row["target_item_oid"]),"group":enhanced[sid]["target_strength_group"],"first":[rid_to_oid[r] for r in i2i_rids if r in rid_to_oid],"second":[int(v) for v in enhanced[sid]["top_item_oids"]]});seen+=1
 return result
def main():
 p=argparse.ArgumentParser(description=__doc__);add_common_arguments(p);a=p.parse_args();c=load_config(a.config);paths=stage6_paths(c,a.debug);contracts=require_contracts(paths,c);root=paths["output_root"]
 outputs=[root/"metrics"/"fused.json",root/"reports"/"fusion_validation.csv",root/"manifests"/"fusion_selection.json",root/"predictions"/"fused_test.parquet"];guard_outputs(outputs,a.overwrite)
 best=contracts["stage5"].get("best_itemcf_selected_on_validation","click3_recent20");stage5_config=load_config(PROJECT_ROOT/"configs"/"stage5.yaml");exp=dict(stage5_config["itemcf"]["experiments"][best]);exp["exclude_history_items"]=stage5_config["retrieval"]["exclude_history_items"];exp["recall_ks"]=c["recall_ks"]
 store=Stage5SequenceStore(paths["stage4_root"]);lookup=NeighborShardLookup(paths["stage5_root"]/"itemcf"/"neighbor_shards","click3",int(stage5_config["itemcf"]["pair_partitions"]));table=ds.dataset(paths["stage3_root"]/"candidates"/"eval_candidates.parquet",format="parquet").to_table(columns=["item_rid","item_oid"]);rid_to_oid={int(r):int(o) for r,o in zip(table.column(0).to_pylist(),table.column(1).to_pylist()) if r is not None}
 maximum=int(c["debug"]["max_eval_samples"]) if a.debug else None
 examples={s:channel_examples(s,paths["stage3_root"]/"samples"/("val_primary.parquet" if s=="validation" else "test_primary.parquet"),enhanced_map(root/"predictions"/f"E1_{s}.parquet"),store,lookup,exp,rid_to_oid,maximum) for s in ("validation","test")}
 alpha,search=select_rrf_weight(examples["validation"],c["fusion"]["validation_weights"],float(c["fusion"]["rrf_c"]),100);save_csv(search,["alpha","Recall"],outputs[1],a.overwrite);save_json({"selection_split":"validation","test_used_for_weight_selection":False,"alpha":alpha,"rrf_c":c["fusion"]["rrf_c"]},outputs[2],a.overwrite)
 all_metrics={};writer=pq.ParquetWriter(outputs[3],SCHEMA,compression="snappy")
 try:
  for split,items in examples.items():
   ranks=[];groups=[];rows=[]
   for ex in items:
    ranking=reciprocal_rank_fusion(ex["first"],ex["second"],alpha,float(c["fusion"]["rrf_c"]),max(map(int,c["recall_ks"])));rank=ranking.index(ex["target"])+1 if ex["target"] in ranking else None;ranks.append(rank);groups.append(ex["group"])
    if split=="test":rows.append({"sample_id":ex["sample_id"],"target_item_oid":ex["target"],"target_strength_group":ex["group"],"target_rank":rank,"top_item_oids":ranking})
   all_metrics[split]=metrics_from_ranks(ranks,groups,c["recall_ks"],c["ndcg_ks"])
   if rows:writer.write_table(pa.Table.from_pylist(rows,schema=SCHEMA))
 finally:writer.close()
 save_json({"stage":"6.fusion","method":"RRF","alpha":alpha,"rrf_c":c["fusion"]["rrf_c"],"metrics":all_metrics},outputs[0],a.overwrite)
if __name__=="__main__":main()

