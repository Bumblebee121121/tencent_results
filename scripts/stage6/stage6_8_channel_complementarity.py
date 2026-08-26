"""Stage 6.9: compare frozen E1 hits with Stage 5 best ItemCF (oracle union only)."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import pyarrow.dataset as ds
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from src.recall.evaluation import complementarity_from_ranks
from src.recall.stage6_runtime import add_common_arguments,guard_outputs,load_config,require_contracts,require_paths,save_json,stage6_paths
def rank_map(path):
 t=ds.dataset(path,format="parquet").to_table(columns=["sample_id","target_rank"]);return {int(i):(None if r is None else int(r)) for i,r in zip(t.column(0).to_pylist(),t.column(1).to_pylist())}
def main():
 p=argparse.ArgumentParser(description=__doc__);add_common_arguments(p);a=p.parse_args();c=load_config(a.config);paths=stage6_paths(c,a.debug);contracts=require_contracts(paths,c);root=paths["output_root"]
 best=contracts["stage5"].get("best_itemcf_selected_on_validation","click3_recent20");first_path=paths["stage5_root"]/"itemcf"/"ranks"/f"{best}_test.parquet";second_path=root/"predictions"/"E1_test.parquet";output=root/"reports"/"channel_complementarity.json";require_paths([first_path,second_path]);guard_outputs([output],a.overwrite)
 first,second=rank_map(first_path),rank_map(second_path)
 ids=sorted(set(first)&set(second)) if a.debug else sorted(first)
 if not a.debug and set(first)!=set(second):raise ValueError("I2I/E1 test sample IDs do not align")
 metrics=complementarity_from_ranks([first[i] for i in ids],[second[i] for i in ids],c["recall_ks"])
 save_json({"first_channel":f"itemcf:{best}","second_channel":"E1","evaluation_split":"test","name":"oracle_union_not_actual_fusion","metrics":metrics},output,a.overwrite)
if __name__=="__main__":main()

