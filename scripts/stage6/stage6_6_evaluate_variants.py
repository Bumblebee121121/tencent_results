"""Stage 6.8: evaluate variants with frozen metrics and strength groups."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from src.recall.stage6_retrieval import evaluate_variant
from src.recall.stage6_runtime import add_common_arguments,guard_outputs,load_config,load_json,require_contracts,save_json,stage6_paths
def main():
 p=argparse.ArgumentParser(description=__doc__);add_common_arguments(p);p.add_argument("--variant",choices=["U1","U2","U3","I1","I2","I3","E1"]);p.add_argument("--device");a=p.parse_args();c=load_config(a.config);paths=stage6_paths(c,a.debug);require_contracts(paths,c);root=paths["output_root"]
 for v in ([a.variant] if a.variant else ["U1","U2","U3","I1","I2","I3","E1"]):
  labels=["best_loss"] if a.debug else ["best_loss","final"];candidate_results={}
  for label in labels:candidate_results[label]=evaluate_variant(v,c,paths,a.debug,a.overwrite,a.device,label,True)
  selected=max(labels,key=lambda label:float(candidate_results[label]["metrics"]["validation"]["Overall"]["Recall@100"]))
  selection_path=root/"manifests"/f"{v.lower()}_checkpoint_selection.json";guard_outputs([selection_path],a.overwrite)
  save_json({"variant":v,"selection_split":"validation","selection_metric":"Overall Recall@100","selected_checkpoint_label":selected,"candidate_recall100":{label:candidate_results[label]["metrics"]["validation"]["Overall"]["Recall@100"] for label in labels},"test_used_for_selection":False,"debug_single_candidate_only":a.debug},selection_path,a.overwrite)
  evaluate_variant(v,c,paths,a.debug,a.overwrite,a.device,selected,False)
  if v=="E1":
   selected_gate=load_json(root/"audits"/f"history_strength_gate_{selected}.json");save_json(selected_gate,root/"audits"/"history_strength_gate.json",a.overwrite)
if __name__=="__main__":main()
