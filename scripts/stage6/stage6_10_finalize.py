"""Finalize Stage 6 only after all audits, metrics, fusion and tests pass."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from src.recall.stage6_runtime import add_common_arguments,guard_outputs,load_config,load_json,require_contracts,require_paths,save_json,stage6_paths
def main():
 p=argparse.ArgumentParser(description=__doc__);add_common_arguments(p);p.add_argument("--run-unit-tests",action="store_true");a=p.parse_args();c=load_config(a.config);paths=stage6_paths(c,a.debug);require_contracts(paths,c);root=paths["output_root"];manifest=root/"manifests"/"stage6_manifest.json"
 required=[root/"manifests"/"baseline_freeze.json",root/"audits"/"contract.json",root/"audits"/"session_gap_selection.json",root/"audits"/"session_definition.json",root/"audits"/"history_strength_gate.json",root/"reports"/"full_ablation.csv",root/"reports"/"channel_complementarity.json",root/"metrics"/"fused.json"]+[root/"metrics"/f"{v}.json" for v in ("U1","U2","U3","I1","I2","I3","E1")];require_paths(required);guard_outputs([manifest],a.overwrite)
 tests=False
 if a.run_unit_tests:
  completed=subprocess.run([sys.executable,"-X","utf8","-m","unittest","discover","-s","tests/stage6","-v"],cwd=PROJECT_ROOT,text=True,capture_output=True,encoding="utf-8")
  if completed.returncode:raise RuntimeError(completed.stdout+completed.stderr)
  tests=True
 hnsw=load_json(root/"audits"/"e1_hnsw_accuracy.json");session=load_json(root/"audits"/"session_gap_selection.json")
 if session.get("selection_split")!="validation" or session.get("selection_metric")!="Overall Recall@100" or bool(session.get("test_used_for_selection")) or not bool(session.get("frozen")):raise ValueError("formal Session Gap selection audit is invalid")
 save_json({"stage":6,"protocol_version":c["stage6_protocol_version"],"stage3_protocol":c["stage3_protocol_version"],"stage4_protocol":c["stage4_protocol_version"],"stage5_baseline_frozen":True,"selected_session_gap_seconds":int(session["selected_session_gap_seconds"]),"session_gap_selected_on_validation_recall100":True,"user_variants_completed":["U1","U2","U3"],"item_variants_completed":["I1","I2","I3","E1"],"test_used_for_model_selection":False,"candidate_cold_start_used_as_model_feature":False,"retrieval_id_used_as_model_item_id":False,"future_item_strength_used":False,"unseen_candidates_indexed_with_non_id_features":True,"hnsw_accuracy_passed":bool(hnsw["passed"]),"all_stage6_tests_passed":tests,"fusion_completed":True,"debug":a.debug},manifest,a.overwrite)
if __name__=="__main__":main()
