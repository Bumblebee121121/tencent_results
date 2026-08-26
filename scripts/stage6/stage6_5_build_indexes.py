"""Stage 6.5: export candidate vectors and build HNSW-IP indexes, including Unseen."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from src.recall.stage6_retrieval import build_variant_index
from src.recall.stage6_runtime import add_common_arguments,load_config,require_contracts,stage6_paths
def main():
 p=argparse.ArgumentParser(description=__doc__);add_common_arguments(p);p.add_argument("--variant",choices=["U1","U2","U3","I1","I2","I3","E1"]);p.add_argument("--checkpoint",choices=["best_loss","final"]);p.add_argument("--device");a=p.parse_args();c=load_config(a.config);paths=stage6_paths(c,a.debug);require_contracts(paths,c)
 labels=[a.checkpoint] if a.checkpoint else (["best_loss"] if a.debug else ["best_loss","final"])
 for v in ([a.variant] if a.variant else ["U1","U2","U3","I1","I2","I3","E1"]):
  for label in labels:build_variant_index(v,c,paths,a.debug,a.overwrite,a.device,label)
if __name__=="__main__":main()
