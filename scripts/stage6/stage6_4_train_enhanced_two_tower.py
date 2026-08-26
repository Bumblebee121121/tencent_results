"""Stage 6.6-6.7: train E1 with the monotonic train-history strength gate."""

from __future__ import annotations
import argparse, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from src.recall.stage6_runtime import add_common_arguments, load_config, require_contracts, stage6_paths
from src.recall.stage6_workflow import train_variant
def main():
    parser=argparse.ArgumentParser(description=__doc__); add_common_arguments(parser); parser.add_argument("--device"); args=parser.parse_args()
    config=load_config(args.config); paths=stage6_paths(config,args.debug); require_contracts(paths,config); train_variant("E1",config,paths,args.debug,args.overwrite,args.device)
if __name__=="__main__": main()

