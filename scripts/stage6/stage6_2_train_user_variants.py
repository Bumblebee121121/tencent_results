"""Stage 6.2-6.4: train U1, U2 and U3 in attribution order."""

from __future__ import annotations
import argparse, sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))
from src.recall.stage6_runtime import add_common_arguments, load_config, require_contracts, stage6_paths
from src.recall.stage6_workflow import train_variant
def main():
    parser=argparse.ArgumentParser(description=__doc__); add_common_arguments(parser); parser.add_argument("--variant",choices=["U1","U2","U3"]); parser.add_argument("--device"); args=parser.parse_args()
    config=load_config(args.config); paths=stage6_paths(config,args.debug); require_contracts(paths,config)
    for variant in ([args.variant] if args.variant else ["U1","U2","U3"]): train_variant(variant,config,paths,args.debug,args.overwrite,args.device)
if __name__=="__main__": main()

