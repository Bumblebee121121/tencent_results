"""Stage 6.7: materialize the preregistered B0/U/I/E ablation tables."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(PROJECT_ROOT))
from src.recall.stage6_evaluation import ablation_rows,experiment_markdown
from src.recall.stage6_runtime import add_common_arguments,guard_outputs,load_config,load_json,require_contracts,require_paths,save_csv,stage6_paths

TEXT={
"U1":("Stage 5 用户历史只做平均，无法区分长期与当前兴趣。","SDM 风格长短期编码能更准确表达当前兴趣。","只将 mean pooling 改为长短期编码，物品塔保持 Pure-ID。"),
"U2":("U1 不区分曝光与点击。","显式行为语义能提供额外信息。","只在 U1 事件表示中加入 Action embedding。"),
"U3":("序列相邻不代表真实时间相近。","连续真实时间特征能提供额外收益。","只在 U2 中加入 Train-only 标准化的 recency/gap/first 特征。"),
"I1":("纯 ID 对弱历史广告表示不足。","结构化 Side Feature 能补足物品表示。","固定 U3，只加入 13 个 Side 字段。"),
"I2":("纯 ID 对弱历史广告表示不足。","32 维 MM 能补足物品表示。","固定 U3，只加入带 valid mask 的 MM。"),
"I3":("Side 与 MM 可能提供互补信息。","两类非 ID 信息组合优于单独使用。","固定 U3，组合 Side 与 MM，仍使用固定融合。"),
"E1":("固定融合不能随广告历史强弱调整 ID 信任度。","单调历史强度 gate 能更合理地融合 ID/非 ID。","只将 I3 固定融合替换为 Train-only history-strength gate。"),
}
def main():
 p=argparse.ArgumentParser(description=__doc__);add_common_arguments(p);a=p.parse_args();c=load_config(a.config);paths=stage6_paths(c,a.debug);contracts=require_contracts(paths,c);root=paths["output_root"]
 files=[root/"metrics"/f"{v}.json" for v in ("U1","U2","U3","I1","I2","I3","E1")];require_paths(files)
 outputs=[root/"reports"/"full_ablation.csv",root/"reports"/"user_tower_ablation.csv",root/"reports"/"item_tower_ablation.csv"]+[root/"reports"/f"{v}_experiment.md" for v in TEXT];guard_outputs(outputs,a.overwrite)
 metrics={"B0":contracts["two_tower"]["metrics"]};metrics.update({v:load_json(path)["metrics"] for v,path in zip(("U1","U2","U3","I1","I2","I3","E1"),files)})
 rows=ablation_rows(metrics,c["recall_ks"]);fields=["variant","split","group","K","Recall","NDCG","delta_vs_previous"]
 save_csv(rows,fields,outputs[0],a.overwrite);save_csv([r for r in rows if r["variant"] in {"B0","U1","U2","U3"}],fields,outputs[1],a.overwrite);save_csv([r for r in rows if r["variant"] in {"U3","I1","I2","I3","E1"}],fields,outputs[2],a.overwrite)
 for v,(problem,hypothesis,solution) in TEXT.items():
  summary=[r for r in rows if r["variant"]==v and r["split"]=="validation" and r["K"]==100]
  result="\n".join(f"- {r['group']}: Recall@100={r['Recall']:.8f}, delta={r['delta_vs_previous']}" for r in summary)
  outputs[list(TEXT).index(v)+3].write_text(experiment_markdown(v,problem,hypothesis,solution,result),encoding="utf-8",newline="\n")
if __name__=="__main__":main()

