"""Stage 6 attribution and experiment-report helpers."""

from __future__ import annotations

from typing import Mapping, Sequence


PREVIOUS_VARIANT = {"U1": "B0", "U2": "U1", "U3": "U2", "I1": "U3", "I2": "U3", "E1": "I3"}


def ablation_rows(metrics: Mapping[str, Mapping[str, object]], ks: Sequence[int]) -> list[dict[str, object]]:
    rows = []
    for variant, splits in metrics.items():
        for split, groups in splits.items():
            for group, values in groups.items():
                for k in ks:
                    previous = PREVIOUS_VARIANT.get(variant)
                    previous_values = metrics.get(previous, {}).get(split, {}).get(group, {}) if previous else {}
                    recall = float(values[f"Recall@{k}"])
                    if variant == "I3":
                        candidates = [metrics.get(name, {}).get(split, {}).get(group, {}) for name in ("I1", "I2")]
                        available = [float(item[f"Recall@{k}"]) for item in candidates if f"Recall@{k}" in item]
                        previous_recall = max(available) if available else None
                    else:
                        previous_recall = float(previous_values[f"Recall@{k}"]) if f"Recall@{k}" in previous_values else None
                    rows.append({
                        "variant": variant, "split": split, "group": group, "K": int(k),
                        "Recall": recall, "NDCG": values.get(f"NDCG@{k}"),
                        "delta_vs_previous": None if previous_recall is None else recall - previous_recall,
                    })
    return rows


def experiment_markdown(variant: str, problem: str, hypothesis: str, solution: str, result: str) -> str:
    return f"# {variant} 实验说明\n\n## 问题\n\n{problem}\n\n## 假设\n\n{hypothesis}\n\n## 方案\n\n{solution}\n\n## 实验与归因\n\n{result}\n"
