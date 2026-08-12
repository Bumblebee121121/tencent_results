"""
所属阶段：
阶段 2.4：候选池、历史未见广告与 cold_start

当前问题：
约 22.57% candidate 无法映射历史 RID，但其 cold_start、侧信息和多模态可用性尚不清楚。

本步目的：
精确对齐 OID，分析 seen/unseen 与字段级 cold_start、侧信息和有效多模态之间的关系。

为什么现在做：
历史长尾结构已经由 2.3 定义，现在可以独立刻画候选池中的历史未见广告。

输入：
candidate、indexer.pkl、32D mm_emb。

输出：
JSON 指标、CSV 四格表与字段级汇总、PNG 图、终端日志。

完成标准：
回答 unseen candidate 与 cold_start 是否一致，以及 unseen 中多少仍有侧信息和有效多模态。

对后续模型的潜在影响：
为 ID-only 与 side/mm 泛化表示的 seen/unseen 分组实验提出假设。

面试时如何讲：
我先区分历史 ID 可见性和数据提供的 cold_start 字段，再分析可用侧信息，不把未见广告等同于数据错误。
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds

from common import (
    CANDIDATE_DIR,
    FIGURES_DIR,
    INDEXER_PATH,
    METRICS_DIR,
    MM_DIR,
    TABLES_DIR,
    TABLE_BATCH_SIZE,
    ensure_output_dirs,
    load_item_mapping,
    lookup_sorted_flags,
    membership_mask,
    plt,
    print_outputs,
    require_columns,
    require_paths,
    save_csv,
    save_figure,
    save_json,
    scan_mm_oid_validity,
    sorted_mapping_oids,
    to_numpy_int,
)


STAGE = "阶段 2.4"
ID_COLUMNS = {"item_id", "retrieval_id"}
NULL_SENTINEL = np.iinfo(np.int64).min


def candidate_feature_names(dataset: ds.Dataset) -> list[str]:
    names = [name for name in dataset.schema.names if name not in ID_COLUMNS]
    for name in names:
        field = dataset.schema.field(name)
        if not pa.types.is_struct(field.type):
            raise ValueError(f"candidate 字段 {name} 不是 struct：{field.type}")
        children = {child.name for child in field.type}
        if not {"cold_start", "feature_value"}.issubset(children):
            raise ValueError(f"candidate 字段 {name} 缺少 cold_start/feature_value")
    return names


def main() -> None:
    print("阶段 2.4：候选池、历史未见广告与 cold_start")
    print(f"candidate 路径：{CANDIDATE_DIR.resolve()}")
    require_paths([CANDIDATE_DIR, INDEXER_PATH, MM_DIR])
    ensure_output_dirs()

    mapping = load_item_mapping()
    historical_oids = sorted_mapping_oids(mapping)
    mm_oids, mm_valid_flags, mm_audit = scan_mm_oid_validity()

    dataset = ds.dataset(CANDIDATE_DIR, format="parquet")
    require_columns(dataset, ["item_id", "retrieval_id"])
    feature_names = candidate_feature_names(dataset)
    print(f"candidate 匿名 struct 字段数：{len(feature_names)}")

    cold_counters: dict[tuple[str, str], Counter] = defaultdict(Counter)
    availability_counts: dict[tuple[str, str], int] = defaultdict(int)
    group_counts = Counter()
    mm_matrix = Counter()
    available_count_parts: list[np.ndarray] = []
    seen_parts: list[np.ndarray] = []
    row_count = 0

    scanner = dataset.scanner(
        columns=["item_id", *feature_names],
        batch_size=TABLE_BATCH_SIZE,
    )
    for batch_number, batch in enumerate(scanner.to_batches(), start=1):
        oids = to_numpy_int(batch.column(0))
        seen = membership_mask(oids, historical_oids)
        _, valid_mm = lookup_sorted_flags(oids, mm_oids, mm_valid_flags)
        available_per_candidate = np.zeros(batch.num_rows, dtype=np.int16)

        group_counts["seen"] += int(np.sum(seen))
        group_counts["unseen"] += int(np.sum(~seen))
        for group_name, group_mask in (("seen", seen), ("unseen", ~seen)):
            mm_matrix[(group_name, "valid_mm")] += int(np.sum(group_mask & valid_mm))
            mm_matrix[(group_name, "missing_mm")] += int(np.sum(group_mask & ~valid_mm))

        for column_index, feature_name in enumerate(feature_names, start=1):
            struct_array = batch.column(column_index)
            parent_null = struct_array.is_null().to_numpy(zero_copy_only=False)
            cold_array = struct_array.field("cold_start")
            feature_array = struct_array.field("feature_value")
            cold_values = to_numpy_int(cold_array,fill_value=NULL_SENTINEL,).copy()
            cold_values[parent_null] = NULL_SENTINEL
            available = ~parent_null & ~feature_array.is_null().to_numpy(zero_copy_only=False)
            available_per_candidate += available.astype(np.int16)

            for group_name, group_mask in (("seen", seen), ("unseen", ~seen)):
                availability_counts[(feature_name, group_name)] += int(np.sum(available & group_mask))
                selected = cold_values[group_mask]
                values, counts = np.unique(selected, return_counts=True)
                for value, count in zip(values, counts):
                    key = "null" if int(value) == NULL_SENTINEL else str(int(value))
                    cold_counters[(feature_name, group_name)][key] += int(count)

        available_count_parts.append(available_per_candidate)
        seen_parts.append(seen)
        row_count += batch.num_rows
        if batch_number % 5 == 0:
            print(f"已处理 {batch_number} 个 candidate 批次，累计候选数：{row_count:,}")

    available_counts_all = np.concatenate(available_count_parts)
    seen_all = np.concatenate(seen_parts)
    if group_counts["seen"] + group_counts["unseen"] != row_count:
        raise RuntimeError("candidate seen/unseen 计数未闭合")

    cold_rows = []
    for feature_name in feature_names:
        for group_name in ("seen", "unseen"):
            denominator = group_counts[group_name]
            for value, count in sorted(cold_counters[(feature_name, group_name)].items()):
                cold_rows.append(
                    {
                        "feature": feature_name,
                        "history_group": group_name,
                        "cold_start_value": value,
                        "candidate_count": count,
                        "candidate_ratio_within_group": count / denominator if denominator else None,
                    }
                )

    availability_rows = []
    for feature_name in feature_names:
        for group_name in ("seen", "unseen"):
            count = availability_counts[(feature_name, group_name)]
            denominator = group_counts[group_name]
            availability_rows.append(
                {
                    "feature": feature_name,
                    "history_group": group_name,
                    "non_null_feature_value_count": count,
                    "candidate_count": denominator,
                    "non_null_feature_value_ratio": count / denominator if denominator else None,
                }
            )

    available_distribution_rows = []
    for group_name, mask in (("seen", seen_all), ("unseen", ~seen_all)):
        values, counts = np.unique(available_counts_all[mask], return_counts=True)
        denominator = int(np.sum(mask))
        for value, count in zip(values, counts):
            available_distribution_rows.append(
                {
                    "history_group": group_name,
                    "available_feature_count": int(value),
                    "candidate_count": int(count),
                    "candidate_ratio_within_group": int(count) / denominator if denominator else None,
                }
            )

    matrix_rows = []
    for group_name in ("seen", "unseen"):
        for mm_group in ("valid_mm", "missing_mm"):
            count = mm_matrix[(group_name, mm_group)]
            denominator = group_counts[group_name]
            matrix_rows.append(
                {
                    "history_group": group_name,
                    "mm_group": mm_group,
                    "candidate_count": count,
                    "candidate_ratio_within_history_group": count / denominator if denominator else None,
                }
            )

    metrics_path = METRICS_DIR / "stage2_4_candidate_coldstart.json"
    matrix_path = TABLES_DIR / "stage2_4_candidate_seen_mm_matrix.csv"
    cold_path = TABLES_DIR / "stage2_4_cold_start_by_field_seen.csv"
    availability_path = TABLES_DIR / "stage2_4_feature_availability_by_field_seen.csv"
    distribution_path = TABLES_DIR / "stage2_4_available_feature_count_distribution.csv"
    figure_paths = [
        FIGURES_DIR / "stage2_4_candidate_seen_mm_matrix.png",
        FIGURES_DIR / "stage2_4_feature_availability.png",
    ]

    seen_count = group_counts["seen"]
    unseen_count = group_counts["unseen"]
    metrics = {
        "stage": "2.4",
        "candidate_count": row_count,
        "history_visibility": {
            "seen_count": seen_count,
            "seen_ratio": seen_count / row_count,
            "unseen_count": unseen_count,
            "unseen_ratio": unseen_count / row_count,
            "definition": "seen iff candidate OID exists in indexer['i']",
        },
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "unseen_with_any_side_feature": {
            "count": int(np.sum((~seen_all) & (available_counts_all > 0))),
            "ratio_within_unseen": (
                float(np.mean(available_counts_all[~seen_all] > 0)) if unseen_count else None
            ),
        },
        "seen_unseen_mm_matrix": matrix_rows,
        "mm_scan_audit": mm_audit,
        "interpretation_guardrails": [
            "cold_start values are reported field by field without assigning business meaning",
            "unseen is defined by OID mapping and is not forced to equal any cold_start value",
            "side/mm availability suggests a testable representation hypothesis, not a solved cold-start claim",
        ],
    }
    save_json(metrics, metrics_path)
    save_csv(matrix_rows, matrix_path)
    save_csv(cold_rows, cold_path)
    save_csv(availability_rows, availability_path)
    save_csv(available_distribution_rows, distribution_path)

    valid_values = [mm_matrix[(name, "valid_mm")] for name in ("seen", "unseen")]
    missing_values = [mm_matrix[(name, "missing_mm")] for name in ("seen", "unseen")]
    x = np.arange(2)
    plt.figure(figsize=(7, 5))
    plt.bar(x, valid_values, label="Valid MM", color="#70AD47")
    plt.bar(x, missing_values, bottom=valid_values, label="Missing MM", color="#A5A5A5")
    plt.xticks(x, ["Seen", "Unseen"])
    plt.ylabel("Candidates")
    plt.title("Candidate History Visibility and MM Availability")
    plt.legend()
    save_figure(figure_paths[0])

    seen_ratios = [
        availability_counts[(feature, "seen")] / seen_count if seen_count else 0
        for feature in feature_names
    ]
    unseen_ratios = [
        availability_counts[(feature, "unseen")] / unseen_count if unseen_count else 0
        for feature in feature_names
    ]
    x = np.arange(len(feature_names))
    width = 0.4
    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, seen_ratios, width=width, label="Seen")
    plt.bar(x + width / 2, unseen_ratios, width=width, label="Unseen")
    plt.xticks(x, feature_names, rotation=45)
    plt.ylabel("Non-null feature_value ratio")
    plt.title("Candidate Side-feature Availability")
    plt.ylim(0, 1.05)
    plt.legend()
    save_figure(figure_paths[1])

    print_outputs(
        STAGE,
        [metrics_path, matrix_path, cold_path, availability_path, distribution_path, *figure_paths],
    )


if __name__ == "__main__":
    main()
