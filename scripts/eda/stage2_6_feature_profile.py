"""
所属阶段：
阶段 2.6：匿名用户/广告特征 Profile

当前问题：
匿名侧信息的缺失程度、实际基数和单值/列表结构尚未形成统一统计画像。

本步目的：
逐字段量化 dtype、缺失率、基数和列表长度，为后续 Feature Embedding 设计提供依据。

为什么现在做：
候选可见性和多模态覆盖已经明确，现在补齐结构化侧信息的统计基础。

输入：
user_feat、item_feat、candidate。

输出：
三个特征 Profile CSV、candidate cold_start CSV、JSON 指标、PNG 图、终端日志。

完成标准：
每个匿名字段都有可复核的缺失率、基数；列表字段额外具有长度和空列表统计。

对后续模型的潜在影响：
为特征 Embedding、列表聚合和缺失感知设计提出候选方案，但不在本阶段决定维度或编码方法。

面试时如何讲：
面对匿名特征，我只依据缺失率、基数和结构做工程判断，不臆测字段的真实业务含义。
"""

from __future__ import annotations

from collections import Counter
import json

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from common import (
    CANDIDATE_DIR,
    FIGURES_DIR,
    ITEM_FEAT_DIR,
    METRICS_DIR,
    TABLES_DIR,
    TABLE_BATCH_SIZE,
    USER_FEAT_DIR,
    ensure_output_dirs,
    plt,
    print_outputs,
    quantile_summary,
    require_paths,
    save_csv,
    save_figure,
    save_json,
    to_numpy_int,
)


STAGE = "阶段 2.6"


def collect_column(dataset: ds.Dataset, field_name: str) -> list[pa.Array]:
    chunks: list[pa.Array] = []
    scanner = dataset.scanner(columns=[field_name], batch_size=TABLE_BATCH_SIZE)
    for batch_number, batch in enumerate(scanner.to_batches(), start=1):
        chunks.append(batch.column(0))
        if batch_number % 20 == 0:
            print(f"  字段 {field_name}：已读取 {batch_number} 个批次")
    return chunks


def count_distinct_chunks(chunks: list[pa.Array]) -> int:
    if not chunks:
        return 0
    chunked = pa.chunked_array(chunks)
    return int(pc.count_distinct(chunked, mode="only_valid").as_py())


def profile_regular_dataset(
    table_name: str,
    path,
    id_column: str,
) -> list[dict]:
    print(f"\n分析 {table_name}：{path.resolve()}")
    dataset = ds.dataset(path, format="parquet")
    feature_names = [name for name in dataset.schema.names if name != id_column]
    rows = []

    for feature_number, field_name in enumerate(feature_names, start=1):
        field = dataset.schema.field(field_name)
        chunks = collect_column(dataset, field_name)
        row_count = sum(len(chunk) for chunk in chunks)
        null_count = sum(chunk.null_count for chunk in chunks)
        row = {
            "table": table_name,
            "feature": field_name,
            "dtype": str(field.type),
            "row_count": row_count,
            "null_count": null_count,
            "non_null_count": row_count - null_count,
            "non_null_ratio": (row_count - null_count) / row_count if row_count else None,
            "unique_count": None,
            "unique_count_definition": None,
            "list_length_min": None,
            "list_length_median": None,
            "list_length_p90": None,
            "list_length_p99": None,
            "list_length_max": None,
            "empty_list_count": None,
            "empty_list_ratio": None,
        }

        if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
            lengths_parts = []
            value_chunks = []
            empty_count = 0
            for chunk in chunks:
                lengths = to_numpy_int(pc.list_value_length(chunk), fill_value=-1)
                valid_lengths = lengths[lengths >= 0]
                lengths_parts.append(valid_lengths)
                empty_count += int(np.sum(valid_lengths == 0))
                flattened = pc.list_flatten(chunk)
                if len(flattened):
                    value_chunks.append(flattened)
            all_lengths = np.concatenate(lengths_parts) if lengths_parts else np.empty(0, dtype=np.int64)
            length_summary = quantile_summary(all_lengths, (0.0, 0.5, 0.9, 0.99, 1.0))
            row.update(
                {
                    "unique_count": count_distinct_chunks(value_chunks),
                    "unique_count_definition": "distinct non-null list elements",
                    "list_length_min": length_summary["min"],
                    "list_length_median": length_summary["p50"],
                    "list_length_p90": length_summary["p90"],
                    "list_length_p99": length_summary["p99"],
                    "list_length_max": length_summary["max"],
                    "empty_list_count": empty_count,
                    "empty_list_ratio": empty_count / (row_count - null_count) if row_count > null_count else None,
                }
            )
        else:
            row["unique_count"] = count_distinct_chunks(chunks)
            row["unique_count_definition"] = "distinct non-null scalar values"

        rows.append(row)
        print(f"完成字段 {feature_number}/{len(feature_names)}：{field_name}")
        del chunks
    return rows


def profile_candidate() -> tuple[list[dict], list[dict]]:
    print(f"\n分析 candidate：{CANDIDATE_DIR.resolve()}")
    dataset = ds.dataset(CANDIDATE_DIR, format="parquet")
    feature_names = [name for name in dataset.schema.names if name not in {"item_id", "retrieval_id"}]
    profile_rows = []
    cold_rows = []

    for feature_number, field_name in enumerate(feature_names, start=1):
        field = dataset.schema.field(field_name)
        if not pa.types.is_struct(field.type):
            raise ValueError(f"candidate 字段 {field_name} 不是 struct")
        chunks = collect_column(dataset, field_name)
        row_count = sum(len(chunk) for chunk in chunks)
        struct_null_count = sum(chunk.null_count for chunk in chunks)
        feature_value_chunks = []
        non_null_count = 0
        cold_counter = Counter()

        for chunk in chunks:
            parent_null = chunk.is_null().to_numpy(zero_copy_only=False)
            feature_values = chunk.field("feature_value")
            feature_null = feature_values.is_null().to_numpy(zero_copy_only=False) | parent_null
            non_null_count += int(np.sum(~feature_null))
            feature_value_chunks.append(feature_values.filter(pa.array(~feature_null)))

            cold_values = chunk.field("cold_start").to_pylist()
            for is_parent_null, value in zip(parent_null, cold_values):
                cold_counter["null" if is_parent_null or value is None else str(value)] += 1

        profile_rows.append(
            {
                "table": "candidate",
                "feature": field_name,
                "dtype": str(field.type),
                "row_count": row_count,
                "struct_null_count": struct_null_count,
                "feature_value_null_count": row_count - non_null_count,
                "feature_value_non_null_count": non_null_count,
                "feature_value_non_null_ratio": non_null_count / row_count if row_count else None,
                "feature_value_unique_count": count_distinct_chunks(feature_value_chunks),
                "cold_start_distribution": json.dumps(dict(sorted(cold_counter.items())), ensure_ascii=False),
            }
        )
        for value, count in sorted(cold_counter.items()):
            cold_rows.append(
                {
                    "feature": field_name,
                    "cold_start_value": value,
                    "candidate_count": count,
                    "candidate_ratio": count / row_count if row_count else None,
                }
            )
        print(f"完成 candidate 字段 {feature_number}/{len(feature_names)}：{field_name}")
        del chunks
    return profile_rows, cold_rows


def plot_non_null_ratios(rows: list[dict], value_key: str, title: str, path) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar([row["feature"] for row in rows], [row[value_key] for row in rows], color="#4472C4")
    plt.xticks(rotation=45)
    plt.ylabel("Non-null ratio")
    plt.xlabel("Anonymous feature")
    plt.title(title)
    plt.ylim(0, 1.05)
    save_figure(path)


def main() -> None:
    print("阶段 2.6：匿名用户/广告特征 Profile")
    require_paths([USER_FEAT_DIR, ITEM_FEAT_DIR, CANDIDATE_DIR])
    ensure_output_dirs()

    user_rows = profile_regular_dataset("user_feat", USER_FEAT_DIR, "user_id")
    item_rows = profile_regular_dataset("item_feat", ITEM_FEAT_DIR, "item_id")
    candidate_rows, candidate_cold_rows = profile_candidate()

    user_path = TABLES_DIR / "stage2_6_user_feature_profile.csv"
    item_path = TABLES_DIR / "stage2_6_item_feature_profile.csv"
    candidate_path = TABLES_DIR / "stage2_6_candidate_feature_profile.csv"
    candidate_cold_path = TABLES_DIR / "stage2_6_candidate_cold_start_distribution.csv"
    metrics_path = METRICS_DIR / "stage2_6_feature_profile.json"
    figure_paths = [
        FIGURES_DIR / "stage2_6_user_feature_non_null_ratio.png",
        FIGURES_DIR / "stage2_6_item_feature_non_null_ratio.png",
        FIGURES_DIR / "stage2_6_candidate_feature_non_null_ratio.png",
    ]

    save_csv(user_rows, user_path)
    save_csv(item_rows, item_path)
    save_csv(candidate_rows, candidate_path)
    save_csv(candidate_cold_rows, candidate_cold_path)
    save_json(
        {
            "stage": "2.6",
            "user_feature_count": len(user_rows),
            "item_feature_count": len(item_rows),
            "candidate_feature_count": len(candidate_rows),
            "profile_tables": [str(user_path), str(item_path), str(candidate_path)],
            "unique_count_policy": {
                "scalar": "distinct non-null scalar values",
                "list": "distinct non-null elements after flattening",
                "candidate": "distinct non-null feature_value strings",
            },
            "interpretation_guardrails": [
                "Anonymous features are not assigned business semantics.",
                "High cardinality alone does not imply hashing.",
                "Embedding dimensions are not selected in stage 2.",
            ],
        },
        metrics_path,
    )

    plot_non_null_ratios(
        user_rows,
        "non_null_ratio",
        "User Feature Non-null Ratios",
        figure_paths[0],
    )
    plot_non_null_ratios(
        item_rows,
        "non_null_ratio",
        "Historical Item Feature Non-null Ratios",
        figure_paths[1],
    )
    plot_non_null_ratios(
        candidate_rows,
        "feature_value_non_null_ratio",
        "Candidate Feature-value Non-null Ratios",
        figure_paths[2],
    )

    print_outputs(
        STAGE,
        [user_path, item_path, candidate_path, candidate_cold_path, metrics_path, *figure_paths],
    )


if __name__ == "__main__":
    main()
