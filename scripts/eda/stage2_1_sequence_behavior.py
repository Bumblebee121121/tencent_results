"""
所属阶段：
阶段 2.1：用户序列长度与行为构成

当前问题：
尚不清楚用户序列长度、点击稀疏程度和未知行为分布。

本步目的：
量化用户行为序列结构，为后续序列模型和行为类型建模提供依据。

为什么现在做：
阶段 1 已证明 seq 数据可靠，现在才能对其真实分布做统计分析。

输入：
data/TencentGR-1M/seq

输出：
JSON 指标、CSV 汇总、PNG 图、终端日志。

完成标准：
能回答序列是否普遍较长、是否存在长度 100 聚集、点击是否稀疏。

对后续模型的潜在影响：
决定是否值得验证 SASRec、行为 Embedding、时间特征等方案。

面试时如何讲：
先通过 EDA 确认序列结构和行为稀疏性，再决定是否引入序列建模，
而不是直接堆叠 Transformer。
"""

from __future__ import annotations

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

from common import (
    FIGURES_DIR,
    METRICS_DIR,
    SEQ_BATCH_SIZE,
    SEQ_DIR,
    TABLES_DIR,
    ensure_output_dirs,
    local_list_offsets,
    plt,
    print_outputs,
    quantile_summary,
    require_columns,
    require_paths,
    safe_ratio,
    save_csv,
    save_figure,
    save_json,
)


STAGE = "阶段 2.1"


def count_by_parent(parent_rows: np.ndarray, mask: np.ndarray, row_count: int) -> np.ndarray:
    """按用户行聚合满足 mask 的事件数。"""
    return np.bincount(parent_rows[mask], minlength=row_count).astype(np.int64, copy=False)


def main() -> None:
    print("阶段 2.1：用户序列长度与行为构成")
    print(f"输入路径：{SEQ_DIR.resolve()}")
    require_paths([SEQ_DIR])
    ensure_output_dirs()

    dataset = ds.dataset(SEQ_DIR, format="parquet")
    require_columns(dataset, ["user_id", "seq"])

    length_parts: list[np.ndarray] = []
    exposure_parts: list[np.ndarray] = []
    click_parts: list[np.ndarray] = []
    unknown_parts: list[np.ndarray] = []

    user_count = 0
    event_count = 0
    exposure_count = 0
    click_count = 0
    unknown_count = 0
    unexpected_action_count = 0

    scanner = dataset.scanner(columns=["seq"], batch_size=SEQ_BATCH_SIZE)
    for batch_number, batch in enumerate(scanner.to_batches(), start=1):
        seq_array = batch.column(0)
        _, lengths = local_list_offsets(seq_array)
        events = pc.list_flatten(seq_array)
        actions = events.field("action_type")
        action_values = pc.fill_null(actions, -1).to_numpy(zero_copy_only=False)

        parent_rows = np.repeat(np.arange(batch.num_rows, dtype=np.int64), lengths)
        exposure_mask = action_values == 0
        click_mask = action_values == 1
        unknown_mask = actions.is_null().to_numpy(zero_copy_only=False)
        unexpected_mask = ~(exposure_mask | click_mask | unknown_mask)

        per_user_exposure = count_by_parent(parent_rows, exposure_mask, batch.num_rows)
        per_user_click = count_by_parent(parent_rows, click_mask, batch.num_rows)
        per_user_unknown = count_by_parent(parent_rows, unknown_mask, batch.num_rows)

        length_parts.append(lengths.astype(np.int64, copy=False))
        exposure_parts.append(per_user_exposure)
        click_parts.append(per_user_click)
        unknown_parts.append(per_user_unknown)

        user_count += batch.num_rows
        event_count += len(events)
        exposure_count += int(np.sum(exposure_mask))
        click_count += int(np.sum(click_mask))
        unknown_count += int(np.sum(unknown_mask))
        unexpected_action_count += int(np.sum(unexpected_mask))

        if batch_number % 20 == 0:
            print(
                f"已处理 {batch_number} 个批次，累计用户数：{user_count:,}，"
                f"累计行为数：{event_count:,}"
            )

    if unexpected_action_count:
        raise ValueError(f"发现 {unexpected_action_count} 条非 0/1 且非空的 action_type")

    lengths = np.concatenate(length_parts)
    exposures = np.concatenate(exposure_parts)
    clicks = np.concatenate(click_parts)
    unknowns = np.concatenate(unknown_parts)
    if int(np.sum(lengths)) != event_count:
        raise RuntimeError("序列长度合计与展平事件数不一致")
    if exposure_count + click_count + unknown_count != event_count:
        raise RuntimeError("三类行为计数未闭合")

    known_events = exposures + clicks
    click_event_ratio = safe_ratio(clicks, known_events)
    unknown_ratio = safe_ratio(unknowns, lengths)

    length_band_specs = [
        ("1-10", 1, 10),
        ("11-20", 11, 20),
        ("21-50", 21, 50),
        ("51-80", 51, 80),
        ("81-99", 81, 99),
        ("100", 100, 100),
    ]
    length_rows = []
    for label, lower, upper in length_band_specs:
        count = int(np.sum((lengths >= lower) & (lengths <= upper)))
        length_rows.append(
            {"length_band": label, "user_count": count, "user_ratio": count / user_count}
        )

    action_rows = []
    for label, count in (
        ("exposure_action_0", exposure_count),
        ("click_action_1", click_count),
        ("unknown_action_null", unknown_count),
    ):
        action_rows.append({"action_group": label, "event_count": count, "event_ratio": count / event_count})

    click_group_rows = []
    for label, mask in (
        ("click_count_0", clicks == 0),
        ("click_count_1", clicks == 1),
        ("click_count_ge_2", clicks >= 2),
    ):
        count = int(np.sum(mask))
        click_group_rows.append({"click_group": label, "user_count": count, "user_ratio": count / user_count})

    ratio_rows = []
    for name, values in (
        ("click_event_ratio", click_event_ratio),
        ("unknown_ratio", unknown_ratio),
    ):
        summary = quantile_summary(values)
        ratio_rows.extend(
            {"metric": name, "statistic": statistic, "value": value}
            for statistic, value in summary.items()
        )

    metrics_path = METRICS_DIR / "stage2_1_sequence_behavior.json"
    length_table_path = TABLES_DIR / "stage2_1_sequence_length_hist.csv"
    action_table_path = TABLES_DIR / "stage2_1_action_composition.csv"
    click_table_path = TABLES_DIR / "stage2_1_click_count_groups.csv"
    ratio_table_path = TABLES_DIR / "stage2_1_user_ratio_quantiles.csv"

    metrics = {
        "stage": "2.1",
        "input": str(SEQ_DIR.relative_to(SEQ_DIR.parents[2])),
        "user_count": user_count,
        "event_count": event_count,
        "sequence_length": quantile_summary(lengths),
        "length_100": {
            "user_count": int(np.sum(lengths == 100)),
            "user_ratio": float(np.mean(lengths == 100)),
        },
        "length_bands": length_rows,
        "action_composition": action_rows,
        "per_user": {
            "click_event_ratio_definition": "click_count / (exposure_count + click_count)",
            "click_event_ratio": quantile_summary(click_event_ratio),
            "undefined_click_event_ratio_user_count": int(np.sum(known_events == 0)),
            "unknown_ratio": quantile_summary(unknown_ratio),
            "click_count_groups": click_group_rows,
        },
        "interpretation_guardrails": [
            "click_event_ratio is not labeled CTR",
            "null action_type is not merged into exposure",
            "length=100 concentration is evidence of a ceiling, not proof of a hidden construction rule",
        ],
    }
    save_json(metrics, metrics_path)
    save_csv(length_rows, length_table_path)
    save_csv(action_rows, action_table_path)
    save_csv(click_group_rows, click_table_path)
    save_csv(ratio_rows, ratio_table_path)

    figure_paths = [
        FIGURES_DIR / "stage2_1_sequence_length_hist.png",
        FIGURES_DIR / "stage2_1_sequence_length_cdf.png",
        FIGURES_DIR / "stage2_1_click_count_per_user.png",
        FIGURES_DIR / "stage2_1_click_event_ratio_distribution.png",
        FIGURES_DIR / "stage2_1_unknown_ratio_distribution.png",
    ]

    plt.figure(figsize=(8, 5))
    plt.hist(lengths, bins=np.arange(0.5, int(lengths.max()) + 1.5), color="#4472C4")
    plt.xlabel("Sequence length (events)")
    plt.ylabel("Users")
    plt.title("User Sequence Length Distribution")
    save_figure(figure_paths[0])

    sorted_lengths = np.sort(lengths)
    plt.figure(figsize=(8, 5))
    plt.step(sorted_lengths, np.arange(1, user_count + 1) / user_count, where="post")
    plt.xlabel("Sequence length (events)")
    plt.ylabel("Cumulative user ratio")
    plt.title("CDF of User Sequence Length")
    plt.grid(alpha=0.25)
    save_figure(figure_paths[1])

    plt.figure(figsize=(8, 5))
    plt.hist(clicks, bins=np.arange(-0.5, int(clicks.max()) + 1.5), color="#ED7D31")
    plt.yscale("log")
    plt.xlabel("Click events per user")
    plt.ylabel("Users (log scale)")
    plt.title("Click Count per User")
    save_figure(figure_paths[2])

    finite_click_ratio = click_event_ratio[np.isfinite(click_event_ratio)]
    plt.figure(figsize=(8, 5))
    plt.hist(finite_click_ratio, bins=np.linspace(0, 1, 51), color="#70AD47")
    plt.xlabel("Click event ratio")
    plt.ylabel("Users")
    plt.title("Distribution of Click Event Ratio")
    save_figure(figure_paths[3])

    plt.figure(figsize=(8, 5))
    plt.hist(unknown_ratio[np.isfinite(unknown_ratio)], bins=np.linspace(0, 1, 51), color="#A5A5A5")
    plt.yscale("symlog", linthresh=1)
    plt.xlabel("Unknown action ratio")
    plt.ylabel("Users (symlog scale)")
    plt.title("Distribution of Unknown Action Ratio")
    save_figure(figure_paths[4])

    output_paths = [
        metrics_path,
        length_table_path,
        action_table_path,
        click_table_path,
        ratio_table_path,
        *figure_paths,
    ]
    print_outputs(STAGE, output_paths)


if __name__ == "__main__":
    main()
