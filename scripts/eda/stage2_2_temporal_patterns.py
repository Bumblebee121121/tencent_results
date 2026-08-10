"""
所属阶段：
阶段 2.2：时间行为模式

当前问题：
尚不清楚用户历史覆盖时长以及相邻行为之间的时间间隔。

本步目的：
量化 history span 和 event time gap，为后续 time-gap / recency 假设提供依据。

为什么现在做：
阶段 2.1 先确认序列结构，本步再判断相同长度序列是否覆盖完全不同的时间尺度。

输入：
data/TencentGR-1M/seq 中的 timestamp。

输出：
JSON 指标、CSV 汇总、PNG 图、终端日志。

完成标准：
获得历史跨度和相邻间隔的分位数、零间隔比例、可解释时间区间与极端间隔计数。

对后续模型的潜在影响：
为比较 position-only 与 position + time-gap / recency encoding 提出可检验假设。

面试时如何讲：
我先量化序列覆盖的真实时间尺度，再决定是否引入时间编码，不把相关性写成模型收益。
"""

from __future__ import annotations

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

from common import (
    DiskBackedInt64,
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
    save_csv,
    save_figure,
    save_json,
    to_numpy_int,
)


STAGE = "阶段 2.2"
SECONDS_PER_DAY = 86400


def main() -> None:
    print("阶段 2.2：时间行为模式")
    print(f"输入路径：{SEQ_DIR.resolve()}")
    require_paths([SEQ_DIR])
    ensure_output_dirs()

    dataset = ds.dataset(SEQ_DIR, format="parquet")
    require_columns(dataset, ["seq"])

    history_span_parts: list[np.ndarray] = []
    user_count = 0
    event_count = 0
    zero_gap_count = 0
    negative_gap_count = 0
    gap_band_counts = {
        "0_seconds": 0,
        "1_to_59_seconds": 0,
        "1_to_59_minutes": 0,
        "1_to_23_hours": 0,
        "1_to_6_days": 0,
        "7_to_29_days": 0,
        "30_days_or_more": 0,
    }

    temp_path = METRICS_DIR / ".stage2_2_delta_t.int64.tmp"
    quantiles = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)

    with DiskBackedInt64(temp_path) as gap_store:
        scanner = dataset.scanner(columns=["seq"], batch_size=SEQ_BATCH_SIZE)
        for batch_number, batch in enumerate(scanner.to_batches(), start=1):
            seq_array = batch.column(0)
            offsets, lengths = local_list_offsets(seq_array)
            events = pc.list_flatten(seq_array)
            timestamp_array = events.field("timestamp")
            if timestamp_array.null_count:
                raise ValueError(f"发现 {timestamp_array.null_count} 条缺失 timestamp")
            timestamps = to_numpy_int(timestamp_array)

            non_empty = lengths > 0
            first_indices = offsets[:-1][non_empty]
            last_indices = offsets[1:][non_empty] - 1
            spans = timestamps[last_indices] - timestamps[first_indices]
            if np.any(spans < 0):
                raise ValueError("发现负数 history_span")
            history_span_parts.append(spans.astype(np.int64, copy=False))

            if timestamps.size > 1:
                parent_rows = np.repeat(np.arange(batch.num_rows, dtype=np.int64), lengths)
                adjacent_mask = parent_rows[1:] == parent_rows[:-1]
                gaps = np.diff(timestamps)[adjacent_mask].astype(np.int64, copy=False)
            else:
                gaps = np.empty(0, dtype=np.int64)

            negative_gap_count += int(np.sum(gaps < 0))
            if np.any(gaps < 0):
                raise ValueError("发现负数相邻时间间隔，阶段 1 时间顺序事实不再成立")
            gap_store.append(gaps)

            zero_gap_count += int(np.sum(gaps == 0))
            gap_band_counts["0_seconds"] += int(np.sum(gaps == 0))
            gap_band_counts["1_to_59_seconds"] += int(np.sum((gaps >= 1) & (gaps < 60)))
            gap_band_counts["1_to_59_minutes"] += int(np.sum((gaps >= 60) & (gaps < 3600)))
            gap_band_counts["1_to_23_hours"] += int(np.sum((gaps >= 3600) & (gaps < SECONDS_PER_DAY)))
            gap_band_counts["1_to_6_days"] += int(
                np.sum((gaps >= SECONDS_PER_DAY) & (gaps < 7 * SECONDS_PER_DAY))
            )
            gap_band_counts["7_to_29_days"] += int(
                np.sum((gaps >= 7 * SECONDS_PER_DAY) & (gaps < 30 * SECONDS_PER_DAY))
            )
            gap_band_counts["30_days_or_more"] += int(np.sum(gaps >= 30 * SECONDS_PER_DAY))

            user_count += batch.num_rows
            event_count += len(events)
            if batch_number % 20 == 0:
                print(
                    f"已处理 {batch_number} 个批次，累计用户数：{user_count:,}，"
                    f"累计行为数：{event_count:,}，累计相邻间隔：{gap_store.count:,}"
                )

        history_spans = np.concatenate(history_span_parts)
        gap_quantiles = gap_store.exact_quantiles(quantiles)
        gap_hist_counts, gap_hist_edges = gap_store.log1p_histogram()
        gap_count = gap_store.count

    expected_gap_count = event_count - int(history_spans.size)
    if gap_count != expected_gap_count:
        raise RuntimeError(f"相邻间隔数量不闭合：实际 {gap_count}，预期 {expected_gap_count}")
    if sum(gap_band_counts.values()) != gap_count:
        raise RuntimeError("时间间隔区间计数未闭合")

    history_summary_seconds = quantile_summary(history_spans, quantiles)
    history_summary_days = {
        key: (value / SECONDS_PER_DAY if isinstance(value, (int, float)) and key != "count" else value)
        for key, value in history_summary_seconds.items()
    }
    gap_summary_seconds = {"count": gap_count, "mean": None, **gap_quantiles}
    # 避免为求 mean 再扫描 seq；从临时文件清理前已经无法访问，因此通过区间外单独累计并不划算。
    # 精确分位数和完整区间计数是本阶段的核心统计口径。
    gap_band_rows = [
        {
            "gap_band": name,
            "gap_count": count,
            "gap_ratio": count / gap_count if gap_count else None,
        }
        for name, count in gap_band_counts.items()
    ]
    quantile_rows = []
    for metric, summary, unit in (
        ("history_span", history_summary_seconds, "seconds"),
        ("history_span", history_summary_days, "days"),
        ("delta_t", gap_summary_seconds, "seconds"),
    ):
        quantile_rows.extend(
            {"metric": metric, "unit": unit, "statistic": key, "value": value}
            for key, value in summary.items()
        )

    metrics_path = METRICS_DIR / "stage2_2_temporal_patterns.json"
    band_table_path = TABLES_DIR / "stage2_2_event_time_gap_bands.csv"
    quantile_table_path = TABLES_DIR / "stage2_2_temporal_quantiles.csv"
    figure_paths = [
        FIGURES_DIR / "stage2_2_history_span_distribution.png",
        FIGURES_DIR / "stage2_2_event_time_gap_distribution.png",
    ]

    metrics = {
        "stage": "2.2",
        "user_count": user_count,
        "event_count": event_count,
        "history_span_seconds": history_summary_seconds,
        "history_span_days": history_summary_days,
        "delta_t_seconds": gap_summary_seconds,
        "delta_t_zero": {
            "count": zero_gap_count,
            "ratio": zero_gap_count / gap_count if gap_count else None,
        },
        "delta_t_bands": gap_band_rows,
        "extreme_gap_counts": {
            "at_least_7_days": gap_band_counts["7_to_29_days"] + gap_band_counts["30_days_or_more"],
            "at_least_30_days": gap_band_counts["30_days_or_more"],
        },
        "negative_gap_count": negative_gap_count,
        "interpretation_guardrail": "Temporal association only; no label construction or model-effect claim.",
    }
    save_json(metrics, metrics_path)
    save_csv(gap_band_rows, band_table_path)
    save_csv(quantile_rows, quantile_table_path)

    plt.figure(figsize=(8, 5))
    plt.hist(np.log1p(history_spans / SECONDS_PER_DAY), bins=100, color="#4472C4")
    plt.xlabel("log1p(history span in days)")
    plt.ylabel("Users")
    plt.title("Distribution of User History Span")
    save_figure(figure_paths[0])

    centers = (gap_hist_edges[:-1] + gap_hist_edges[1:]) / 2
    widths = np.diff(gap_hist_edges)
    plt.figure(figsize=(8, 5))
    plt.bar(centers, gap_hist_counts, width=widths, color="#ED7D31", align="center")
    plt.yscale("log")
    plt.xlabel("log1p(delta_t in seconds)")
    plt.ylabel("Adjacent event pairs (log scale)")
    plt.title("Distribution of Adjacent Event Time Gaps")
    save_figure(figure_paths[1])

    print_outputs(
        STAGE,
        [metrics_path, band_table_path, quantile_table_path, *figure_paths],
    )


if __name__ == "__main__":
    main()
