"""
所属阶段：
阶段 2.3：广告热度与长尾分析

当前问题：
尚不清楚历史广告交互是否高度集中，以及曝光与点击是否具有明显长尾。

本步目的：
量化 popularity bias、头尾部结构和热门广告对总行为的贡献。

为什么现在做：
在进入候选冷启动和多模态覆盖分析前，需要先建立历史广告热度分组。

输入：
data/TencentGR-1M/seq、data/TencentGR-1M/item_feat。

输出：
JSON 指标、CSV 汇总、可复用的 item popularity Parquet、PNG 图、终端日志。

完成标准：
回答交互是否长尾、Popularity baseline 是否必要，以及后续是否应按 head/tail 报告指标。

对后续模型的潜在影响：
为 popularity baseline、头尾部分组评估和多模态缺失分组提出可检验假设。

面试时如何讲：
我用全量历史交互构造可复核的热度分层，再讨论潜在热门偏置，而不直接声称模型会失败。
"""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from common import (
    FIGURES_DIR,
    ITEM_FEAT_DIR,
    METRICS_DIR,
    SEQ_BATCH_SIZE,
    SEQ_DIR,
    TABLES_DIR,
    TABLE_BATCH_SIZE,
    ensure_output_dirs,
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


STAGE = "阶段 2.3"
GROUP_NAMES = ("tail", "mid", "head")


def load_historical_item_ids() -> np.ndarray:
    dataset = ds.dataset(ITEM_FEAT_DIR, format="parquet")
    require_columns(dataset, ["item_id"])
    parts = []
    scanner = dataset.scanner(columns=["item_id"], batch_size=TABLE_BATCH_SIZE)
    for batch in scanner.to_batches():
        parts.append(to_numpy_int(batch.column(0)))
    item_ids = np.concatenate(parts)
    if np.any(item_ids <= 0):
        raise ValueError("item_feat 中存在非正 RID")
    return item_ids


def popularity_band_rows(counts: np.ndarray) -> list[dict[str, float | int | str]]:
    specs = [
        ("1", 1, 1),
        ("2-5", 2, 5),
        ("6-10", 6, 10),
        ("11-100", 11, 100),
        (">100", 101, None),
    ]
    rows = []
    for label, lower, upper in specs:
        mask = counts >= lower if upper is None else (counts >= lower) & (counts <= upper)
        count = int(np.sum(mask))
        rows.append({"event_count_band": label, "item_count": count, "item_ratio": count / counts.size})
    return rows


def main() -> None:
    print("阶段 2.3：广告热度与长尾分析")
    print(f"输入路径：{SEQ_DIR.resolve()}")
    require_paths([SEQ_DIR, ITEM_FEAT_DIR])
    ensure_output_dirs()

    item_ids = load_historical_item_ids()
    max_rid = int(item_ids.max())
    array_size = max_rid + 1
    total_counts = np.zeros(array_size, dtype=np.uint32)
    exposure_counts = np.zeros(array_size, dtype=np.uint32)
    click_counts = np.zeros(array_size, dtype=np.uint32)

    dataset = ds.dataset(SEQ_DIR, format="parquet")
    require_columns(dataset, ["seq"])
    user_count = 0
    event_count = 0
    unexpected_action_count = 0

    scanner = dataset.scanner(columns=["seq"], batch_size=SEQ_BATCH_SIZE)
    for batch_number, batch in enumerate(scanner.to_batches(), start=1):
        events = pc.list_flatten(batch.column(0))
        items = to_numpy_int(events.field("item_id"))
        actions_array = events.field("action_type")
        actions = to_numpy_int(actions_array, fill_value=-1)
        if np.any((items <= 0) | (items > max_rid)):
            raise ValueError("seq 中出现超出 item_feat RID 范围的 item_id")

        total_counts += np.bincount(items, minlength=array_size).astype(np.uint32, copy=False)
        exposure_counts += np.bincount(items[actions == 0], minlength=array_size).astype(
            np.uint32, copy=False
        )
        click_counts += np.bincount(items[actions == 1], minlength=array_size).astype(
            np.uint32, copy=False
        )
        unexpected_action_count += int(np.sum(~np.isin(actions, [-1, 0, 1])))
        user_count += batch.num_rows
        event_count += len(events)

        if batch_number % 20 == 0:
            print(
                f"已处理 {batch_number} 个批次，累计用户数：{user_count:,}，"
                f"累计行为数：{event_count:,}"
            )

    if unexpected_action_count:
        raise ValueError(f"发现 {unexpected_action_count} 条意外 action_type")
    if int(np.sum(total_counts, dtype=np.uint64)) != event_count:
        raise RuntimeError("item 总交互计数与 seq 展平事件数不一致")

    item_total = total_counts[item_ids].astype(np.uint64, copy=False)
    item_exposure = exposure_counts[item_ids].astype(np.uint64, copy=False)
    item_click = click_counts[item_ids].astype(np.uint64, copy=False)
    item_unknown = item_total - item_exposure - item_click
    if np.any(item_total == 0):
        raise ValueError("item_feat 中存在未在 seq 出现的 RID，无法按既定历史 item 口径分组")

    p50, p90 = np.quantile(item_total, [0.5, 0.9])
    group_codes = np.zeros(item_total.size, dtype=np.int8)
    group_codes[(item_total > p50) & (item_total <= p90)] = 1
    group_codes[item_total > p90] = 2

    sorted_desc = np.sort(item_total)[::-1]
    cumulative = np.cumsum(sorted_desc, dtype=np.uint64)
    total_events_from_items = int(cumulative[-1])
    contribution_rows = []
    for fraction in (0.01, 0.05, 0.10, 0.20):
        top_n = max(1, math.ceil(item_total.size * fraction))
        contribution = int(cumulative[top_n - 1])
        contribution_rows.append(
            {
                "top_item_fraction": fraction,
                "top_item_count": top_n,
                "event_count": contribution,
                "event_ratio": contribution / total_events_from_items,
            }
        )

    group_rows = []
    for code, name in enumerate(GROUP_NAMES):
        mask = group_codes == code
        group_event_count = int(np.sum(item_total[mask], dtype=np.uint64))
        group_rows.append(
            {
                "popularity_group": name,
                "item_count": int(np.sum(mask)),
                "item_ratio": float(np.mean(mask)),
                "event_count": group_event_count,
                "event_ratio": group_event_count / total_events_from_items,
            }
        )

    band_rows = popularity_band_rows(item_total)
    metrics_path = METRICS_DIR / "stage2_3_item_long_tail.json"
    band_table_path = TABLES_DIR / "stage2_3_item_popularity_bands.csv"
    contribution_table_path = TABLES_DIR / "stage2_3_top_item_contribution.csv"
    group_table_path = TABLES_DIR / "stage2_3_head_mid_tail_summary.csv"
    popularity_path = TABLES_DIR / "stage2_3_item_popularity.parquet"

    metrics = {
        "stage": "2.3",
        "user_count": user_count,
        "historical_item_count": int(item_ids.size),
        "event_count": event_count,
        "item_total_event_count": quantile_summary(item_total),
        "item_exposure_count": quantile_summary(item_exposure),
        "item_click_count": quantile_summary(item_click),
        "item_unknown_count": quantile_summary(item_unknown),
        "popularity_bands": band_rows,
        "top_item_contribution": contribution_rows,
        "group_definition": {
            "tail": "total_event_count <= p50",
            "mid": "p50 < total_event_count <= p90",
            "head": "total_event_count > p90",
            "p50_threshold": float(p50),
            "p90_threshold": float(p90),
        },
        "group_summary": group_rows,
        "unique_user_count_note": "Not computed because exact item-user pairs would add substantial memory cost.",
    }
    save_json(metrics, metrics_path)
    save_csv(band_rows, band_table_path)
    save_csv(contribution_rows, contribution_table_path)
    save_csv(group_rows, group_table_path)

    dictionary = pa.array(GROUP_NAMES)
    group_array = pa.DictionaryArray.from_arrays(pa.array(group_codes, type=pa.int8()), dictionary)
    popularity_table = pa.table(
        {
            "item_id": pa.array(item_ids, type=pa.int64()),
            "total_event_count": pa.array(item_total, type=pa.uint64()),
            "exposure_count": pa.array(item_exposure, type=pa.uint64()),
            "click_count": pa.array(item_click, type=pa.uint64()),
            "unknown_count": pa.array(item_unknown, type=pa.uint64()),
            "popularity_group": group_array,
        }
    )
    pq.write_table(popularity_table, popularity_path, compression="snappy")

    figure_paths = [
        FIGURES_DIR / "stage2_3_item_popularity_loglog.png",
        FIGURES_DIR / "stage2_3_item_popularity_cdf.png",
        FIGURES_DIR / "stage2_3_head_tail_contribution.png",
    ]
    ranks = np.arange(1, sorted_desc.size + 1)
    if ranks.size > 100_000:
        sample_indices = np.unique(
    np.geomspace(
        1,
        ranks.size,
        100_000,
    ).astype(np.int64) - 1
)
    else:
        sample_indices = np.arange(ranks.size)
    plt.figure(figsize=(8, 5))
    plt.loglog(ranks[sample_indices], sorted_desc[sample_indices], linewidth=1)
    plt.xlabel("Item popularity rank")
    plt.ylabel("Total event count")
    plt.title("Item Popularity Rank-Frequency Curve")
    plt.grid(alpha=0.25)
    save_figure(figure_paths[0])

    sorted_asc = sorted_desc[::-1]
    plt.figure(figsize=(8, 5))
    plt.step(sorted_asc, np.arange(1, sorted_asc.size + 1) / sorted_asc.size, where="post")
    plt.xscale("log")
    plt.xlabel("Total event count per item (log scale)")
    plt.ylabel("Cumulative item ratio")
    plt.title("CDF of Item Popularity")
    plt.grid(alpha=0.25)
    save_figure(figure_paths[1])

    plt.figure(figsize=(7, 5))
    plt.bar(
        [row["popularity_group"] for row in group_rows],
        [row["event_ratio"] for row in group_rows],
        color=["#A5A5A5", "#5B9BD5", "#ED7D31"],
    )
    plt.xlabel("Popularity group")
    plt.ylabel("Share of historical events")
    plt.title("Historical Event Contribution by Popularity Group")
    plt.ylim(0, 1)
    save_figure(figure_paths[2])

    print_outputs(
        STAGE,
        [
            metrics_path,
            band_table_path,
            contribution_table_path,
            group_table_path,
            popularity_path,
            *figure_paths,
        ],
    )


if __name__ == "__main__":
    main()
