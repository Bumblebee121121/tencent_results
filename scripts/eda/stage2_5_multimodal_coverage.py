"""
所属阶段：
阶段 2.5：多模态覆盖与缺失模式

当前问题：
多模态覆盖率较高但并非 100%，尚不清楚缺失是否集中在 seen/unseen 或 head/mid/tail 群体。

本步目的：
区分 ID coverage 与 valid embedding coverage，并按候选可见性和历史热度分组。

为什么现在做：
2.3 已生成历史热度聚合，2.4 已明确 seen/unseen 口径，本步可复用这些口径分析缺失模式。

输入：
32D mm_emb、indexer.pkl、candidate、2.3 item popularity Parquet/JSON。

输出：
JSON 指标、CSV 分组覆盖表、PNG 图、终端日志。

完成标准：
报告历史和候选的有效多模态覆盖，并量化 seen/unseen、head/mid/tail 之间的覆盖差异。

对后续模型的潜在影响：
为 zero-fill + mask、learnable missing embedding 和 gating 的后续消融实验提供依据。

面试时如何讲：
我区分“存在多模态记录”和“存在合法非空向量”，再按关键群体报告缺失，而不假设缺失随机。
"""

from __future__ import annotations

import json

import numpy as np
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


STAGE = "阶段 2.5"
POPULARITY_PATH = TABLES_DIR / "stage2_3_item_popularity.parquet"
POPULARITY_METRICS_PATH = METRICS_DIR / "stage2_3_item_long_tail.json"
GROUP_NAMES = ("tail", "mid", "head")


def load_popularity_groups() -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    with POPULARITY_METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)
    definition = metrics["group_definition"]
    p50 = float(definition["p50_threshold"])
    p90 = float(definition["p90_threshold"])

    dataset = ds.dataset(POPULARITY_PATH, format="parquet")
    require_columns(dataset, ["item_id", "total_event_count"])
    item_parts = []
    group_parts = []
    scanner = dataset.scanner(
        columns=["item_id", "total_event_count"],
        batch_size=TABLE_BATCH_SIZE,
    )
    for batch in scanner.to_batches():
        item_ids = to_numpy_int(batch.column(0))
        counts = to_numpy_int(batch.column(1))
        groups = np.zeros(batch.num_rows, dtype=np.int8)
        groups[(counts > p50) & (counts <= p90)] = 1
        groups[counts > p90] = 2
        item_parts.append(item_ids)
        group_parts.append(groups)
    return np.concatenate(item_parts), np.concatenate(group_parts), {"p50": p50, "p90": p90}


def map_mm_to_history_rids(
    mapping: dict[int, int],
    mm_oids: np.ndarray,
    mm_valid: np.ndarray,
    max_rid: int,
) -> tuple[np.ndarray, np.ndarray]:
    id_covered = np.zeros(max_rid + 1, dtype=bool)
    valid_covered = np.zeros(max_rid + 1, dtype=bool)
    oid_buffer = np.empty(100_000, dtype=np.int64)
    rid_buffer = np.empty(100_000, dtype=np.int64)
    buffered = 0
    processed = 0

    def flush(size: int) -> None:
        if size == 0:
            return
        oids = oid_buffer[:size]
        rids = rid_buffer[:size]
        found, valid = lookup_sorted_flags(oids, mm_oids, mm_valid)
        id_covered[rids[found]] = True
        valid_covered[rids[valid]] = True

    for oid, rid in mapping.items():
        if rid <= 0 or rid > max_rid:
            raise ValueError(f"indexer RID 超出 popularity 表范围：{rid}")
        oid_buffer[buffered] = int(oid)
        rid_buffer[buffered] = int(rid)
        buffered += 1
        processed += 1
        if buffered == oid_buffer.size:
            flush(buffered)
            buffered = 0
            if processed % 1_000_000 == 0:
                print(f"已对齐 {processed:,} 个历史 OID -> RID")
    flush(buffered)
    return id_covered, valid_covered


def coverage_row(
    group_type: str,
    group_name: str,
    item_count: int,
    id_count: int | None,
    valid_count: int,
) -> dict[str, float | int | str | None]:
    return {
        "group_type": group_type,
        "group_name": group_name,
        "item_count": item_count,
        "mm_id_covered_count": id_count,
        "mm_id_coverage_ratio": id_count / item_count if id_count is not None and item_count else None,
        "valid_mm_count": valid_count,
        "valid_mm_ratio": valid_count / item_count if item_count else None,
        "missing_mm_count": item_count - valid_count,
        "missing_mm_ratio": (item_count - valid_count) / item_count if item_count else None,
    }


def main() -> None:
    print("阶段 2.5：多模态覆盖与缺失模式")
    require_paths(
        [MM_DIR, INDEXER_PATH, CANDIDATE_DIR, POPULARITY_PATH, POPULARITY_METRICS_PATH]
    )
    ensure_output_dirs()

    item_ids, popularity_groups, thresholds = load_popularity_groups()
    max_rid = int(item_ids.max())
    group_by_rid = np.full(max_rid + 1, -1, dtype=np.int8)
    group_by_rid[item_ids] = popularity_groups

    mapping = load_item_mapping()
    historical_oids = sorted_mapping_oids(mapping)
    mm_oids, mm_valid, mm_audit = scan_mm_oid_validity()
    history_id_covered, history_valid_covered = map_mm_to_history_rids(
        mapping, mm_oids, mm_valid, max_rid
    )

    history_rows = []
    history_rows.append(
        coverage_row(
            "history_popularity",
            "all",
            int(item_ids.size),
            int(np.sum(history_id_covered[item_ids])),
            int(np.sum(history_valid_covered[item_ids])),
        )
    )
    for code, group_name in enumerate(GROUP_NAMES):
        group_item_ids = item_ids[popularity_groups == code]
        history_rows.append(
            coverage_row(
                "history_popularity",
                group_name,
                int(group_item_ids.size),
                int(np.sum(history_id_covered[group_item_ids])),
                int(np.sum(history_valid_covered[group_item_ids])),
            )
        )

    candidate_dataset = ds.dataset(CANDIDATE_DIR, format="parquet")
    require_columns(candidate_dataset, ["item_id"])
    candidate_counts = {"all": 0, "seen": 0, "unseen": 0}
    candidate_valid_counts = {"all": 0, "seen": 0, "unseen": 0}
    scanner = candidate_dataset.scanner(columns=["item_id"], batch_size=TABLE_BATCH_SIZE)
    for batch_number, batch in enumerate(scanner.to_batches(), start=1):
        oids = to_numpy_int(batch.column(0))
        seen = membership_mask(oids, historical_oids)
        _, valid = lookup_sorted_flags(oids, mm_oids, mm_valid)
        candidate_counts["all"] += batch.num_rows
        candidate_counts["seen"] += int(np.sum(seen))
        candidate_counts["unseen"] += int(np.sum(~seen))
        candidate_valid_counts["all"] += int(np.sum(valid))
        candidate_valid_counts["seen"] += int(np.sum(valid & seen))
        candidate_valid_counts["unseen"] += int(np.sum(valid & ~seen))
        if batch_number % 5 == 0:
            print(f"已处理 {batch_number} 个 candidate 批次")

    candidate_rows = [
        coverage_row(
            "candidate_visibility",
            group_name,
            candidate_counts[group_name],
            None,
            candidate_valid_counts[group_name],
        )
        for group_name in ("all", "seen", "unseen")
    ]
    all_rows = history_rows + candidate_rows

    popularity_valid_ratios = [row["valid_mm_ratio"] for row in history_rows[1:]]
    valid_ratio_gap = float(max(popularity_valid_ratios) - min(popularity_valid_ratios))
    candidate_seen_ratio = candidate_rows[1]["valid_mm_ratio"]
    candidate_unseen_ratio = candidate_rows[2]["valid_mm_ratio"]
    candidate_gap = (
        float(candidate_seen_ratio - candidate_unseen_ratio)
        if candidate_seen_ratio is not None and candidate_unseen_ratio is not None
        else None
    )

    metrics_path = METRICS_DIR / "stage2_5_multimodal_coverage.json"
    coverage_path = TABLES_DIR / "stage2_5_mm_coverage_by_group.csv"
    candidate_path = TABLES_DIR / "stage2_5_candidate_mm_coverage.csv"
    figure_paths = [
        FIGURES_DIR / "stage2_5_mm_coverage_by_popularity.png",
        FIGURES_DIR / "stage2_5_candidate_mm_coverage.png",
    ]
    metrics = {
        "stage": "2.5",
        "popularity_thresholds": thresholds,
        "coverage_by_group": all_rows,
        "coverage_difference_percentage_points": {
            "history_head_mid_tail_max_minus_min": valid_ratio_gap * 100,
            "candidate_seen_minus_unseen": candidate_gap * 100 if candidate_gap is not None else None,
        },
        "mm_scan_audit": mm_audit,
        "interpretation_guardrail": (
            "Report effect-size differences in coverage; do not call them causal or model improvements."
        ),
    }
    save_json(metrics, metrics_path)
    save_csv(all_rows, coverage_path)
    save_csv(candidate_rows, candidate_path)

    plt.figure(figsize=(8, 5))
    plt.bar(
        [row["group_name"] for row in history_rows[1:]],
        [row["valid_mm_ratio"] for row in history_rows[1:]],
        color=["#A5A5A5", "#5B9BD5", "#ED7D31"],
    )
    plt.ylabel("Valid MM coverage ratio")
    plt.xlabel("Historical popularity group")
    plt.title("Valid Multimodal Coverage by Item Popularity")
    plt.ylim(0, 1.05)
    save_figure(figure_paths[0])

    plt.figure(figsize=(7, 5))
    plt.bar(
        [row["group_name"] for row in candidate_rows],
        [row["valid_mm_ratio"] for row in candidate_rows],
        color=["#4472C4", "#70AD47", "#FFC000"],
    )
    plt.ylabel("Valid MM coverage ratio")
    plt.xlabel("Candidate history visibility")
    plt.title("Candidate Valid Multimodal Coverage")
    plt.ylim(0, 1.05)
    save_figure(figure_paths[1])

    print_outputs(STAGE, [metrics_path, coverage_path, candidate_path, *figure_paths])


if __name__ == "__main__":
    main()
