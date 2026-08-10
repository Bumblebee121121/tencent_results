from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds


DATA_DIR = Path("data") / "TencentGR-1M"

SEQ_DIR = DATA_DIR / "seq"
USER_FEAT_DIR = DATA_DIR / "user_feat"
ITEM_FEAT_DIR = DATA_DIR / "item_feat"

BATCH_SIZE = 8192


def to_numpy_int(array, fill_value=-1):
    """将 Arrow 整数数组转换为 NumPy 数组。"""
    array = pc.fill_null(array, fill_value)

    return array.to_numpy(
        zero_copy_only=False
    ).astype(
        np.int64,
        copy=False,
    )


def load_primary_keys(path: Path, column: str):
    """
    读取一张特征表的主键，只检查：

    1. 总行数；
    2. 空主键；
    3. 非法主键；
    4. 重复主键。
    """
    dataset = ds.dataset(path, format="parquet")

    key_parts = []
    row_count = 0
    null_count = 0

    scanner = dataset.scanner(
        columns=[column],
        batch_size=65536,
    )

    for batch in scanner.to_batches():
        array = batch.column(0)

        row_count += batch.num_rows
        null_count += array.null_count
        key_parts.append(to_numpy_int(array))

    all_ids = np.concatenate(key_parts)
    valid_ids = all_ids[all_ids > 0]
    unique_ids = np.unique(valid_ids)

    invalid_count = int(np.sum(all_ids <= 0))
    duplicate_count = len(valid_ids) - len(unique_ids)

    print("\n" + "=" * 70)
    print(f"数据表：{path.name}")
    print(f"总行数：{row_count}")
    print(f"空主键数：{null_count}")
    print(f"非正数主键数：{invalid_count}")
    print(f"唯一主键数：{len(unique_ids)}")
    print(f"重复主键额外行数：{duplicate_count}")

    return unique_ids


def membership_mask(values, sorted_reference):
    """
    检查 values 是否存在于已排序的参考数组中。
    """
    positions = np.searchsorted(sorted_reference, values)

    result = np.zeros(len(values), dtype=bool)
    valid = positions < len(sorted_reference)

    result[valid] = (
        sorted_reference[positions[valid]]
        == values[valid]
    )

    return result


def scan_seq(user_feat_ids, item_feat_ids):
    """
    检查用户行为序列的基本健康状况和跨表覆盖率。
    """
    dataset = ds.dataset(SEQ_DIR, format="parquet")

    seq_user_parts = []

    row_count = 0
    null_user_count = 0
    null_seq_count = 0
    empty_seq_count = 0

    total_event_count = 0

    action_0_count = 0
    action_1_count = 0
    null_action_count = 0
    invalid_action_count = 0

    null_item_count = 0
    invalid_item_count = 0
    valid_item_event_count = 0
    matched_item_event_count = 0

    null_timestamp_count = 0
    reversed_pair_count = 0
    reversed_user_count = 0

    max_item_id = int(item_feat_ids.max())

    item_exists = np.zeros(
        max_item_id + 1,
        dtype=bool,
    )
    item_exists[item_feat_ids] = True

    seq_item_seen = np.zeros(
        max_item_id + 1,
        dtype=bool,
    )

    out_of_range_items = set()

    scanner = dataset.scanner(
        columns=["user_id", "seq"],
        batch_size=BATCH_SIZE,
    )

    for batch_number, batch in enumerate(
        scanner.to_batches(),
        start=1,
    ):
        row_count += batch.num_rows

        if batch_number % 20 == 0:
            print(
                f"已处理 {batch_number} 个批次，"
                f"累计用户数：{row_count}"
            )

        # ----------------------------------------------------------
        # 用户主键
        # ----------------------------------------------------------
        user_array = batch.column(0)

        null_user_count += user_array.null_count
        seq_user_parts.append(to_numpy_int(user_array))

        # ----------------------------------------------------------
        # 序列基本情况
        # ----------------------------------------------------------
        seq_array = batch.column(1)

        null_seq_count += seq_array.null_count

        offsets = seq_array.offsets.to_numpy(
            zero_copy_only=False
        )
        seq_lengths = np.diff(offsets)

        seq_valid_mask = ~seq_array.is_null().to_numpy(
            zero_copy_only=False
        )

        empty_seq_count += int(
            np.sum(
                (seq_lengths == 0)
                & seq_valid_mask
            )
        )

        events = pc.list_flatten(seq_array)
        event_count = len(events)

        total_event_count += event_count

        if event_count == 0:
            continue

        item_array = events.field("item_id")
        action_array = events.field("action_type")
        timestamp_array = events.field("timestamp")

        item_ids = to_numpy_int(item_array)
        action_values = to_numpy_int(
            action_array,
            fill_value=-999,
        )
        timestamps = to_numpy_int(
            timestamp_array,
            fill_value=-1,
        )

        # ----------------------------------------------------------
        # 行为类型检查
        # ----------------------------------------------------------
        null_action_count += action_array.null_count

        action_0_count += int(
            np.sum(action_values == 0)
        )
        action_1_count += int(
            np.sum(action_values == 1)
        )

        non_null_action = action_values != -999

        invalid_action_count += int(
            np.sum(
                non_null_action
                & ~np.isin(action_values, [0, 1])
            )
        )

        # ----------------------------------------------------------
        # 广告 ID 及 item_feat 覆盖率
        # ----------------------------------------------------------
        null_item_count += item_array.null_count

        valid_item_mask = item_ids > 0
        invalid_item_count += int(
            np.sum(~valid_item_mask)
        )

        valid_item_ids = item_ids[valid_item_mask]
        valid_item_event_count += len(valid_item_ids)

        in_range_mask = valid_item_ids <= max_item_id
        in_range_ids = valid_item_ids[in_range_mask]

        matched_item_event_count += int(
            np.sum(item_exists[in_range_ids])
        )

        seq_item_seen[in_range_ids] = True

        outside_ids = np.unique(
            valid_item_ids[~in_range_mask]
        )

        out_of_range_items.update(
            int(value)
            for value in outside_ids
        )

        # ----------------------------------------------------------
        # 时间顺序检查
        # ----------------------------------------------------------
        null_timestamp_count += timestamp_array.null_count

        event_user_index = np.repeat(
            np.arange(batch.num_rows),
            seq_lengths,
        )

        if event_count > 1:
            same_user = (
                event_user_index[1:]
                == event_user_index[:-1]
            )

            timestamps_valid = (
                (timestamps[1:] >= 0)
                & (timestamps[:-1] >= 0)
            )

            reversed_mask = (
                same_user
                & timestamps_valid
                & (timestamps[1:] < timestamps[:-1])
            )

            reversed_pair_count += int(
                np.sum(reversed_mask)
            )

            bad_user_rows = np.unique(
                event_user_index[:-1][reversed_mask]
            )

            reversed_user_count += len(bad_user_rows)

    # --------------------------------------------------------------
    # 用户主键和 user_feat 覆盖率
    # --------------------------------------------------------------
    all_seq_users = np.concatenate(seq_user_parts)
    valid_seq_users = all_seq_users[all_seq_users > 0]
    unique_seq_users = np.unique(valid_seq_users)

    duplicate_seq_user_count = (
        len(valid_seq_users)
        - len(unique_seq_users)
    )

    user_matched_mask = membership_mask(
        unique_seq_users,
        user_feat_ids,
    )

    matched_user_count = int(
        np.sum(user_matched_mask)
    )

    user_coverage = (
        matched_user_count / len(unique_seq_users)
        if len(unique_seq_users) > 0
        else 0.0
    )

    # --------------------------------------------------------------
    # 唯一广告覆盖率
    # --------------------------------------------------------------
    unique_seq_item_count = (
        int(np.sum(seq_item_seen))
        + len(out_of_range_items)
    )

    matched_unique_item_count = int(
        np.sum(seq_item_seen & item_exists)
    )

    unique_item_coverage = (
        matched_unique_item_count
        / unique_seq_item_count
        if unique_seq_item_count > 0
        else 0.0
    )

    event_item_coverage = (
        matched_item_event_count
        / valid_item_event_count
        if valid_item_event_count > 0
        else 0.0
    )

    print("\n" + "=" * 70)
    print("seq 核心检查结果")

    print(f"\n用户记录行数：{row_count}")
    print(f"唯一用户数：{len(unique_seq_users)}")
    print(f"缺失用户 ID 数：{null_user_count}")
    print(f"重复用户额外行数：{duplicate_seq_user_count}")
    print(f"缺失序列数：{null_seq_count}")
    print(f"空序列数：{empty_seq_count}")

    print(f"\n总行为数：{total_event_count}")
    print(f"曝光行为数 action_type=0：{action_0_count}")
    print(f"点击行为数 action_type=1：{action_1_count}")
    print(f"缺失行为类型数：{null_action_count}")
    print(f"非法行为类型数：{invalid_action_count}")

    print(f"\n缺失广告 ID 的行为数：{null_item_count}")
    print(f"非法广告 ID 的行为数：{invalid_item_count}")

    print(f"\n缺失时间戳的行为数：{null_timestamp_count}")
    print(f"时间戳逆序相邻对数：{reversed_pair_count}")
    print(f"存在时间逆序的用户数：{reversed_user_count}")

    print("\n跨表覆盖率：")
    print(f"user_feat 用户覆盖率：{user_coverage:.6%}")
    print(
        f"item_feat 唯一广告覆盖率："
        f"{unique_item_coverage:.6%}"
    )
    print(
        f"item_feat 行为级覆盖率："
        f"{event_item_coverage:.6%}"
    )

    if out_of_range_items:
        print(
            "\n超出 item_feat 最大 ID 范围的广告样例："
        )
        print(sorted(out_of_range_items)[:10])


def main():
    print("阶段 1.4A：核心数据健康检查")
    print(f"数据目录：{DATA_DIR.resolve()}")

    user_feat_ids = load_primary_keys(
        USER_FEAT_DIR,
        "user_id",
    )

    item_feat_ids = load_primary_keys(
        ITEM_FEAT_DIR,
        "item_id",
    )

    scan_seq(
        user_feat_ids,
        item_feat_ids,
    )

    print("\n" + "=" * 70)
    print("阶段 1.4A 完成。")


if __name__ == "__main__":
    main()