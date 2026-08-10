from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds


DATA_DIR = Path("data") / "TencentGR-1M"
SEQ_DIR = DATA_DIR / "seq"

BATCH_SIZE = 8192
MAX_SAMPLES = 10


def to_numpy_int(array, fill_value=-1):
    """将 Arrow 整数列转换为 NumPy 数组。"""
    return pc.fill_null(
        array,
        fill_value,
    ).to_numpy(
        zero_copy_only=False
    ).astype(
        np.int64,
        copy=False,
    )


def main():
    print("阶段 1.4A 补充：action_type 缺失模式检查")
    print(f"数据目录：{SEQ_DIR.resolve()}")

    dataset = ds.dataset(
        SEQ_DIR,
        format="parquet",
    )

    total_users = 0
    total_events = 0
    total_null_actions = 0

    affected_users = 0

    null_at_first = 0
    null_at_middle = 0
    null_at_last = 0

    users_with_one_null_at_last = 0
    users_with_other_pattern = 0

    null_count_per_user = Counter()
    samples = []

    scanner = dataset.scanner(
        columns=["user_id", "seq"],
        batch_size=BATCH_SIZE,
    )

    for batch_number, batch in enumerate(
        scanner.to_batches(),
        start=1,
    ):
        user_ids = to_numpy_int(batch.column(0))
        seq_array = batch.column(1)

        offsets = seq_array.offsets.to_numpy(
            zero_copy_only=False
        )

        # 防止 Arrow 切片导致 offsets 不从 0 开始。
        local_offsets = offsets - offsets[0]
        seq_lengths = np.diff(local_offsets)

        events = pc.list_flatten(seq_array)
        action_array = events.field("action_type")

        null_mask = action_array.is_null().to_numpy(
            zero_copy_only=False
        )

        null_indices = np.flatnonzero(null_mask)

        total_users += batch.num_rows
        total_events += len(events)
        total_null_actions += len(null_indices)

        if len(null_indices) == 0:
            continue

        # 找到每个缺失行为属于批次中的哪名用户。
        row_indices = np.searchsorted(
            local_offsets[1:],
            null_indices,
            side="right",
        )

        # 该行为位于用户序列中的位置，从 0 开始。
        positions = (
            null_indices
            - local_offsets[row_indices]
        )

        affected_row_indices = np.unique(row_indices)
        affected_users += len(affected_row_indices)

        null_counts = np.bincount(
            row_indices,
            minlength=batch.num_rows,
        )

        last_mask = (
            positions
            == seq_lengths[row_indices] - 1
        )

        last_counts = np.bincount(
            row_indices[last_mask],
            minlength=batch.num_rows,
        )

        null_at_first += int(
            np.sum(positions == 0)
        )

        null_at_last += int(
            np.sum(last_mask)
        )

        null_at_middle += int(
            np.sum(
                (positions > 0)
                & ~last_mask
            )
        )

        affected_mask = null_counts > 0

        one_null_at_last_mask = (
            affected_mask
            & (null_counts == 1)
            & (last_counts == 1)
        )

        users_with_one_null_at_last += int(
            np.sum(one_null_at_last_mask)
        )

        users_with_other_pattern += int(
            np.sum(
                affected_mask
                & ~one_null_at_last_mask
            )
        )

        for count in null_counts[affected_mask]:
            null_count_per_user[int(count)] += 1

        # 保存少量样例，只记录 ID、序列长度和缺失位置。
        if len(samples) < MAX_SAMPLES:
            for row_index in affected_row_indices:
                row_positions = positions[
                    row_indices == row_index
                ]

                samples.append(
                    {
                        "user_id": int(
                            user_ids[row_index]
                        ),
                        "sequence_length": int(
                            seq_lengths[row_index]
                        ),
                        "null_positions_0_based": [
                            int(position)
                            for position in row_positions
                        ],
                    }
                )

                if len(samples) >= MAX_SAMPLES:
                    break

        if batch_number % 20 == 0:
            print(
                f"已处理 {batch_number} 个批次，"
                f"累计用户数：{total_users}"
            )

    null_ratio = (
        total_null_actions / total_events
        if total_events > 0
        else 0.0
    )

    print("\n" + "=" * 70)
    print("检查结果")

    print(f"总用户数：{total_users}")
    print(f"总行为数：{total_events}")
    print(f"缺失 action_type 数：{total_null_actions}")
    print(f"缺失比例：{null_ratio:.6%}")

    print(f"\n出现缺失行为的用户数：{affected_users}")

    print("\n缺失位置统计：")
    print(f"位于序列第一个位置：{null_at_first}")
    print(f"位于序列中间位置：{null_at_middle}")
    print(f"位于序列最后一个位置：{null_at_last}")

    print("\n用户级模式：")
    print(
        "恰好一个缺失且位于最后位置的用户数："
        f"{users_with_one_null_at_last}"
    )
    print(
        "其他缺失模式的用户数："
        f"{users_with_other_pattern}"
    )

    print("\n每名用户的缺失行为数量分布：")

    for count in sorted(null_count_per_user):
        print(
            f"每名用户缺失 {count} 次："
            f"{null_count_per_user[count]} 名用户"
        )

    print("\n缺失样例：")

    for sample in samples:
        print(sample)

    print("\n" + "=" * 70)
    print("action_type 缺失模式检查完成。")


if __name__ == "__main__":
    main()