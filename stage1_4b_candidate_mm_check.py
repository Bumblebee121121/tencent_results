from pathlib import Path
import pickle

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds


DATA_DIR = Path("data") / "TencentGR-1M"

CANDIDATE_DIR = DATA_DIR / "candidate"
MM_DIR = DATA_DIR / "mm_emb" / "emb_81_32_parquet"
INDEXER_PATH = DATA_DIR / "indexer.pkl"

BATCH_SIZE = 65536
EXPECTED_MM_DIM = 32


def to_numpy_int(array, fill_value=-1):
    """将 Arrow 整数数组转换为 NumPy int64。"""
    return pc.fill_null(
        array,
        fill_value,
    ).to_numpy(
        zero_copy_only=False,
    ).astype(
        np.int64,
        copy=False,
    )


def membership_mask(values, sorted_reference):
    """判断 values 是否存在于已排序的参考 ID 数组中。"""
    positions = np.searchsorted(sorted_reference, values)

    result = np.zeros(len(values), dtype=bool)
    valid_positions = positions < len(sorted_reference)

    result[valid_positions] = (
        sorted_reference[positions[valid_positions]]
        == values[valid_positions]
    )

    return result


def load_item_oid_reference():
    """
    从 indexer.pkl 中提取所有训练广告 OID。

    只保留一个排序后的 NumPy 数组，
    不额外创建巨大的反向 Python 字典。
    """
    print("加载 indexer.pkl……")

    with INDEXER_PATH.open("rb") as file:
        indexer = pickle.load(file)

    item_mapping = indexer["i"]

    item_oids = np.fromiter(
        item_mapping.keys(),
        dtype=np.int64,
        count=len(item_mapping),
    )

    item_oids.sort()

    print(f"indexer 物品映射数量：{len(item_oids)}")

    return item_oids


def scan_candidate(item_oid_reference):
    """检查 candidate 的主键、retrieval_id 和映射率。"""
    print("\n" + "=" * 70)
    print("检查 candidate")
    print(f"路径：{CANDIDATE_DIR.resolve()}")

    dataset = ds.dataset(
        CANDIDATE_DIR,
        format="parquet",
    )

    oid_parts = []
    retrieval_parts = []

    row_count = 0
    null_oid_count = 0
    null_retrieval_count = 0

    scanner = dataset.scanner(
        columns=["item_id", "retrieval_id"],
        batch_size=BATCH_SIZE,
    )

    for batch in scanner.to_batches():
        row_count += batch.num_rows

        oid_array = batch.column(0)
        retrieval_array = batch.column(1)

        null_oid_count += oid_array.null_count
        null_retrieval_count += retrieval_array.null_count

        oid_parts.append(to_numpy_int(oid_array))
        retrieval_parts.append(
            to_numpy_int(retrieval_array)
        )

    all_oids = np.concatenate(oid_parts)
    all_retrieval_ids = np.concatenate(retrieval_parts)

    valid_oids = all_oids[all_oids > 0]
    valid_retrieval_ids = all_retrieval_ids[
        all_retrieval_ids >= 0
    ]

    unique_oids = np.unique(valid_oids)
    unique_retrieval_ids = np.unique(
        valid_retrieval_ids
    )

    duplicate_oid_count = (
        len(valid_oids) - len(unique_oids)
    )

    duplicate_retrieval_count = (
        len(valid_retrieval_ids)
        - len(unique_retrieval_ids)
    )

    mapped_mask = membership_mask(
        valid_oids,
        item_oid_reference,
    )

    mapped_count = int(np.sum(mapped_mask))
    unmapped_count = len(valid_oids) - mapped_count

    mapping_rate = (
        mapped_count / len(valid_oids)
        if len(valid_oids) > 0
        else 0.0
    )

    retrieval_is_contiguous = (
        len(unique_retrieval_ids) == row_count
        and unique_retrieval_ids.min() == 0
        and unique_retrieval_ids.max() == row_count - 1
    )

    print(f"总行数：{row_count}")
    print(f"缺失 item_id 数：{null_oid_count}")
    print(f"重复 item_id 额外行数：{duplicate_oid_count}")

    print(
        f"缺失 retrieval_id 数："
        f"{null_retrieval_count}"
    )
    print(
        f"重复 retrieval_id 额外行数："
        f"{duplicate_retrieval_count}"
    )
    print(
        f"retrieval_id 是否为 0 到 N-1 连续编号："
        f"{retrieval_is_contiguous}"
    )

    print(f"\n能映射到训练 RID 的候选数：{mapped_count}")
    print(f"不能映射的候选数：{unmapped_count}")
    print(f"candidate OID 映射率：{mapping_rate:.6%}")

    if unmapped_count > 0:
        print("\n未映射候选 OID 样例：")
        print(valid_oids[~mapped_mask][:10])


def scan_mm_embedding(item_oid_reference):
    """检查多模态 OID、维度、有限性和全零向量。"""
    print("\n" + "=" * 70)
    print("检查 mm_emb_81_32")
    print(f"路径：{MM_DIR.resolve()}")

    dataset = ds.dataset(
        MM_DIR,
        format="parquet",
    )

    oid_parts = []

    row_count = 0
    null_oid_count = 0
    invalid_oid_count = 0
    null_embedding_count = 0
    wrong_dimension_count = 0
    non_finite_value_count = 0
    all_zero_vector_count = 0

    scanner = dataset.scanner(
        columns=["anonymous_cid", "emb"],
        batch_size=BATCH_SIZE,
    )

    for batch_number, batch in enumerate(
        scanner.to_batches(),
        start=1,
    ):
        row_count += batch.num_rows

        if batch_number % 10 == 0:
            print(
                f"已处理 {batch_number} 个批次，"
                f"累计向量数：{row_count}"
            )

        cid_array = batch.column(0)
        emb_array = batch.column(1)

        null_oid_count += cid_array.null_count
        null_embedding_count += emb_array.null_count

        # anonymous_cid 是数字字符串，转换成 int64 OID。
        try:
            cast_oids = pc.cast(
                cid_array,
                pa.int64(),
            )

            oid_values = to_numpy_int(cast_oids)

        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            # 只有出现非法字符串时才使用较慢的逐条处理。
            oid_values = np.full(
                batch.num_rows,
                -1,
                dtype=np.int64,
            )

            for index, value in enumerate(
                cid_array.to_pylist()
            ):
                if value is None:
                    continue

                try:
                    oid_values[index] = int(value)
                except (TypeError, ValueError):
                    pass

        invalid_oid_count += int(
            np.sum(oid_values <= 0)
        )

        oid_parts.append(oid_values)

        # 检查每个 Embedding 的长度。
        lengths = pc.fill_null(
            pc.list_value_length(emb_array),
            -1,
        ).to_numpy(
            zero_copy_only=False,
        )

        wrong_dimension_count += int(
            np.sum(
                (lengths >= 0)
                & (lengths != EXPECTED_MM_DIM)
            )
        )

        # 检查所有向量元素是否为有限数值。
        flat_values = emb_array.values.to_numpy(
            zero_copy_only=False,
        )

        non_finite_value_count += int(
            np.sum(~np.isfinite(flat_values))
        )

        # 正常情况下每个批次都能直接 reshape。
        if (
            emb_array.null_count == 0
            and np.all(lengths == EXPECTED_MM_DIM)
        ):
            vectors = flat_values.reshape(
                -1,
                EXPECTED_MM_DIM,
            )

            all_zero_vector_count += int(
                np.sum(
                    np.all(vectors == 0, axis=1)
                )
            )

        else:
            # 只在存在异常长度或空向量时逐条检查。
            for vector in emb_array.to_pylist():
                if (
                    vector is not None
                    and len(vector) == EXPECTED_MM_DIM
                    and all(value == 0 for value in vector)
                ):
                    all_zero_vector_count += 1

    all_oids = np.concatenate(oid_parts)
    valid_oids = all_oids[all_oids > 0]

    unique_oids = np.unique(valid_oids)

    duplicate_oid_count = (
        len(valid_oids) - len(unique_oids)
    )

    mapped_unique_mask = membership_mask(
        unique_oids,
        item_oid_reference,
    )

    mapped_unique_count = int(
        np.sum(mapped_unique_mask)
    )

    unmapped_unique_count = (
        len(unique_oids) - mapped_unique_count
    )

    mapping_rate = (
        mapped_unique_count / len(unique_oids)
        if len(unique_oids) > 0
        else 0.0
    )

    historical_item_coverage = (
        mapped_unique_count / len(item_oid_reference)
        if len(item_oid_reference) > 0
        else 0.0
    )

    print(f"\n总向量记录数：{row_count}")
    print(f"缺失 anonymous_cid 数：{null_oid_count}")
    print(f"非法 OID 数：{invalid_oid_count}")
    print(f"重复 OID 额外行数：{duplicate_oid_count}")

    print(f"\n缺失 Embedding 数：{null_embedding_count}")
    print(f"非 32 维向量数：{wrong_dimension_count}")
    print(f"NaN/Inf 元素数量：{non_finite_value_count}")
    print(f"全零向量数：{all_zero_vector_count}")

    print(
        f"\n能映射到训练 RID 的唯一 OID 数："
        f"{mapped_unique_count}"
    )
    print(
        f"不能映射的唯一 OID 数："
        f"{unmapped_unique_count}"
    )
    print(f"多模态 OID 映射率：{mapping_rate:.6%}")
    print(
        f"历史广告多模态覆盖率："
        f"{historical_item_coverage:.6%}"
    )

    if unmapped_unique_count > 0:
        print("\n未映射多模态 OID 样例：")
        print(unique_oids[~mapped_unique_mask][:10])


def main():
    print("阶段 1.4B：候选广告与多模态一致性检查")
    print(f"数据根目录：{DATA_DIR.resolve()}")

    required_paths = [
        CANDIDATE_DIR,
        MM_DIR,
        INDEXER_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"缺少必要路径：{path.resolve()}"
            )

    item_oid_reference = load_item_oid_reference()

    scan_candidate(item_oid_reference)
    scan_mm_embedding(item_oid_reference)

    print("\n" + "=" * 70)
    print("阶段 1.4B 完成。")


if __name__ == "__main__":
    main()