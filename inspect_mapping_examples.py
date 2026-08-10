from pathlib import Path
import pickle

import pyarrow.dataset as ds


DATA_DIR = Path("data") / "TencentGR-1M"
INDEXER_PATH = DATA_DIR / "indexer.pkl"


def first_row(path: Path) -> dict:
    """
    从一个 Parquet 数据集读取第一行。
    只读一行，不会加载完整数据。
    """
    dataset = ds.dataset(path, format="parquet")
    rows = dataset.head(1).to_pylist()

    if not rows:
        raise ValueError(f"数据表为空：{path}")

    return rows[0]


def main() -> None:
    print("加载 indexer.pkl……")

    with INDEXER_PATH.open("rb") as file:
        indexer = pickle.load(file)

    item_indexer = indexer["i"]
    user_indexer = indexer["u"]
    action_indexer = indexer["a"]
    feature_indexer = indexer["f"]

    print("\n" + "=" * 80)
    print("1. 顶层映射规模")

    print(f"物品映射数量：{len(item_indexer)}")
    print(f"用户映射数量：{len(user_indexer)}")
    print(f"行为映射数量：{len(action_indexer)}")
    print(f"特征命名空间数量：{len(feature_indexer)}")

    print("\n行为映射完整内容：")
    for raw_value, mapped_value in action_indexer.items():
        print(
            f"原始行为 {raw_value!r} "
            f"({type(raw_value).__name__})"
            f" -> 映射值 {mapped_value!r}"
        )

    print("\n" + "=" * 80)
    print("2. 所有特征命名空间")

    feature_names = sorted(
        feature_indexer.keys(),
        key=lambda value: str(value),
    )

    for feature_name in feature_names:
        mapping = feature_indexer[feature_name]

        print(
            f"特征 {feature_name!r}: "
            f"type={type(mapping).__name__}, "
            f"size={len(mapping)}"
        )

    print("\n" + "=" * 80)
    print("3. candidate OID -> RID")

    candidate_row = first_row(DATA_DIR / "candidate")
    candidate_oid = int(candidate_row["item_id"])
    candidate_rid = item_indexer.get(candidate_oid)

    print(f"candidate.item_id：{candidate_oid}")
    print(f"OID 类型：{type(candidate_oid).__name__}")
    print(f"映射后的 RID：{candidate_rid}")
    print(f"是否成功映射：{candidate_rid is not None}")

    print("\n" + "=" * 80)
    print("4. multimodal OID -> RID")

    mm_row = first_row(
        DATA_DIR / "mm_emb" / "emb_81_32_parquet"
    )

    mm_oid_string = mm_row["anonymous_cid"]
    mm_oid = int(mm_oid_string)
    mm_rid = item_indexer.get(mm_oid)

    print(f"anonymous_cid 原始值：{mm_oid_string!r}")
    print(f"原始类型：{type(mm_oid_string).__name__}")
    print(f"转换后的 OID：{mm_oid}")
    print(f"映射后的 RID：{mm_rid}")
    print(f"是否成功映射：{mm_rid is not None}")
    print(f"多模态向量维度：{len(mm_row['emb'])}")

    print("\n" + "=" * 80)
    print("5. seq 和 item_feat 使用的 ID")

    seq_row = first_row(DATA_DIR / "seq")
    seq_first_event = seq_row["seq"][0]
    seq_rid = seq_first_event["item_id"]

    item_feat_row = first_row(DATA_DIR / "item_feat")
    item_feat_rid = item_feat_row["item_id"]

    print(f"seq 第一条物品 RID：{seq_rid}")
    print(f"item_feat 第一条物品 RID：{item_feat_rid}")

    print("\n" + "=" * 80)
    print("6. 映射样例")

    for index, (oid, rid) in enumerate(item_indexer.items()):
        print(
            f"item sample {index + 1}: "
            f"OID={oid!r} -> RID={rid!r}"
        )

        if index >= 4:
            break

    for index, (raw_user_id, user_rid) in enumerate(
        user_indexer.items()
    ):
        print(
            f"user sample {index + 1}: "
            f"raw={raw_user_id!r} -> RID={user_rid!r}"
        )

        if index >= 4:
            break

    print("\n映射检查完成。")


if __name__ == "__main__":
    main()