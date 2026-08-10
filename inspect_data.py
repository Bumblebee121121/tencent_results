from pathlib import Path
import json

import pyarrow.dataset as ds


# 数据集根目录
DATA_DIR = Path("data") / "TencentGR-1M"

# 目前需要检查的五张 Parquet 表
TABLE_PATHS = {
    "seq": DATA_DIR / "seq",
    "item_feat": DATA_DIR / "item_feat",
    "user_feat": DATA_DIR / "user_feat",
    "candidate": DATA_DIR / "candidate",
    "mm_emb_81_32": DATA_DIR / "mm_emb" / "emb_81_32_parquet",
}


def compact_value(value, max_items=2):
    """
    压缩较长的列表或嵌套字段，防止终端输出过多内容。
    """

    if isinstance(value, list):
        return {
            "length": len(value),
            "preview": [
                compact_value(item, max_items)
                for item in value[:max_items]
            ],
        }

    if isinstance(value, dict):
        return {
            key: compact_value(val, max_items)
            for key, val in value.items()
        }

    return value


def inspect_table(name: str, path: Path) -> None:
    """
    输出指定 Parquet 目录的文件数量、Schema 和两条样例。
    """

    print("\n" + "=" * 80)
    print(f"数据表：{name}")
    print(f"路径：{path.resolve()}")

    if not path.exists():
        print("状态：目录不存在")
        return

    parquet_files = sorted(path.rglob("*.parquet"))

    print(f"Parquet 文件数量：{len(parquet_files)}")

    if not parquet_files:
        print("状态：没有找到 Parquet 文件")
        return

    # PyArrow 可以将一个包含多个 Parquet 文件的目录视为一张数据表
    dataset = ds.dataset(path, format="parquet")

    print("\nSchema：")
    print(dataset.schema)

    print("\n前两条样例：")

    # 只读取前两行，不把整张表载入内存
    sample_table = dataset.head(2)
    sample_rows = sample_table.to_pylist()

    for index, row in enumerate(sample_rows, start=1):
        compact_row = compact_value(row)

        print(f"\n样例 {index}：")
        print(
            json.dumps(
                compact_row,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )


def inspect_indexer() -> None:
    """
    当前只检查 indexer.pkl 是否存在及文件大小。
    暂时不执行 pickle.load。
    """

    indexer_path = DATA_DIR / "indexer.pkl"

    print("\n" + "=" * 80)
    print("文件：indexer.pkl")
    print(f"路径：{indexer_path.resolve()}")

    if not indexer_path.exists():
        print("状态：文件不存在")
        return

    size_mb = indexer_path.stat().st_size / (1024 ** 2)

    print("状态：存在")
    print(f"文件大小：{size_mb:.2f} MB")


def main() -> None:
    print(f"数据集根目录：{DATA_DIR.resolve()}")

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"没有找到数据目录：{DATA_DIR.resolve()}"
        )

    for name, path in TABLE_PATHS.items():
        inspect_table(name, path)

    inspect_indexer()

    print("\n" + "=" * 80)
    print("数据结构检查完成。")


if __name__ == "__main__":
    main()