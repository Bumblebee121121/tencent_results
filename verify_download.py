#检查数据集是否下载完整的脚本
from pathlib import Path

DATA_DIR = Path("data") / "TencentGR-1M"

EXPECTED = {
    "seq": DATA_DIR / "seq",
    "item_feat": DATA_DIR / "item_feat",
    "user_feat": DATA_DIR / "user_feat",
    "candidate": DATA_DIR / "candidate",
    "mm_emb_81_32": DATA_DIR / "mm_emb" / "emb_81_32_parquet",
    "indexer.pkl": DATA_DIR / "indexer.pkl",
    "README.md": DATA_DIR / "README.md",
}


def directory_size(path: Path) -> int:
    """递归计算目录大小，单位为字节。"""
    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file()
    )


def main() -> None:
    print(f"检查目录：{DATA_DIR.resolve()}\n")

    missing = []

    for name, path in EXPECTED.items():
        exists = path.exists()
        status = "存在" if exists else "缺失"
        print(f"{name:<18} {status}")

        if not exists:
            missing.append(name)

    parquet_files = list(DATA_DIR.rglob("*.parquet"))
    total_bytes = directory_size(DATA_DIR)
    total_gb = total_bytes / (1024 ** 3)

    print(f"\nParquet 文件总数：{len(parquet_files)}")
    print(f"数据目录总大小：{total_gb:.2f} GB")

    forbidden_dirs = [
        "emb_82_1024_parquet",
        "emb_83_3584_parquet",
        "emb_84_4096_parquet",
        "emb_85_3584_parquet",
        "emb_86_3584_parquet",
    ]

    found_forbidden = [
        name
        for name in forbidden_dirs
        if any(path.name == name for path in DATA_DIR.rglob("*"))
    ]

    if found_forbidden:
        print("\n错误：发现了不应下载的高维多模态目录：")
        for name in found_forbidden:
            print(f"- {name}")

    if missing:
        print("\n下载不完整，缺失：")
        for name in missing:
            print(f"- {name}")
        raise SystemExit(1)

    if found_forbidden:
        raise SystemExit(1)

    print("\n基础目录校验通过。")


if __name__ == "__main__":
    main()