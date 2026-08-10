from pathlib import Path
from collections.abc import Mapping
import pickle


INDEXER_PATH = Path("data") / "TencentGR-1M" / "indexer.pkl"


def preview_object(obj, name="root", depth=0, max_depth=2, max_items=5):
    """
    只打印对象结构和少量样例，避免输出整个 142 MB 的映射对象。
    """
    indent = "    " * depth

    print(f"{indent}{name}: type={type(obj).__name__}")

    if depth >= max_depth:
        try:
            print(f"{indent}size={len(obj)}")
        except TypeError:
            pass
        return

    if isinstance(obj, Mapping):
        print(f"{indent}size={len(obj)}")

        for index, (key, value) in enumerate(obj.items()):
            if index >= max_items:
                print(f"{indent}... 其余内容省略")
                break

            print(
                f"{indent}key[{index}] = "
                f"{repr(key)} ({type(key).__name__})"
            )

            preview_object(
                value,
                name=f"value[{index}]",
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
            )

    elif isinstance(obj, (list, tuple)):
        print(f"{indent}size={len(obj)}")

        for index, value in enumerate(obj[:max_items]):
            preview_object(
                value,
                name=f"item[{index}]",
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
            )

    else:
        representation = repr(obj)

        if len(representation) > 200:
            representation = representation[:200] + "..."

        print(f"{indent}value={representation}")


def main():
    if not INDEXER_PATH.exists():
        raise FileNotFoundError(
            f"未找到文件：{INDEXER_PATH.resolve()}"
        )

    size_mb = INDEXER_PATH.stat().st_size / (1024 ** 2)

    print(f"文件路径：{INDEXER_PATH.resolve()}")
    print(f"文件大小：{size_mb:.2f} MB")
    print("开始加载 indexer.pkl……")

    with INDEXER_PATH.open("rb") as file:
        indexer = pickle.load(file)

    print("\n加载成功。")
    print("=" * 80)

    preview_object(indexer)


if __name__ == "__main__":
    main()