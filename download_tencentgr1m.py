#下载数据集的脚本
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download


# 增大网络请求超时时间，降低大文件下载时的超时概率
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")


REPO_ID = "TAAC2025/TencentGR-1M"

# 只下载项目需要的数据，禁止下载整个 137 GB 仓库
ALLOW_PATTERNS = [
    "seq/**",
    "item_feat/**",
    "user_feat/**",
    "candidate/**",
    "mm_emb/emb_81_32_parquet/**",
    "indexer.pkl",
    "README.md",
]

TARGET_DIR = Path("data") / "TencentGR-1M"


def download_dataset(max_retries: int = 3) -> None:
    """下载 TencentGR-1M 的最小可用数据子集。"""

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n开始下载，第 {attempt}/{max_retries} 次尝试")
            print(f"数据集：{REPO_ID}")
            print(f"保存位置：{TARGET_DIR.resolve()}")
            print("仅下载约 2.1 GB 的必要数据。\n")

            local_path = snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                revision="main",
                local_dir=TARGET_DIR,
                allow_patterns=ALLOW_PATTERNS,
                max_workers=4,
            )

            print("\n下载完成。")
            print(f"数据目录：{Path(local_path).resolve()}")
            return

        except KeyboardInterrupt:
            print("\n用户终止下载。")
            print("下次重新运行脚本即可继续检查和下载缺失文件。")
            sys.exit(1)

        except Exception as error:
            print(f"\n第 {attempt} 次下载失败：")
            print(repr(error))

            if attempt == max_retries:
                print("\n已达到最大重试次数。")
                print("检查网络后重新运行本脚本即可。")
                raise

            wait_seconds = attempt * 10
            print(f"{wait_seconds} 秒后重试……")
            time.sleep(wait_seconds)


if __name__ == "__main__":
    download_dataset()