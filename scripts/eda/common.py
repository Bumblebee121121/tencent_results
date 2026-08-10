"""阶段 2 EDA 脚本共享的路径、统计、输出和多模态辅助函数。"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "TencentGR-1M"
SEQ_DIR = DATA_DIR / "seq"
USER_FEAT_DIR = DATA_DIR / "user_feat"
ITEM_FEAT_DIR = DATA_DIR / "item_feat"
CANDIDATE_DIR = DATA_DIR / "candidate"
MM_DIR = DATA_DIR / "mm_emb" / "emb_81_32_parquet"
INDEXER_PATH = DATA_DIR / "indexer.pkl"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "eda"
METRICS_DIR = ARTIFACT_DIR / "metrics"
TABLES_DIR = ARTIFACT_DIR / "tables"
FIGURES_DIR = ARTIFACT_DIR / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

SEQ_BATCH_SIZE = 8192
TABLE_BATCH_SIZE = 65536
EXPECTED_MM_DIM = 32
DEFAULT_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)


def ensure_output_dirs() -> None:
    """创建固定的阶段 2 输出目录。"""
    for path in (METRICS_DIR, TABLES_DIR, FIGURES_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def require_paths(paths: Iterable[Path]) -> None:
    """在扫描开始前检查所有必要输入。"""
    missing = [path.resolve() for path in paths if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"缺少必要输入：\n{joined}")


def require_columns(dataset: ds.Dataset, required: Sequence[str]) -> None:
    """检查数据集是否包含预期列。"""
    missing = [name for name in required if name not in dataset.schema.names]
    if missing:
        raise ValueError(f"数据 Schema 缺少字段：{missing}")


def to_numpy_int(array: pa.Array, fill_value: int = -1) -> np.ndarray:
    """将 Arrow 整数数组转成 int64，并显式填充空值。"""
    return (
        pc.fill_null(array, fill_value)
        .to_numpy(zero_copy_only=False)
        .astype(np.int64, copy=False)
    )


def local_list_offsets(array: pa.Array) -> tuple[np.ndarray, np.ndarray]:
    """返回从零开始的 ListArray offsets 和逐行长度。"""
    offsets = array.offsets.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    offsets = offsets - offsets[0]
    return offsets, np.diff(offsets)


def safe_ratio(numerator: float | np.ndarray, denominator: float | np.ndarray):
    """安全计算比例；分母为零时返回 NaN。"""
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    result = np.full(np.broadcast_shapes(numerator_array.shape, denominator_array.shape), np.nan)
    np.divide(numerator_array, denominator_array, out=result, where=denominator_array != 0)
    return float(result) if result.ndim == 0 else result


def quantile_summary(
    values: np.ndarray,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
) -> dict[str, float | int | None]:
    """为内存中的一维数组生成常用描述统计。"""
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.floating):
        array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "mean": None, **{quantile_name(q): None for q in quantiles}}

    result: dict[str, float | int | None] = {
        "count": int(array.size),
        "mean": float(np.mean(array, dtype=np.float64)),
    }
    computed = np.quantile(array, quantiles)
    for q, value in zip(quantiles, computed):
        result[quantile_name(q)] = float(value)
    return result


def quantile_name(q: float) -> str:
    """把 0.95 转成 p95，把 0 和 1 转成 min/max。"""
    if q == 0:
        return "min"
    if q == 1:
        return "max"
    return f"p{int(round(q * 100))}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(data: Mapping[str, Any], path: Path) -> None:
    """以 UTF-8 保存可复核的 JSON 指标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(_json_ready(data), file, ensure_ascii=False, indent=2)
        file.write("\n")


def save_csv(rows: Sequence[Mapping[str, Any]] | pd.DataFrame, path: Path) -> None:
    """保存小型报告表；大型中间结果应使用 Parquet。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_figure(path: Path) -> None:
    """统一保存并关闭当前 Matplotlib figure。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def membership_mask(values: np.ndarray, sorted_reference: np.ndarray) -> np.ndarray:
    """向量化判断 ID 是否存在于排序参考数组。"""
    values = np.asarray(values)
    positions = np.searchsorted(sorted_reference, values)
    result = np.zeros(values.size, dtype=bool)
    valid = positions < sorted_reference.size
    result[valid] = sorted_reference[positions[valid]] == values[valid]
    return result


def lookup_sorted_flags(
    values: np.ndarray,
    sorted_keys: np.ndarray,
    sorted_flags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 key 是否存在以及存在时对应的布尔标记。"""
    positions = np.searchsorted(sorted_keys, values)
    found = positions < sorted_keys.size
    found_indices = np.flatnonzero(found)
    found[found_indices] &= sorted_keys[positions[found_indices]] == values[found_indices]
    flags = np.zeros(values.size, dtype=bool)
    matched = np.flatnonzero(found)
    flags[matched] = sorted_flags[positions[matched]]
    return found, flags


def load_item_mapping() -> dict[int, int]:
    """加载唯一允许的 candidate/mm OID -> history RID 映射。"""
    require_paths([INDEXER_PATH])
    print("加载 indexer.pkl 的 item OID -> RID 映射……")
    with INDEXER_PATH.open("rb") as file:
        indexer = pickle.load(file)
    if not isinstance(indexer, Mapping) or "i" not in indexer:
        raise ValueError("indexer.pkl 缺少 indexer['i']")
    mapping = indexer["i"]
    if not isinstance(mapping, dict):
        mapping = dict(mapping)
    print(f"历史 item 映射数：{len(mapping):,}")
    return mapping


def sorted_mapping_oids(mapping: Mapping[int, int]) -> np.ndarray:
    """提取排序后的历史 OID，用于批量 seen/unseen 判断。"""
    values = np.fromiter(mapping.keys(), dtype=np.int64, count=len(mapping))
    values.sort()
    return values


def scan_mm_oid_validity(
    progress_every: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """扫描 32D 多模态表并返回排序 OID、有效向量标记及审计计数。"""
    require_paths([MM_DIR])
    dataset = ds.dataset(MM_DIR, format="parquet")
    require_columns(dataset, ["anonymous_cid", "emb"])

    oid_parts: list[np.ndarray] = []
    valid_parts: list[np.ndarray] = []
    row_count = 0
    null_embedding_count = 0
    wrong_dimension_count = 0
    invalid_oid_count = 0

    scanner = dataset.scanner(
        columns=["anonymous_cid", "emb"],
        batch_size=TABLE_BATCH_SIZE,
    )
    for batch_number, batch in enumerate(scanner.to_batches(), start=1):
        cid_array = batch.column(0)
        emb_array = batch.column(1)
        row_count += batch.num_rows

        try:
            oids = to_numpy_int(pc.cast(cid_array, pa.int64()))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            oids = np.full(batch.num_rows, -1, dtype=np.int64)
            for index, value in enumerate(cid_array.to_pylist()):
                try:
                    oids[index] = int(value)
                except (TypeError, ValueError):
                    pass

        lengths = to_numpy_int(pc.list_value_length(emb_array), fill_value=-1)
        valid = (oids > 0) & (lengths == EXPECTED_MM_DIM) & ~emb_array.is_null().to_numpy(
            zero_copy_only=False
        )
        invalid_oid_count += int(np.sum(oids <= 0))
        null_embedding_count += emb_array.null_count
        wrong_dimension_count += int(np.sum((lengths >= 0) & (lengths != EXPECTED_MM_DIM)))
        oid_parts.append(oids)
        valid_parts.append(valid)

        if batch_number % progress_every == 0:
            print(f"已处理 {batch_number} 个多模态批次，累计记录：{row_count:,}")

    all_oids = np.concatenate(oid_parts)
    all_valid = np.concatenate(valid_parts)
    keep = all_oids > 0
    all_oids = all_oids[keep]
    all_valid = all_valid[keep]
    order = np.argsort(all_oids, kind="mergesort")
    all_oids = all_oids[order]
    all_valid = all_valid[order]
    if all_oids.size > 1 and np.any(all_oids[1:] == all_oids[:-1]):
        raise ValueError("mm_emb 中存在重复 OID，无法安全构造唯一覆盖标记")

    audit = {
        "row_count": int(row_count),
        "invalid_oid_count": int(invalid_oid_count),
        "null_embedding_count": int(null_embedding_count),
        "wrong_dimension_count": int(wrong_dimension_count),
        "valid_embedding_count": int(np.sum(all_valid)),
    }
    return all_oids, all_valid, audit


class DiskBackedInt64:
    """顺序写入 int64 临时文件，并以内存映射方式做精确分位数和直方图。"""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("wb")
        self.count = 0
        self.minimum: int | None = None
        self.maximum: int | None = None

    def append(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.int64)
        if array.size == 0:
            return
        array.tofile(self._file)
        self.count += int(array.size)
        local_min = int(array.min())
        local_max = int(array.max())
        self.minimum = local_min if self.minimum is None else min(self.minimum, local_min)
        self.maximum = local_max if self.maximum is None else max(self.maximum, local_max)

    def _close_writer(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def exact_quantiles(self, quantiles: Sequence[float]) -> dict[str, float | None]:
        self._close_writer()
        if self.count == 0:
            return {quantile_name(q): None for q in quantiles}

        array = np.memmap(self.path, dtype=np.int64, mode="r+", shape=(self.count,))
        positions = [(self.count - 1) * q for q in quantiles]
        lower = [int(np.floor(position)) for position in positions]
        upper = [int(np.ceil(position)) for position in positions]
        kth = np.unique(np.asarray(lower + upper, dtype=np.int64))
        array.partition(kth)
        result: dict[str, float | None] = {}
        for q, position, low, high in zip(quantiles, positions, lower, upper):
            fraction = position - low
            value = float(array[low]) * (1.0 - fraction) + float(array[high]) * fraction
            result[quantile_name(q)] = value
        array.flush()
        del array
        return result

    def log1p_histogram(self, bins: int = 120) -> tuple[np.ndarray, np.ndarray]:
        """分块计算 log1p(value) 直方图；可在 partition 后调用。"""
        self._close_writer()
        if self.count == 0:
            return np.zeros(bins, dtype=np.int64), np.linspace(0, 1, bins + 1)
        max_log = float(np.log1p(max(self.maximum or 0, 1)))
        edges = np.linspace(0.0, max_log, bins + 1)
        counts = np.zeros(bins, dtype=np.int64)
        array = np.memmap(self.path, dtype=np.int64, mode="r", shape=(self.count,))
        chunk_size = 2_000_000
        for start in range(0, self.count, chunk_size):
            chunk = np.asarray(array[start : start + chunk_size])
            counts += np.histogram(np.log1p(chunk), bins=edges)[0]
        del array
        return counts, edges

    def cleanup(self) -> None:
        self._close_writer()
        if self.path.exists():
            self.path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()


def print_outputs(stage_name: str, paths: Sequence[Path]) -> None:
    print("\n" + "=" * 72)
    print(f"{stage_name} 完成")
    print("关键输出文件：")
    for path in paths:
        print(f"- {path.resolve()}")
