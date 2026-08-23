# Stage 5 运行说明

Stage 5 的六个入口统一位于 `scripts/stage5/`。Two-Tower、训练和 embedding 导出仅使用 PyTorch；FAISS 是 CPU HNSW Inner Product 索引。`--debug` 只把本阶段输出改到 `artifacts/stage5_debug/`，输入仍是正式完整的 Stage 3/4 合同。

先运行测试：

```bat
python -X utf8 -m unittest discover -s tests\stage5 -v
```

正式运行（首次生成不要加 `--overwrite`；确认要覆盖已有 Stage 5 产物时才加）：

```bat
python -X utf8 -u scripts\stage5\stage5_1_build_itemcf.py --config configs\stage5.yaml && python -X utf8 -u scripts\stage5\stage5_2_evaluate_itemcf.py --config configs\stage5.yaml && python -X utf8 -u scripts\stage5\stage5_3_train_two_tower.py --config configs\stage5.yaml && python -X utf8 -u scripts\stage5\stage5_4_build_faiss_index.py --config configs\stage5.yaml && python -X utf8 -u scripts\stage5\stage5_5_evaluate_two_tower.py --config configs\stage5.yaml && python -X utf8 -u scripts\stage5\stage5_6_compare_baselines.py --config configs\stage5.yaml --run-unit-tests
```

调试运行时给每个入口增加 `--debug`。Stage 5.1 即使在 debug 下也会扫描完整候选合同，但只处理配置指定的前 1000 个用户；Stage 5.3/5.5 会限制样本数和 epoch。

## 目录职责

- `scripts/stage5/`：仅放六个可直接运行的流水线入口。
- `src/recall/`：ItemCF、流式 Dataset、负采样、FAISS 对齐、指标和 checkpoint 公共逻辑。
- `src/models/vanilla_two_tower.py`：唯一的深度模型定义。
- `tests/stage5/`：不依赖全量训练的小型确定性测试。

`--overwrite` 不是正常运行必需参数。它只用于明确重建已经存在的对应输出，保护脚本默认不会静默覆盖长时间任务的结果。
