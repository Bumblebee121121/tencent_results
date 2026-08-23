# Stage 5 运行说明

Stage 5 的流水线入口和两个协议审计入口统一位于 `scripts/stage5/`。Two-Tower、训练和 embedding 导出仅使用 PyTorch；FAISS 是 CPU HNSW Inner Product 索引。`--debug` 只把本阶段输出改到 `artifacts/stage5_debug/`，输入仍是正式完整的 Stage 3/4 合同。

当前协议为 `stage5_recall_baseline_v2`：历史中的 PAD=0 和 UNK=1 都不参与 User Tower 平均池化；训练负样本池只由 `train_item_count > 0` 的全体 Train-Seen item 构成；Stage 3 没有禁止 repeated target，因此 ItemCF 和 Two-Tower 默认不会过滤历史 item。

如果忽略 PAD/UNK 后某个 Validation/Test 用户历史没有任何 Train-Seen item，该样本不会以零向量查询 FAISS；其 `target_rank` 记为空、继续保留在评估分母，并在 Two-Tower metrics 与 Stage 5 manifest 中报告数量和比例。

先运行测试：

```bat
python -X utf8 -m unittest discover -s tests\stage5 -v
```

正式运行（首次生成不要加 `--overwrite`；确认要覆盖已有 Stage 5 产物时才加）：

```bat
python -X utf8 -u scripts\stage5\stage5_0_audit_repeated_targets.py --config configs\stage5.yaml && python -X utf8 -u scripts\stage5\stage5_1_build_itemcf.py --config configs\stage5.yaml && python -X utf8 -u scripts\stage5\stage5_2_evaluate_itemcf.py --config configs\stage5.yaml && python -X utf8 -u scripts\stage5\stage5_3_train_two_tower.py --config configs\stage5.yaml && python -X utf8 -u scripts\stage5\stage5_4_build_faiss_index.py --config configs\stage5.yaml && python -X utf8 -u scripts\stage5\stage5_4a_audit_hnsw_accuracy.py --config configs\stage5.yaml --fail-on-low-recall && python -X utf8 -u scripts\stage5\stage5_5_evaluate_two_tower.py --config configs\stage5.yaml && python -X utf8 -u scripts\stage5\stage5_6_compare_baselines.py --config configs\stage5.yaml --run-unit-tests
```

调试运行时给每个入口增加 `--debug`。Stage 5.0 会对每个 split 最多审计 10000 条样本；Stage 5.1 即使在 debug 下也会扫描完整候选合同，但只处理配置指定的前 1000 个用户；Stage 5.3/5.5 会限制样本数和 epoch；Stage 5.4a 默认抽取 100 个用户向量。

Stage 5.4a 使用相同 item embedding 分别执行 HNSW 和精确 `IndexFlatIP`，报告 Top-K 集合交集 Recall。低于 `configs/stage5.yaml` 中阈值时，`--fail-on-low-recall` 会在保留审计 JSON 后终止流水线，避免把 ANN 误差当成模型误差。

## 目录职责

- `scripts/stage5/`：放六个主流水线入口以及 Stage 5.0、Stage 5.4a 两个审计入口。
- `src/recall/`：ItemCF、流式 Dataset、负采样、FAISS 对齐、指标和 checkpoint 公共逻辑。
- `src/models/vanilla_two_tower.py`：唯一的深度模型定义。
- `tests/stage5/`：不依赖全量训练的小型确定性测试。

`--overwrite` 不是正常运行必需参数。它只用于明确重建已经存在的对应输出，保护脚本默认不会静默覆盖长时间任务的结果。
