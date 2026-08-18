# TencentGR-1M Stage 4 运行说明

Stage 4 固定消费 Stage 3 的 next-click、时间切分、候选池和 Train-only strength，
不会重新定义样本、cutoff 或候选池。所有入口位于 `scripts/stage4/`，配置为
`configs/stage4.yaml`。

在项目根目录激活环境：

```bat
conda activate tencent_rec
```

## 1. 单元测试

```bat
python -X utf8 -m unittest discover -s tests\stage4 -v
```

## 2. Stage 4.1 合同审计

正式审计默认完整扫描，不生成大型 store：

```bat
python -X utf8 scripts\stage4\stage4_1_feature_contract_audit.py --config configs\stage4.yaml
```

重点检查：

```text
artifacts\stage4\audits\feature_contract.json
artifacts\stage4\audits\candidate_item_feature_consistency.csv
artifacts\stage4\audits\multimodal_source_audit.json
```

candidate/item_feat 存在 mismatch 时脚本只报告，不会静默覆盖。正式 source precedence
固定为：official candidate 使用 `candidate.feature_value`；Stage 3 新增的 Val/Test
target 使用 `item_feat`。

## 3. Debug / Smoke

`--debug` 同时使用 `artifacts/stage3_debug` 并写入 `artifacts/stage4_debug`。
数据截断必须显式指定，避免仅因为 debug 就悄悄改变统计范围。下面的 10,000 item/MM
行和 10,000 candidate 只验证链路，不可作为正式统计结论：

```bat
python -X utf8 scripts\stage4\stage4_1_feature_contract_audit.py --config configs\stage4.yaml --debug --max-items 10000 --max-candidates 10000
python -X utf8 scripts\stage4\stage4_2_build_train_item_base.py --config configs\stage4.yaml --debug
python -X utf8 scripts\stage4\stage4_3_build_user_features.py --config configs\stage4.yaml --debug
python -X utf8 scripts\stage4\stage4_4_build_item_side_features.py --config configs\stage4.yaml --debug --max-items 10000 --max-candidates 10000
python -X utf8 scripts\stage4\stage4_5_build_multimodal_store.py --config configs\stage4.yaml --debug --max-items 10000 --max-candidates 10000
python -X utf8 scripts\stage4\stage4_6_build_sequence_store.py --config configs\stage4.yaml --debug
python -X utf8 scripts\stage4\stage4_8_feature_dataset_smoke.py --config configs\stage4.yaml --debug --run-unit-tests
```

若对应输出已存在且确认需要重建，仅对当前命令追加 `--overwrite`。

## 4. 正式全量构建

Stage 4.1 人工审计通过后，依次运行：

```bat
python -X utf8 scripts\stage4\stage4_2_build_train_item_base.py --config configs\stage4.yaml
python -X utf8 scripts\stage4\stage4_3_build_user_features.py --config configs\stage4.yaml
python -X utf8 scripts\stage4\stage4_4_build_item_side_features.py --config configs\stage4.yaml
python -X utf8 scripts\stage4\stage4_5_build_multimodal_store.py --config configs\stage4.yaml
python -X utf8 scripts\stage4\stage4_6_build_sequence_store.py --config configs\stage4.yaml
python -X utf8 scripts\stage4\stage4_8_feature_dataset_smoke.py --config configs\stage4.yaml --run-unit-tests
```

最后检查：

```text
artifacts\stage4\manifests\stage4_manifest.json
artifacts\stage4\manifests\feature_dataset_smoke.json
artifacts\stage4\audits\leakage_audit.json
```

正式 manifest 中应满足：

```text
stage3_count_consistency_passed = true
feature_dataset_smoke_passed = true
all_stage4_tests_passed = true
candidate_cold_start_used_as_model_feature = false
retrieval_id_used_as_model_item_id = false
materialize_history_per_sample = false
```

Stage 4 不执行模型训练、Negative Sampling、FAISS、MM projection 或 Fusion Gate。
