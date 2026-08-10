# TencentGR-1M 阶段 2 运行命令

请在项目根目录 `D:\AI-file\tecent` 打开 VS Code 终端或 CMD，并激活项目环境：

```cmd
conda activate tencent_rec
```

## 第一步：现在只运行 2.1

```cmd
python -X utf8 -u scripts\eda\stage2_1_sequence_behavior.py > logs\stage2_1_sequence_behavior.log 2>&1
```

运行结束后，先检查：

```cmd
type logs\stage2_1_sequence_behavior.log
```

确认日志最后出现“阶段 2.1 完成”，并检查 `artifacts\eda` 中的 JSON、CSV 和 PNG。

## 后续命令：上一步验收后再逐个运行

```cmd
python -X utf8 -u scripts\eda\stage2_2_temporal_patterns.py > logs\stage2_2_temporal_patterns.log 2>&1
```

```cmd
python -X utf8 -u scripts\eda\stage2_3_item_long_tail.py > logs\stage2_3_item_long_tail.log 2>&1
```

```cmd
python -X utf8 -u scripts\eda\stage2_4_candidate_coldstart.py > logs\stage2_4_candidate_coldstart.log 2>&1
```

```cmd
python -X utf8 -u scripts\eda\stage2_5_multimodal_coverage.py > logs\stage2_5_multimodal_coverage.log 2>&1
```

```cmd
python -X utf8 -u scripts\eda\stage2_6_feature_profile.py > logs\stage2_6_feature_profile.log 2>&1
```

不要并行运行这些脚本。阶段 2.5 依赖阶段 2.3 生成的 item popularity Parquet。
