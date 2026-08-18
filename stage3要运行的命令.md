# Stage 3（click_target_prefix_v2）运行命令

修正版 Stage 3 不再执行 `click -> preceding exposure` 归因。公开 `seq` 中每个
`action_type=1` interaction 直接作为 pseudo next-click target，历史严格定义为：

```text
history = seq 中所有 timestamp < target_timestamp 的 interaction
```

空历史 click target 会进入审计统计，但不会进入正式 Train / Validation / Test。

## 1. 进入项目与环境

```bat
D:
cd D:\AI-file\tecent
conda activate tencent_rec

if not exist logs\stage3 mkdir logs\stage3
```

旧 attribution 协议的产物不能与 `click_target_prefix_v2` 混用。脚本会检查上游
manifest 的 `protocol_version`。只有确认要重建时才追加 `--overwrite`。

## 2. 先运行单元测试

```bat
python -X utf8 -m unittest discover -s tests\stage3 -v
```

重点测试包括：

```text
click-labeled interaction 直接成为 target
action_type != 1 不成为 target
history timestamp 严格小于 target_timestamp
同 timestamp interaction 被排除
target 自身不进入 history
空历史 target 被计数并跳过
全局时间切分与 timestamp tie
Candidate coverage / Train-only strength / OID-RID / Recall-NDCG
```

测试通过后再逐步运行 smoke，不要一次性粘贴整条 pipeline。

## 3. Stage 3.1 Click-target Audit（1,000 用户）

```bat
python -X utf8 -u scripts\stage3\stage3_1_click_target_audit.py --max-users 1000
```

查看：

```text
artifacts\stage3_debug\click_target_audit\click_target_audit_report.json
```

重点检查：

```text
processed_user_count
total_event_count
click_target_count
users_with_click_count / users_with_click_ratio
unknown_action_count
click_targets_per_user
history_length_before_target
empty_history_target_count / empty_history_target_ratio
same_timestamp_prefix_excluded_count
```

本阶段不再产生或检查 `attribution_coverage`、`attribution_gap`。

## 4. Stage 3.2 构造样本（1,000 用户）

```bat
python -X utf8 -u scripts\stage3\stage3_2_build_next_click_samples.py --max-users 1000
```

查看：

```text
artifacts\stage3_debug\samples\sample_manifest.json
artifacts\stage3_debug\samples\all_samples.parquet
```

必须满足：

```text
sample_count = click_target_count - empty_history_target_count
history = seq[:history_end_position]
所有 history timestamp < target_timestamp
target_action_type = 1
```

## 5. Stage 3.3 全局时间切分（Debug）

```bat
python -X utf8 -u scripts\stage3\stage3_3_temporal_split.py --debug  --overwrite
```

查看：

```text
artifacts\stage3_debug\splits\split_manifest.json
```

确认 `sample_time_field` 为 `target_timestamp`，且：

```text
max(train target_timestamp) < min(validation target_timestamp)
max(validation target_timestamp) < min(test target_timestamp)
timestamp_ties_cross_splits = false
```

## 6. Stage 3.4 Evaluation Candidate Pool（Debug）

```bat
python -X utf8 -u scripts\stage3\stage3_4_build_eval_candidates.py --debug  --overwrite
```

查看：

```text
artifacts\stage3_debug\candidates\eval_candidate_manifest.json
```

确认 Validation / Test target 最终 coverage 均为 100%。

## 7. Stage 3.5 Train-only Item Strength（1,000 用户）

```bat
python -X utf8 -u scripts\stage3\stage3_5_build_item_strength.py --max-users 1000 --overwrite
```

查看：

```text
artifacts\stage3_debug\item_strength\item_strength_thresholds.json
artifacts\stage3_debug\item_strength\val_target_strength_distribution.csv
artifacts\stage3_debug\item_strength\test_target_strength_distribution.csv
```

Smoke 的 Head / Mid / Tail / Unseen 比例只用于验证代码链路，不能作为项目结论。

## 8. Stage 3.6 Evaluation Protocol（Debug）

```bat
python -X utf8 -u scripts\stage3\stage3_6_build_eval_protocol.py --debug --overwrite 
```

查看：

```text
artifacts\stage3_debug\evaluation\evaluation_protocol.json
```

确认 `protocol_version=click_target_prefix_v2`，并包含统一 Recall、HitRate、NDCG
以及 Overall / Head / Mid / Tail / Unseen 分组。

## 9. Smoke 完成后再次测试

```bat
python -X utf8 -m unittest discover -s tests\stage3 -v
```

全部正常后，才逐步运行正式数据。

## 10. 正式全量运行顺序

先运行审计：

```bat
python -X utf8 -u scripts\stage3\stage3_1_click_target_audit.py
```

检查：

```text
artifacts\stage3\click_target_audit\click_target_audit_report.json
```

确认 click target 数量、用户覆盖、空历史比例和历史长度分布合理后，再逐项运行：

```bat
python -X utf8 -u scripts\stage3\stage3_2_build_next_click_samples.py
python -X utf8 -u scripts\stage3\stage3_3_temporal_split.py
python -X utf8 -u scripts\stage3\stage3_4_build_eval_candidates.py
python -X utf8 -u scripts\stage3\stage3_5_build_item_strength.py
python -X utf8 -u scripts\stage3\stage3_6_build_eval_protocol.py
```

最后再次运行：

```bat
python -X utf8 -m unittest discover -s tests\stage3 -v
```

## 11. 已有产物处理

若提示：

```text
output already exists
```

确认确实需要重建后，对当前步骤追加：

```text
--overwrite
```

例如：

```bat
python -X utf8 -u scripts\stage3\stage3_3_temporal_split.py --debug --overwrite
```

不要混用 debug 与正式产物，也不要把旧 attribution 协议产物接入 v2 pipeline。
