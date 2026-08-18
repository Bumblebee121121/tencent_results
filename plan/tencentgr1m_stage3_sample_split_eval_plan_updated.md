# TencentGR-1M 判别式推荐项目
# Stage 3：样本构造、时间切分与评估协议 Plan（修正版，交给 Codex）

> **重要修正**：删除旧版 Stage 3 中的 `click -> preceding exposure` 同-item 归因假设。  
> TencentGR-1M 的 `seq` 中，每个历史交互本身就是一个 `(item_id, action_type, timestamp)` 序列元素；`action_type=0/1` 分别表示 exposure/click 反馈类型。  
> 因此离线 pseudo next-click 样本应直接从 **click-labeled interaction** 构造，而不是再向前寻找同 item 的 exposure 记录。

---

# 0. 当前项目任务不变

项目主任务仍然是：

> **根据用户目标广告曝光之前的历史行为，从候选广告池中预测用户下一次点击的广告。**

官方 TencentGR-1M 的真实比赛任务中，目标广告是 answer window 内首个点击对应的 target impression，公开历史严格截止在该 target exposure 之前。

但公开 `seq` 中并没有给出这个隐藏 Ground Truth target，因此为了建立离线 Train / Validation / Test，本项目在公开历史序列内部构造 **pseudo next-click targets**。

离线 proxy protocol：

> 对公开历史序列中的 `action_type=1` interaction，把该 interaction 的 item 当作 pseudo next-click target，只使用它严格之前的历史作为输入。

设：

\[
S_u=[x_1,\dots,x_L],\qquad x_p=(i_p,a_p,t_p)
\]

若：

\[
a_p=1
\]

则：

\[
y=i_p
\]

并定义历史：

\[
H_u^{(p)}=[x_k\mid t_k<t_p]
\]

必须保证：

\[
\max t_{\mathrm{history}}<t_{\mathrm{target}}
\]

即 target 自身以及同 timestamp 的事件均不进入 history。

---

# 1. 本次 Codex 修改目标

把旧协议：

```text
click
↓
寻找同 item 的最近 preceding exposure
↓
把那个 exposure 当 target exposure
↓
构造样本
```

改成：

```text
公开 seq
↓
找到 action_type == 1 的 interaction
↓
该 interaction 的 item 直接作为 pseudo next-click target
↓
history = 所有 timestamp < target_timestamp 的 interaction
↓
构造样本
```

**任务没有改变，改变的只是离线 pseudo sample 构造方式。**

---

# 2. Codex 工作边界

Codex：

1. 只修改 Stage 3 相关代码、测试、配置与运行说明；
2. 不重写、不重跑 Stage 1 / Stage 2；
3. 不训练模型；
4. 不运行 FAISS；
5. 不修改原始数据；
6. 不安装/升级依赖；
7. 不编造统计结果；
8. 修改完成后只给用户运行命令，不替用户执行全量 Stage 3。

Stage 1 / 2 已完成的 Schema、ID mapping、sequence behavior、time-gap、item long-tail、candidate seen/unseen、MM coverage、feature profile 结果继续保留。

---

# 3. 废弃旧 Attribution 逻辑

以下逻辑不再作为正式协议：

```text
click -> same-item preceding exposure attribution
```

因此正式 Stage 3 不再：

- 要求 click 前存在同 item 的 `action_type=0`；
- 统计 `attribution_coverage`；
- 统计 `attribution_gap`；
- 保存 `target_exposure_position` 与 `target_click_position` 两套位置；
- 使用 `nearest_preceding_exposure_is_used` 测试；
- 因找不到 preceding exposure 而丢弃 click target。

旧文件如：

```text
src/data/attribution.py
scripts/stage3/stage3_1_click_exposure_attribution.py
```

应从正式 pipeline 移除，可保留为 legacy/deprecated，但 Stage 3.2 不得继续依赖它们。

---

# 4. Stage 3.1：Click-target Audit

建议正式脚本：

```text
scripts/stage3/stage3_1_click_target_audit.py
```

建议核心模块：

```text
src/data/click_target.py
```

## 4.1 目的

检查公开 `seq` 中 `action_type=1` interaction 是否足以支持 pseudo next-click 样本构造。

这一步**不做 click→exposure 配对**。

## 4.2 必须统计

至少输出：

```text
processed_user_count
total_event_count
click_target_count
users_with_click_count
users_with_click_ratio
unknown_action_count
```

每用户 click target 数量分布：

```text
min / p25 / p50 / p75 / p90 / p95 / p99 / max / mean
```

每个 pseudo target 可用历史长度：

```text
history_length_before_target
```

同样统计：

```text
min / p25 / p50 / p75 / p90 / p95 / p99 / max / mean
```

另统计：

```text
empty_history_target_count
empty_history_target_ratio
same_timestamp_prefix_excluded_count
```

其中历史长度必须按：

```text
timestamp < target_timestamp
```

计算，而不是简单按 `target_position`。

输出：

```text
artifacts/stage3/click_target_audit/click_target_audit_report.json
```

Debug：

```text
artifacts/stage3_debug/click_target_audit/click_target_audit_report.json
```

---

# 5. Stage 3.2：构造 Pseudo Next-click Sample Index

正式脚本仍可使用：

```text
scripts/stage3/stage3_2_build_next_click_samples.py
```

但内部逻辑必须切换到 click-target prefix protocol。

## 5.1 样本规则

若第 \(p\) 个 interaction：

\[
a_p=1
\]

则：

```text
target_item = item_p
target_timestamp = timestamp_p
```

历史：

```text
所有 timestamp < target_timestamp 的事件
```

推荐使用 `bisect_left` 或等价逻辑：

```python
history_end_position = first_position_with_timestamp_ge(target_timestamp)
history = seq[:history_end_position]
```

必须通过：

```python
assert all(t < target_timestamp for t in history_timestamps)
```

## 5.2 禁止

不得：

- 再寻找 same-item preceding exposure；
- 要求 target item 已经出现在 history；
- 把 target interaction 本身放进 history；
- 把与 target 同 timestamp 的 interaction 放进 history；
- 使用 target 之后事件；
- 因不存在 preceding exposure 而丢弃 target。

## 5.3 空历史

如果：

```text
history_length == 0
```

默认不进入正式 Train / Val / Test，但必须统计数量与比例，不允许 silently drop。

---

# 6. Compact Sample Index 字段

删除旧字段：

```text
target_exposure_timestamp
target_click_timestamp
target_exposure_position
target_click_position
attribution_gap
```

正式 sample 建议：

```text
sample_id
user_id
target_item_rid
target_item_oid
target_timestamp
target_position
history_end_position
history_length
target_action_type
```

其中：

```text
target_action_type = 1
```

如需复现原序列位置，可额外保存：

```text
source_seq_position
```

---

# 7. OID / RID 规则保持不变

继续区分：

- OID：原始匿名广告 ID；
- RID：历史连续映射 ID；
- `candidate.retrieval_id`：candidate 局部编号。

pseudo target 来自 `seq.item_id`，首先得到 `target_item_rid`，再通过已有 inverse mapping 得到 `target_item_oid`。

不得修改 Stage 1 已确认的 ID 规则。

---

# 8. Stage 3.3：Temporal Split

主体逻辑保留，但时间字段统一改为：

```text
target_timestamp
```

不再使用：

```text
target_exposure_timestamp
```

按所有合法 pseudo targets 的 `target_timestamp` 做全局时间切分。

默认：

```yaml
train_ratio: 0.80
val_ratio: 0.10
test_ratio: 0.10
```

要求：

```text
相同 timestamp 不跨 split
```

并满足：

\[
\max T_{\mathrm{train}}<\min T_{\mathrm{val}}
\]

\[
\max T_{\mathrm{val}}<\min T_{\mathrm{test}}
\]

---

# 9. Train / Validation / Test Target 策略

## Train

允许每用户贡献多个合法 pseudo click targets。

```yaml
max_train_targets_per_user: null
```

## Validation / Test Primary

每个用户在对应 evaluation window 中保留**最早一个合法 pseudo target**：

```text
val_primary.parquet
test_primary.parquet
```

同时保留：

```text
val_all_targets.parquet
test_all_targets.parquet
```

---

# 10. Stage 3.4：Evaluation Candidate Pool

主体逻辑保持不变。

先统计：

```text
val target in official candidate ratio
test target in official candidate ratio
```

然后构造：

\[
C_{\mathrm{eval}}
=
C_{\mathrm{official}}
\cup
Y_{\mathrm{val}}
\cup
Y_{\mathrm{test}}
\]

并去重。

必须：

```python
assert all(val_target in C_eval)
assert all(test_target in C_eval)
```

后续所有召回模型共用同一 `C_eval`。

---

# 11. Stage 3.5：Train-only Item History Strength

主体逻辑保持不变。

定义：

\[
n_i^{train}
\]

为 Train cutoff 之前原始行为流中 item \(i\) 的总交互次数。

必须直接扫描原始 seq，并按：

```text
timestamp < train_cutoff
```

统计。

**禁止从 sample history 累加**，否则 raw event 会因出现在多个 prefix sample 中被重复计数。

---

# 12. Head / Mid / Tail / Unseen

仅根据 Train seen item 的：

\[
n_i^{train}
\]

计算：

\[
P50_{\mathrm{train}},\quad P90_{\mathrm{train}}
\]

定义：

\[
\mathrm{Unseen}: n_i^{train}=0
\]

\[
\mathrm{Tail}: 0<n_i^{train}\le P50_{\mathrm{train}}
\]

\[
\mathrm{Mid}: P50_{\mathrm{train}}<n_i^{train}\le P90_{\mathrm{train}}
\]

\[
\mathrm{Head}: n_i^{train}>P90_{\mathrm{train}}
\]

禁止硬编码 Stage 2 EDA 的 `2` / `23`。

必须重新生成：

```text
val_target_strength_distribution.csv
test_target_strength_distribution.csv
```

统计 Head / Mid / Tail / Unseen 的 target 数量与比例。

---

# 13. Stage 3.6：统一 Retrieval Evaluation Protocol

主体保持不变。

Primary evaluation 每用户一个 Ground Truth。

\[
\mathrm{Recall@K}
=
\frac{1}{N}
\sum_{u=1}^{N}
\mathbf{1}(i_u^*\in TopK_u)
\]

由于每用户一个 target：

\[
\mathrm{Recall@K}=\mathrm{HitRate@K}
\]

至少支持：

```text
Recall@10
Recall@50
Recall@100
Recall@500
```

若 target rank 为 \(r\le K\)：

\[
\mathrm{NDCG@K}=\frac{1}{\log_2(r+1)}
\]

否则为 0。

至少支持：

```text
NDCG@10
NDCG@50
NDCG@100
```

统一分组：

```text
Overall
Head
Mid
Tail
Unseen
```

---

# 15. Stage 3.7：Temporal Drift Audit（时间漂移检查）

建议新增脚本：

```text
scripts/stage3/stage3_7_temporal_drift_audit.py
```

## 14.1 目的

Stage 3.5 的全量结果已经能够给出 Validation / Test 中：

```text
Head / Mid / Tail / Unseen
```

的总体 target 分布。

但是，如果出现：

```text
Test 的 Unseen target ratio
明显高于
Validation 的 Unseen target ratio
```

不能立即把它解释成模型问题，也不能立即把它解释成数据 Bug。

Stage 3.7 的目的只是做一个 **sanity check**：

> 检查随着 target 时间逐渐远离 Train cutoff，Train-unseen target 的比例是否发生系统性变化，从而判断是否存在明显的 temporal item churn / item cold-start drift。

Stage 3.7 **不修改任何 Stage 3 主协议**，也不重新构造样本。

---

## 14.2 输入

优先直接读取 Stage 3 已有产物：

```text
artifacts/stage3/samples/val_primary.parquet
artifacts/stage3/samples/test_primary.parquet

artifacts/stage3/item_strength/item_train_counts.parquet
artifacts/stage3/item_strength/item_strength_thresholds.json

artifacts/stage3/splits/split_manifest.json
```

如果 `val_primary.parquet` / `test_primary.parquet` 已经带有：

```text
item_strength_group
```

则直接复用。

如果没有，则根据：

```text
target_item
+
item_train_counts
+
item_strength_thresholds
```

重新 join 出：

```text
Head / Mid / Tail / Unseen
```

但**不得重新计算 P50 / P90**。

Stage 3.7 必须继续使用 Stage 3.5 已固定的：

```text
Train-only P50_train
Train-only P90_train
```

---

## 14.3 时间轴定义

统一使用：

```text
target_timestamp
```

并读取 Stage 3.3 的 Train cutoff。

对每个 Validation / Test primary target 计算：

\[
\Delta t
=
t_{\mathrm{target}}
-
t_{\mathrm{train\ cutoff}}
\]

再转为：

```text
days_from_train_cutoff
```

时间单位转换必须复用项目已有 timestamp 解析逻辑，不得凭空假设原始 timestamp 一定是秒或毫秒。

建议同时保留：

```text
calendar_date
days_from_train_cutoff
split
```

---

## 14.4 Primary 统计

Stage 3.7 的主分析只使用：

```text
val_primary.parquet
test_primary.parquet
```

因为正式 Retrieval evaluation 本身采用 one-user-one-target primary protocol。

按自然日或等价的固定 24h 时间桶统计：

```text
date
days_from_train_cutoff
split

target_count

head_count
mid_count
tail_count
unseen_count

head_ratio
mid_ratio
tail_ratio
unseen_ratio
```

其中：

\[
unseen\_ratio_d
=
\frac{N_{\mathrm{Unseen},d}}
{N_{\mathrm{all},d}}
\]

并保证：

\[
head\_ratio
+
mid\_ratio
+
tail\_ratio
+
unseen\_ratio
\approx 1
\]

允许浮点误差。

---

## 14.5 Split-level Summary

另外输出 Validation / Test 的总体汇总，用于与 Stage 3.5 对账：

```text
split
target_count
head_count
mid_count
tail_count
unseen_count
head_ratio
mid_ratio
tail_ratio
unseen_ratio
```

必须验证 Stage 3.7 汇总出的：

```text
Validation Head/Mid/Tail/Unseen
Test Head/Mid/Tail/Unseen
```

与 Stage 3.5 的：

```text
val_target_strength_distribution.csv
test_target_strength_distribution.csv
```

完全一致。

如果不一致：

```text
fail loudly
```

不要继续生成“时间漂移结论”。

---

## 14.6 Drift Quantification

不要只画曲线，至少输出可复现的数值摘要。

建议同时保留 **Raw 全时间桶统计** 与 **Eligible 趋势统计**。

Raw 指标记录全部时间桶：

```text
first_day
last_day
first_day_unseen_ratio
last_day_unseen_ratio
raw_unseen_ratio_change_first_to_last

min_daily_unseen_ratio
max_daily_unseen_ratio
weighted_mean_unseen_ratio
```

但正式趋势解释不能直接使用极稀疏尾部桶。

定义 eligible bucket：

```text
target_count >= min_targets_per_bucket
```

默认：

```yaml
min_targets_per_bucket: 1000
```

并额外输出：

```text
trend_bucket_count

first_eligible_day
last_eligible_day

first_eligible_day_target_count
last_eligible_day_target_count

first_eligible_day_unseen_ratio
last_eligible_day_unseen_ratio

eligible_unseen_ratio_change_first_to_last
```

正式报告“Unseen 是否随时间上升”时，优先使用 eligible 指标。

另外计算：

```text
linear_slope_unseen_ratio_per_day
```

斜率也只能在 eligible buckets 上拟合。

线性斜率只是描述性统计，不作为显著性检验，也不能单独证明因果关系。

如果某一天样本过少，应在 CSV 中保留真实 `target_count`，不要偷偷删除。

正式趋势统计使用：

```yaml
min_targets_per_bucket: 1000
```

即只有：

```text
target_count >= 1000
```

的时间桶参与趋势拟合和“首尾趋势变化”统计。

所有时间桶仍然必须完整保存在 CSV 中，只通过：

```text
used_for_trend = true / false
```

标记是否进入趋势计算。

---

## 14.7 可选的辅助 all-target 分析

可选读取：

```text
val_all_targets.parquet
test_all_targets.parquet
```

重复上述统计，输出辅助结果：

```text
temporal_drift_all_targets.csv
```

但：

```text
Primary = one-user-one-target
Auxiliary = all-target
```

二者必须严格区分。

项目正式结论优先使用 Primary。

---

## 14.8 输出

建议：

```text
artifacts/stage3/temporal_drift/
├── temporal_drift_primary.csv
├── temporal_drift_split_summary.csv
└── temporal_drift_report.json
```

可选：

```text
artifacts/stage3/temporal_drift/
└── temporal_drift_all_targets.csv
```

Debug：

```text
artifacts/stage3_debug/temporal_drift/
```

### `temporal_drift_report.json` 至少包含

```text
protocol_version
train_cutoff
analysis_start_timestamp
analysis_end_timestamp

primary_target_count
validation_target_count
test_target_count

validation_unseen_ratio
test_unseen_ratio

first_day
last_day
first_day_unseen_ratio
last_day_unseen_ratio
raw_unseen_ratio_change_first_to_last

min_daily_unseen_ratio
max_daily_unseen_ratio
weighted_mean_unseen_ratio

min_targets_per_bucket
trend_bucket_count
first_eligible_day
last_eligible_day
first_eligible_day_target_count
last_eligible_day_target_count
first_eligible_day_unseen_ratio
last_eligible_day_unseen_ratio
eligible_unseen_ratio_change_first_to_last
linear_slope_unseen_ratio_per_day

stage3_5_distribution_consistency_passed
```

---

## 14.9 不允许的做法

Stage 3.7 禁止：

- 用 Validation / Test 数据重新定义 Head/Mid/Tail/Unseen；
- 重新计算 P50/P90 并改变 Stage 3.5 分组；
- 为了让曲线“更好看”而删除高 Unseen 的日期；
- 把 `Unseen` 解释成“整个 TencentGR-1M 从未出现的广告”；
- 把时间相关性直接解释成因果关系；
- 因为 Stage 3.7 结果不符合预期而修改 Stage 3.3 时间切分；
- 重跑或覆盖 Stage 3.1～3.6 正式产物。

这里的 `Unseen` 始终定义为：

\[
n_i^{train}=0
\]

即：

> **在 Train cutoff 之前未出现，而不是在整个数据集中从未出现。**

---

## 14.10 结果解释规则

正式判断趋势时，优先查看：

```text
eligible_unseen_ratio_change_first_to_last
linear_slope_unseen_ratio_per_day
```

以及 `temporal_drift_primary.csv` 中所有：

```text
used_for_trend = true
```

的时间桶。

如果观察到：

```text
days_from_train_cutoff ↑
        ↓
Unseen ratio 整体上升
```

可以表述为：

> **严格时间切分下，随着评估时间远离训练截止点，Train-unseen target 的占比增加，说明存在明显的 item temporal churn / cold-start drift。**

这将进一步支持后续验证：

```text
History-strength-aware ID / non-ID fusion
```

但仍然属于：

```text
EDA / data protocol evidence
```

而不是：

```text
model improvement result
```

如果趋势不明显，也不能强行下结论。

此时应保留 Stage 3.5 的总体 Head/Mid/Tail/Unseen 分布，并把 Stage 3.7 结论写成：

> Validation/Test 的总体 Unseen 差异存在，但按时间桶未表现出稳定单调趋势，因此暂不把差异完全归因于 temporal drift。

---

## 14.11 Stage 3.7 Tests

建议新增：

```text
tests/stage3/test_temporal_drift_audit.py
```

至少覆盖：

```text
test_reuses_train_only_strength_thresholds
test_daily_group_ratios_sum_to_one
test_split_summary_matches_stage3_5
test_days_from_train_cutoff_is_nonnegative
test_primary_and_all_target_outputs_are_separate
test_sparse_tail_buckets_are_excluded_from_trend
```

Toy data 中必须包含：

```text
Train-seen target
Train-unseen target
不同 target_timestamp
```

并手工验证每天的 Unseen ratio。

另外必须增加稀疏尾部测试，例如：

```text
Day 0: 10000 targets
Day 1: 5000 targets
Day 2: 3 targets
```

配置：

```yaml
min_targets_per_bucket: 1000
```

应验证：

```text
trend_bucket_count == 2
first_eligible_day == 0
last_eligible_day == 1
Day 2 used_for_trend == false
```

并保证：

```text
eligible_unseen_ratio_change_first_to_last
```

只使用 Day 0 与 Day 1，而不是 Day 2。

---

## 14.12 Stage 3.7 完成标准

- [ ] 不重新构造样本；
- [ ] 不重新切分 Train/Val/Test；
- [ ] 不重新计算 Head/Mid/Tail/Unseen 阈值；
- [ ] 使用 `target_timestamp` 与 Train cutoff 计算时间距离；
- [ ] 输出按天的 Head/Mid/Tail/Unseen target 分布；
- [ ] Stage 3.7 split 汇总与 Stage 3.5 完全一致；
- [ ] 输出 Unseen ratio 的时间趋势摘要；
- [ ] `min_targets_per_bucket=1000`；
- [ ] 稀疏尾部时间桶保留在 CSV，但 `used_for_trend=false`；
- [ ] eligible first/last 与 eligible change 指标已输出；
- [ ] slope 只使用 eligible buckets；
- [ ] Primary 与 all-target（如有）严格分离；
- [ ] 单元测试通过；
- [ ] 只做 sanity check，不把相关性写成因果结论。

---


## Stage 3.7 修正后的单独重跑命令

Stage 3.1～3.6 已经完成时，不需要重新运行它们。

只执行：

```bat
python -X utf8 -u scripts\stage3\stage3_7_temporal_drift_audit.py --include-all-targets --overwrite

python -X utf8 -m unittest discover -s tests\stage3 -v
```

然后检查：

```text
artifacts/stage3/temporal_drift/temporal_drift_primary.csv
artifacts/stage3/temporal_drift/temporal_drift_split_summary.csv
artifacts/stage3/temporal_drift/temporal_drift_report.json
```

重点确认：

```text
stage3_5_distribution_consistency_passed = true
min_targets_per_bucket = 1000
trend_bucket_count > 0
last_eligible_day 的 target_count >= 1000
```

# 15. 必须修改 Stage 3 Tests

旧测试：

```text
test_nearest_preceding_exposure_is_used
```

不再适用，必须删除/替换。

新增或替换为：

```text
test_click_labeled_interaction_becomes_target
test_non_click_interaction_is_not_target
test_history_stops_before_target_timestamp
test_same_timestamp_events_are_excluded
test_target_itself_is_not_in_history
test_empty_history_target_is_counted_and_skipped
```

继续保留：

```text
test_temporal_split
test_candidate_coverage
test_item_strength_leakage
test_id_mapping
test_retrieval_metrics
```

---

# 16. 必须覆盖的 Toy Example

```text
user u:

t=10, item=A, action=0
t=20, item=B, action=1
t=30, item=C, action=0
t=40, item=D, action=1
```

应该生成：

## Sample 1

```text
history = [A]
target = B
```

## Sample 2

```text
history = [A, B, C]
target = D
```

不得要求 B 或 D 前面存在 same-item exposure。

---

# 17. Timestamp Tie Test

```text
t=10, A, action=0
t=20, B, action=0
t=20, C, action=1
```

以 C 为 target 时：

```text
history = [A]
```

不能包含同 timestamp 的 B。

必须满足：

\[
t_{\mathrm{history}}<t_{\mathrm{target}}
\]

而不是：

\[
t_{\mathrm{history}}\le t_{\mathrm{target}}
\]

---

# 18. 文件命名建议

```text
scripts/stage3/
├── stage3_1_click_target_audit.py
├── stage3_2_build_next_click_samples.py
├── stage3_3_temporal_split.py
├── stage3_4_build_eval_candidates.py
├── stage3_5_build_item_strength.py
├── stage3_6_build_eval_protocol.py
└── stage3_7_temporal_drift_audit.py

src/data/
├── click_target.py
├── sample_builder.py
└── temporal_split.py
```

旧：

```text
attribution.py
stage3_1_click_exposure_attribution.py
```

不得继续进入正式 pipeline。

---

# 19. 输出目录

```text
artifacts/stage3/
├── click_target_audit/
│   └── click_target_audit_report.json
├── samples/
│   ├── sample_manifest.json
│   ├── train_samples.parquet
│   ├── val_primary.parquet
│   ├── test_primary.parquet
│   ├── val_all_targets.parquet
│   └── test_all_targets.parquet
├── splits/
│   └── split_manifest.json
├── candidates/
│   ├── eval_candidates.parquet
│   └── eval_candidate_manifest.json
├── item_strength/
│   ├── item_train_counts.parquet
│   ├── item_strength_thresholds.json
│   ├── val_target_strength_distribution.csv
│   └── test_target_strength_distribution.csv
├── evaluation/
│   └── evaluation_protocol.json
└── temporal_drift/
    ├── temporal_drift_primary.csv
    ├── temporal_drift_split_summary.csv
    └── temporal_drift_report.json
```

Debug：

```text
artifacts/stage3_debug/
```

---

# 20. 配置修正

`configs/stage3.yaml` 建议：

```yaml
data_root:
output_root:

protocol_version: click_target_prefix_v2

train_ratio: 0.80
val_ratio: 0.10
test_ratio: 0.10

max_train_targets_per_user: null
materialize_history: false
random_seed: 42

temporal_drift:
  min_targets_per_bucket: 1000
```

删除 attribution-window 等旧 attribution 配置。

---

# 21. Smoke 重新执行顺序

Codex 修改完成后只给命令，不执行。

用户重新：

```text
1. Stage 3 unit tests
        ↓
2. 3.1 click-target audit --max-users 1000
        ↓
3. 检查 click_target_count / history_length
        ↓
4. 3.2 build samples --max-users 1000
        ↓
5. 3.3 temporal split --debug
        ↓
6. 3.4 candidate pool --debug
        ↓
7. 3.5 item strength --max-users 1000
        ↓
8. 3.6 eval protocol --debug
        ↓
9. 3.7 temporal drift audit --debug
        ↓
10. 再跑 Stage 3 unit tests
```

Smoke 全部正常后才进入全量。

Stage 3.1 不再看 attribution coverage，而重点检查：

```text
processed_user_count
click_target_count
users_with_click_ratio
click_targets_per_user
empty_history_target_ratio
history_length_before_target
```

---

# 22. Stage 3 完成标准（修正版）

- [ ] 不再使用 click→preceding exposure attribution；
- [ ] `action_type=1` interaction 可直接成为 pseudo target；
- [ ] history 只包含 `timestamp < target_timestamp`；
- [ ] target 自身不进入 history；
- [ ] 同 timestamp interaction 不进入 history；
- [ ] Train / Val / Test 使用全局时间切分；
- [ ] same timestamp 不跨 split；
- [ ] Evaluation Candidate Pool 固定；
- [ ] Val/Test target candidate coverage = 100%；
- [ ] Head/Mid/Tail/Unseen 只用 Train 定义；
- [ ] Recall / NDCG 共用统一实现；
- [ ] 修正版 unit tests 全部通过；
- [ ] 1000-user smoke 结果合理；
- [ ] 全量结果由用户实际运行产生；
- [ ] Stage 3.7 时间漂移检查完成；
- [ ] Stage 3.7 汇总与 Stage 3.5 分组结果一致；
- [ ] 对 Test Unseen 上升是否伴随时间漂移给出数据化结论。

---

# 23. Stage 1 / Stage 2 不需要重做

本次错误只发生在旧版 Stage 3 的 pseudo target 构造假设。

Stage 1 / 2 一直按：

```text
interaction = (item_id, action_type, timestamp)
```

直接处理数据。

因此以下结果继续有效：

```text
Schema / ID mapping
sequence length
exposure/click composition
time gap
item long-tail
candidate history-seen/unseen
multimodal coverage
feature profile
```

若旧文档出现：

> “同一广告先产生 exposure token，之后再产生 click token”

应改为：

> **历史序列中的广告 interaction 带有 exposure/click 行为反馈标签。**

---

# 24. 最终任务表述

项目级：

> **根据用户历史行为，从候选广告池中预测用户下一次点击广告。**

官方任务语义：

> **历史严格截止在真实 target impression 的曝光之前。**

本项目离线 proxy：

> **从公开历史序列中选取 click-labeled interaction 作为 pseudo next-click target，并使用其严格之前的序列前缀作为历史。**

三者不要混淆。

---

# 25. 最终原则

修正后的 Stage 3 固定：

```text
Click-labeled Pseudo Target
        +
Strict Historical Prefix
        +
Global Temporal Split
        +
One Evaluation Candidate Pool
        +
Train-only Item Strength
        +
One Retrieval Evaluation Protocol
        +
One Temporal Drift Sanity Check
```

后续：

```text
ItemCF / I2I
Vanilla Two-Tower
Enhanced Two-Tower
Multi-channel Retrieval
Ranking
```

必须共同复用这一协议。
