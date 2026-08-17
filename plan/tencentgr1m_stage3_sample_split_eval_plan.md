# TencentGR-1M 判别式推荐项目
# Stage 3：样本构造、时间切分与评估协议 Plan（交给 Codex）

> 项目任务：基于 TencentGR-1M 构建判别式推荐系统。  
> 本阶段只负责建立统一、严格、可复现的样本定义、时间切分与评估协议，不训练模型。  
> 后续 ItemCF / I2I、Vanilla Two-Tower、Enhanced Two-Tower、多路召回与排序模型都必须复用本阶段产物。

---

# 0. Stage 3 的核心目标

Stage 3 必须固定四件事：

1. **预测时点是什么**；
2. **历史行为 \(H_u\) 截止到哪里**；
3. **Ground Truth 下一点击广告是什么**；
4. **Train / Validation / Test 与评估 Candidate Pool 如何统一构造**。

本项目正式采用的离线主任务定义为：

> **对于一个目标点击广告，找到触发该点击的目标曝光，以目标曝光时刻 \(T\) 作为预测时点；仅使用 \(T\) 之前的用户历史行为 \(H_u(T)\)，从统一候选广告池中预测该目标广告。**

形式化定义：

\[
H_u(T)
=
\left[
(i_1,a_1,t_1),
(i_2,a_2,t_2),
\dots,
(i_L,a_L,t_L)
\right],
\qquad t_k<T
\]

Ground Truth 为：

\[
y_u=i^*
\]

其中 \(i^*\) 是目标曝光对应、随后被用户点击的广告。

必须满足：

\[
\max_{k} t_k < T
\]

目标曝光本身、目标点击本身，以及 \(T\) 之后发生的任何事件均不得进入历史。

---

# 1. Codex 工作边界

Codex 在本阶段：

1. 可以阅读已有 Stage 1、Stage 2 脚本、日志、Markdown、Schema 和 artifacts；
2. 编写 Stage 3 所需 Python、配置和测试代码；
3. 只给出运行命令，不替用户执行全量数据任务；
4. 不训练任何推荐模型；
5. 不运行 FAISS；
6. 不修改 `data/TencentGR-1M/` 下的原始数据；
7. 不安装或升级 Python 包；
8. 不伪造样本数量、切分点、分组比例和指标；
9. 不偷偷修改 Stage 1/2 已确认的 OID/RID 规则；
10. 所有真实统计必须由用户实际运行脚本后生成。

---

# 2. Stage 3 总流程

```text
原始用户行为序列
        ↓
识别 click 事件
        ↓
为 click 寻找同 item 的最近前序 exposure
        ↓
得到 target exposure time T
        ↓
仅保留 timestamp < T 的历史行为
        ↓
构造 next-click pseudo sample
        ↓
按 target exposure time 做全局时间切分
        ↓
Train / Validation / Test
        ↓
构造统一 Evaluation Candidate Pool
        ↓
仅用 Train 统计 Item History Strength
        ↓
Head / Mid / Tail / Unseen
        ↓
固定 Recall@K / NDCG@K 等统一评估协议
```

---

# 3. Stage 3.1：Click → Exposure Attribution Audit

建议脚本：

```text
scripts/stage3/stage3_1_click_exposure_attribution.py
```

## 3.1 目的

TencentGR-1M 中同一广告可能先曝光、随后点击。

如果直接采用：

```text
点击前的所有行为 → 下一点击广告
```

那么目标广告自己的 exposure 很可能已经出现在历史中，形成：

```text
...
target item exposure
target item click
```

此时模型可能只需要复制最近曝光广告，导致任务严重变简单。

因此主协议要求：

> 找到目标 click 对应的 preceding exposure，并把预测时点放在该 exposure 之前。

## 3.2 第一版 Attribution Rule

对于用户 \(u\) 的点击事件：

\[
(i^*,\mathrm{click},t_c)
\]

向前寻找：

> 同一用户、同一 item、满足 \(t_e \le t_c\) 的最近一次 exposure。

得到：

\[
T=t_e
\]

如果不存在对应 exposure：

- 不强制配对；
- 不生成正式 target；
- 记录 attribution failure。

不要自行设置 30 分钟、1 小时、1 天等 attribution window。

先统计真实：

\[
t_c-t_e
\]

分布，再决定未来是否需要 attribution window。

## 3.3 必须统计

输出：

```text
artifacts/stage3/attribution/attribution_report.json
```

至少包含：

- click 总数；
- 成功 attribution 数；
- attribution failure 数；
- attribution coverage；
- exposure → click gap 的 min / median / P90 / P95 / P99 / max；
- 同一 item 存在多个 preceding exposure 的比例；
- exposure 与 click timestamp 相同的比例。

---

# 4. Stage 3.2：构造 Next-click Sample Index

建议脚本：

```text
scripts/stage3/stage3_2_build_next_click_samples.py
```

## 4.1 样本定义

对于成功 attribution 的目标：

```text
target_item = i*
target_exposure_time = T
target_click_time = Tc
```

历史：

\[
H_u(T)
=
\{(i_k,a_k,t_k)\mid t_k<T\}
\]

必须：

\[
\max t_k<T
\]

不能使用：

- target exposure；
- target click；
- target exposure 之后但 target click 之前的任何事件；
- target click 之后的未来行为。

## 4.2 Compact Sample Index

优先只保存索引，不为每个样本复制完整历史。

字段建议：

```text
sample_id
user_id
target_item_rid
target_item_oid
target_exposure_timestamp
target_click_timestamp
target_exposure_position
target_click_position
history_end_position
history_length
attribution_gap
```

后续 Dataset 通过：

```python
history = seq[:history_end_position]
```

动态读取历史。

配置：

```yaml
materialize_history: false
```

默认 `false`。

---

# 5. ID 规范

必须同时区分：

- OID：原始匿名广告 ID；
- RID：历史连续映射 ID；
- `candidate.retrieval_id`：candidate 局部检索编号。

Target sample 同时保留：

```text
target_item_oid
target_item_rid
```

允许 history-unseen candidate：

```text
rid = null
```

禁止伪造 historical RID。

---

# 6. Action 处理

历史 Action：

```text
0 = exposure
1 = click
None = unknown
```

本阶段：

- `None` 不替换成 exposure；
- `None` 历史事件不删除；
- 保留其 item 和 timestamp；
- 只有明确 `action_type == 1` 的事件可生成 click target。

Action-aware 编码属于后续特征/模型阶段。

---

# 7. Stage 3.3：全局时间切分

建议脚本：

```text
scripts/stage3/stage3_3_temporal_split.py
```

禁止正式实验使用：

```python
train_test_split(..., shuffle=True)
```

或随机按：

- 行；
- sample；
- 用户；

做主切分。

## 7.1 Sample 时间

使用：

```text
target_exposure_timestamp
```

作为每个样本的时间。

默认：

```yaml
train_ratio: 0.80
val_ratio: 0.10
test_ratio: 0.10
```

按所有合法 target 的 `target_exposure_timestamp` 全局排序后确定两个 cutoff。

不要硬编码绝对时间。

## 7.2 Timestamp Tie

同一个 timestamp 不能被拆进两个 split。

如果 ratio cutoff 落在相同 timestamp 内，移动 cutoff 到 timestamp boundary。

应满足：

\[
\max T_{\mathrm{train}}
<
\min T_{\mathrm{val}}
\]

以及：

\[
\max T_{\mathrm{val}}
<
\min T_{\mathrm{test}}
\]

## 7.3 同一用户能否跨 Split

允许。

同一用户可以：

```text
较早 target → Train
中间 target → Validation
较晚 target → Test
```

因为本项目验证的是时间泛化，不是 cold-user 泛化。

---

# 8. Train 与 Evaluation Target 策略

## 8.1 Train

允许每个用户贡献多个合法 target。

配置：

```yaml
max_train_targets_per_user: null
```

默认不限。

## 8.2 Validation / Test Primary Protocol

Primary evaluation 建议：

> 每个用户在对应 evaluation window 中仅保留最早一个合法 target。

原因：

- 更接近 one-user-one-target 检索协议；
- 避免高活跃用户过度主导总体指标；
- 便于 Recall/HR 解释。

同时保存：

```text
val_all_targets.parquet
test_all_targets.parquet
```

用于辅助分析。

正式主指标默认使用：

```text
val_primary.parquet
test_primary.parquet
```

---

# 9. Stage 3.4：Evaluation Candidate Pool

建议脚本：

```text
scripts/stage3/stage3_4_build_eval_candidates.py
```

先统计：

```text
Validation target ∈ official candidate 的比例
Test target ∈ official candidate 的比例
```

由于当前是从历史 seq 自行构造 pseudo targets，不假设覆盖率必然为 100%。

正式 evaluation candidate：

\[
C_{\mathrm{eval}}
=
C_{\mathrm{official}}
\cup
Y_{\mathrm{val}}
\cup
Y_{\mathrm{test}}
\]

然后去重。

必须保证：

```python
assert all(val_target in C_eval)
assert all(test_target in C_eval)
```

禁止简单删除“不在 official candidate 中”的真实 target。

输出：

```text
artifacts/stage3/candidates/eval_candidates.parquet
artifacts/stage3/candidates/eval_candidate_manifest.json
```

记录：

```text
official_candidate_count
added_validation_target_count
added_test_target_count
final_candidate_count
```

---

# 10. Stage 3.5：Train-only Item History Strength

建议脚本：

```text
scripts/stage3/stage3_5_build_item_strength.py
```

定义：

\[
n_i^{train}
\]

为 Train 时间范围内原始行为流中 item \(i\) 出现的事件总次数。

统计必须直接来自：

```text
raw events with timestamp < train cutoff
```

不能从重复展开的 sample history 中累计，否则同一个事件可能被重复计数。

## 10.1 Head / Mid / Tail / Unseen

仅对 Train 中 seen item 的 \(n_i^{train}\) 计算：

\[
P50_{\mathrm{train}},\qquad
P90_{\mathrm{train}}
\]

定义：

\[
\mathrm{Unseen}:
n_i^{train}=0
\]

\[
\mathrm{Tail}:
0<n_i^{train}\le P50_{\mathrm{train}}
\]

\[
\mathrm{Mid}:
P50_{\mathrm{train}}
<n_i^{train}
\le
P90_{\mathrm{train}}
\]

\[
\mathrm{Head}:
n_i^{train}>P90_{\mathrm{train}}
\]

禁止直接硬编码 Stage 2 全量 EDA 得到的 `2`、`23`。

正式阈值必须重新只根据 Train 计算。

---

# 11. Validation / Test Target Strength Distribution

必须输出：

```text
Head target count / ratio
Mid target count / ratio
Tail target count / ratio
Unseen target count / ratio
```

分别针对 Validation 与 Test。

文件：

```text
artifacts/stage3/item_strength/val_target_strength_distribution.csv
artifacts/stage3/item_strength/test_target_strength_distribution.csv
```

这一结果用于判断：

> Tail / Unseen 是否真的是 next-click 主任务的重要性能瓶颈，而不只是 candidate pool 中数量很多。

---

# 12. Stage 3.6：统一 Retrieval Evaluation Protocol

实现：

```text
src/evaluation/retrieval_metrics.py
```

所有后续召回模型必须调用同一套 metrics。

## 12.1 Recall@K / HitRate@K

Primary evaluation 每用户一个 Ground Truth：

\[
\mathrm{Recall@K}
=
\frac{1}{N}
\sum_{u=1}^{N}
\mathbf{1}
\left(
i_u^*\in TopK_u
\right)
\]

此时：

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

## 12.2 NDCG@K

若 target rank 为 \(r\le K\)：

\[
\mathrm{NDCG@K}
=
\frac{1}{\log_2(r+1)}
\]

否则：

\[
\mathrm{NDCG@K}=0
\]

至少支持：

```text
NDCG@10
NDCG@50
NDCG@100
```

---

# 13. 分组指标

统一支持：

```text
Overall
Head
Mid
Tail
Unseen
```

例如：

```text
Overall Recall@100
Head Recall@100
Mid Recall@100
Tail Recall@100
Unseen Recall@100
```

分组标签必须来自 Stage 3 的 Train-only item strength。

模型不得各自重新定义。

---

# 14. 为未来多路召回预留接口

Stage 3 不实际执行多路融合，但 metrics API 预留：

```text
Channel intersection
Channel union
Jaccard / overlap
Incremental hits
Incremental Recall
```

未来比较：

```text
ItemCF / I2I
Enhanced Two-Tower
```

时使用。

---

# 15. Stage 3.7：Data Leakage Tests

新增：

```text
tests/stage3/
├── test_target_history_leakage.py
├── test_temporal_split.py
├── test_candidate_coverage.py
├── test_item_strength_leakage.py
├── test_id_mapping.py
└── test_retrieval_metrics.py
```

## 15.1 Target History Leakage

必须验证：

```python
assert all(t < target_exposure_timestamp for t in history_timestamps)
```

并保证：

- target exposure 不在 history；
- target click 不在 history；
- target exposure 之后的事件不在 history。

## 15.2 Temporal Split

验证：

```text
Train < Val < Test
```

且同一 timestamp 不跨 split。

## 15.3 Candidate Coverage

必须：

```text
100% validation target ∈ C_eval
100% test target ∈ C_eval
```

## 15.4 Item Strength Leakage

Synthetic example：

```text
Train:
A × 2

Validation:
A × 100
```

必须：

```text
n_A_train = 2
```

不能为 `102`。

## 15.5 ID Mapping

历史 item：

```text
OID → RID → OID
```

应保持一致。

history-unseen candidate：

```text
RID = null
```

合法。

## 15.6 Retrieval Metrics

构造 toy ranking，手工验证 Recall@K / NDCG@K。

---

# 16. 推荐目录

```text
scripts/
└── stage3/
    ├── stage3_1_click_exposure_attribution.py
    ├── stage3_2_build_next_click_samples.py
    ├── stage3_3_temporal_split.py
    ├── stage3_4_build_eval_candidates.py
    ├── stage3_5_build_item_strength.py
    └── stage3_6_build_eval_protocol.py

src/
├── data/
│   ├── attribution.py
│   ├── sample_builder.py
│   └── temporal_split.py
└── evaluation/
    └── retrieval_metrics.py

configs/
└── stage3.yaml

tests/
└── stage3/
    ├── test_target_history_leakage.py
    ├── test_temporal_split.py
    ├── test_candidate_coverage.py
    ├── test_item_strength_leakage.py
    ├── test_id_mapping.py
    └── test_retrieval_metrics.py
```

优先复用仓库已有 utils，不重复造轮子。

---

# 17. 输出目录

```text
artifacts/stage3/
├── attribution/
│   └── attribution_report.json
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
└── evaluation/
    └── evaluation_protocol.json
```

日志：

```text
logs/stage3/
```

---

# 18. 工程要求

大表处理优先：

- PyArrow Dataset；
- Parquet batch scan；
- NumPy；
- streaming aggregation。

禁止一次性将 9000 万级事件全部加载到 pandas。

所有脚本必须：

1. `main()`；
2. CLI 参数；
3. 路径检查；
4. progress logging；
5. fixed output path；
6. overwrite protection；
7. deterministic behavior；
8. fail loudly；
9. 支持 debug/smoke 模式。

配置示例：

```yaml
data_root:
output_root:

train_ratio: 0.80
val_ratio: 0.10
test_ratio: 0.10

max_train_targets_per_user: null
materialize_history: false
random_seed: 42
```

路径不能硬编码某个 Windows 绝对路径。

---

# 19. Debug / Smoke 模式

所有 Stage 3 脚本支持：

```text
--max-users
```

或：

```text
--debug
```

Debug 输出必须进入：

```text
artifacts/stage3_debug/
```

禁止覆盖正式 Stage 3 产物。

---

# 20. Stage 3 不固定负样本

本阶段只固定：

```text
positive next-click target
```

不生成永久：

```text
train_negative_samples.parquet
```

因为后续：

- ItemCF 不需要训练负样本；
- Two-Tower 可使用 in-batch negative；
- Deep Retrieval 可使用 sampled/hard negative；
- Ranking 有独立的负样本协议。

负采样属于后续模型训练协议。

---

# 21. 用户实际执行顺序

Codex 写完代码后，不替用户运行。

建议用户逐步执行：

```text
3.1 Attribution Audit
        ↓
检查 attribution coverage
        ↓
3.2 Build Next-click Sample Index
        ↓
检查样本合法性
        ↓
3.3 Temporal Split
        ↓
检查 split manifest
        ↓
3.4 Build Evaluation Candidate Pool
        ↓
检查 target coverage
        ↓
3.5 Build Train-only Item Strength
        ↓
重点检查 Val/Test Head/Mid/Tail/Unseen 分布
        ↓
3.6 Build Evaluation Protocol
        ↓
运行 Stage 3 tests
```

不要一次性执行整个全量 pipeline。

---

# 22. Stage 3 完成标准

- [ ] Click → Exposure attribution 代码完成；
- [ ] attribution coverage 被真实统计；
- [ ] next-click sample index 构造完成；
- [ ] history 严格截止于 target exposure 之前；
- [ ] target exposure / target click 不进入 history；
- [ ] Train / Val / Test 使用全局时间切分；
- [ ] 不随机按 sample 切分；
- [ ] Evaluation Candidate Pool 固定；
- [ ] Val/Test target candidate coverage = 100%；
- [ ] Head/Mid/Tail/Unseen 仅根据 Train 定义；
- [ ] Val/Test target strength distribution 已生成；
- [ ] Recall@K / NDCG@K 统一实现；
- [ ] leakage tests 完成；
- [ ] 没有训练任何推荐模型；
- [ ] 所有真实数字来自用户实际运行。

---

# 23. Stage 3 最终原则

Stage 3 的核心不是生成若干 parquet，而是固定：

```text
One Prediction Time
        +
One Target Definition
        +
One Temporal Split
        +
One Candidate Pool
        +
One Item-strength Definition
        +
One Evaluation Protocol
```

从后续 Stage 5 开始：

```text
ItemCF / I2I
Vanilla Two-Tower
Enhanced Two-Tower
多路召回
Ranking
```

必须共同使用这一协议，确保模型之间的指标可公平比较。
