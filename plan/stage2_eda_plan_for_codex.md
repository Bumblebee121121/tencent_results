# TencentGR-1M 项目：阶段 2 EDA 与问题发现计划

> 使用对象：Codex  
> 项目目标：基于 TencentGR-1M 构建“根据用户目标曝光之前的历史行为，从候选广告池中预测下一次被点击广告”的判别式推荐项目。  
> 当前状态：阶段 1（数据理解与数据预处理）已完成。  
> 本阶段要求：**Codex 只负责阅读现有项目、编写/修改代码和运行说明；不要替用户执行任何 EDA 脚本、不要启动全量扫描、不要训练模型。用户会在 VSCode/CMD 中亲自运行代码。**

---

# 0. Codex 的工作边界

## 0.1 必须遵守

1. **只编写代码，不运行数据处理程序。**
2. 不执行 `python xxx.py`、不启动全量 Parquet 扫描、不训练模型、不运行 FAISS。
3. 不安装/升级包，不修改 Conda 环境。
4. 可以阅读项目中的现有 Python 脚本、阶段 1 日志、README/Markdown、Schema 输出。
5. 不移动、不重命名、不删除阶段 1 已有文件。
6. 不修改任何 `data/TencentGR-1M/` 下的原始数据。
7. 不伪造 EDA 数字；所有结论必须等用户真实运行后再从日志、JSON、CSV、图片中得到。
8. 匿名字段语义不明确时，只报告缺失率、基数、分布及统计关系，不擅自赋予业务含义。
9. 阶段 2 不训练任何推荐模型，也不调参。
10. 阶段 2 不正式构造 train/dev/test，不进行负采样；这些属于后续样本与评估协议阶段。
11. 不把 EDA 中观察到的相关性直接写成因果结论。
12. 每个脚本必须有 `main()`、路径检查、批次进度日志、固定输出路径，并对大表使用分批扫描。

---

# 1. 阶段 1 已确认的项目事实

Codex 应把这些事实作为阶段 2 的前提，不要重新做完整数据审计。

## 1.1 核心规模

- 用户数：`1,001,845`
- 历史广告数：`4,783,154`
- 历史行为总数：`90,223,574`
- 候选广告数：`660,000`

## 1.2 核心表一致性

`seq`、`user_feat`、`item_feat` 已确认：

- 主键无缺失、无重复；
- 用户序列无空序列；
- item ID 无非法值；
- timestamp 无缺失；
- 用户序列无时间逆序；
- `seq -> user_feat` 用户覆盖率 100%；
- `seq -> item_feat` 广告覆盖率 100%。

## 1.3 ID 体系

已确认：

- `seq.item_id`：RID
- `item_feat.item_id`：RID
- `candidate.item_id`：OID
- `mm_emb.anonymous_cid`：字符串形式 OID
- OID -> RID 使用 `indexer.pkl` 中的 `indexer["i"]`

**严禁直接拿 candidate/mm_emb 的 OID 与 seq/item_feat 的 RID 做 join。**

## 1.4 行为类型

原始历史序列中：

- `action_type = 0`：曝光
- `action_type = 1`：点击
- `action_type = None`：未知行为类型

全量检查发现：

- `action_type` 缺失 `790,581` 条；
- 缺失比例约 `0.876%`；
- 涉及 `344,817` 名用户；
- 缺失主要发生在序列中间。

阶段 2 中：

- 不把 `None` 强行填成 0 或 1；
- 可以单独统计未知行为；
- 不删除整个事件，因为 item_id 和 timestamp 仍然有效。

## 1.5 候选广告

candidate：

- 660,000 条；
- item_id 无缺失、无重复；
- retrieval_id 无缺失、无重复；
- retrieval_id 为 `0 ~ N-1` 连续编号；
- 511,029 个候选 OID 能映射到历史 RID；
- 148,971 个不能映射；
- OID -> 历史 RID 映射率约 `77.43%`；
- 即约 `22.57%` 为历史 ID 空间未见候选。

这**不是已确认的数据错误**。阶段 2 要进一步分析它是否与 candidate 中的 `cold_start` 信息、侧信息和多模态可用性有关。

## 1.6 多模态

32 维多模态表：

- 总记录约 4,742,961；
- anonymous_cid 无缺失、无重复；
- 所有非空 embedding 均为 32 维；
- 没有 NaN / Inf；
- 没有全零有效向量；
- 有 9,678 条 embedding 为空；
- 多模态 OID 映射到历史 RID 的比例约 96.89%；
- 历史广告多模态 ID 覆盖率约 96.08%。

注意：`96.08%` 是 ID 级覆盖，不完全等于“有效非空向量覆盖率”。

---

# 2. 阶段 2 总目标

阶段 2 不是继续检查“数据有没有坏”，而是回答：

> **这份数据具有哪些会影响推荐模型设计和评估的结构性特点？**

要求形成：

```text
数据事实
    ↓
EDA 统计证据
    ↓
潜在问题
    ↓
可检验的建模假设
```

阶段 2 结束时，必须能回答：

1. 用户历史序列到底有多长？是否大量触及最大长度 100？
2. 曝光、点击、未知行为在整体及用户层面如何分布？
3. 点击行为是否明显稀疏？
4. 用户行为时间跨度和相邻事件时间间隔如何分布？
5. 广告交互是否具有明显长尾？
6. 22.57% 历史未见候选具有什么特征？
7. candidate 中 `cold_start` 与“OID 无法映射历史 RID”是什么关系？
8. 历史未见候选是否仍拥有匿名属性或多模态向量？
9. 多模态缺失集中在哪类广告？是否与历史未见、长尾等因素相关？
10. 哪些问题最值得进入后续特征工程和模型实验？

---

# 3. 推荐目录结构

不要移动阶段 1 文件，只新增：

```text
scripts/
└── eda/
    ├── common.py
    ├── stage2_1_sequence_behavior.py
    ├── stage2_2_temporal_patterns.py
    ├── stage2_3_item_long_tail.py
    ├── stage2_4_candidate_coldstart.py
    ├── stage2_5_multimodal_coverage.py
    └── stage2_6_feature_profile.py

artifacts/
└── eda/
    ├── metrics/
    ├── tables/
    └── figures/

logs/
reports/
```

并创建 `RUN_STAGE2.md`，其中只写用户需要亲自执行的命令。

---

# 4. 统一代码规范

## 4.1 大数据读取

优先使用：

```python
pyarrow.dataset
pyarrow.compute
numpy
```

只有在聚合结果已经很小的时候才转换为 pandas。

禁止：

```python
pd.read_parquet("整个 seq")
```

因为 seq 含约 9000 万条行为。

## 4.2 输出要求

每个 EDA 脚本必须同时输出：

1. 人类可读终端日志；
2. JSON 指标；
3. 必要的 CSV 汇总表；
4. PNG 图。

例如：

```text
artifacts/eda/metrics/stage2_1_sequence_behavior.json
artifacts/eda/tables/stage2_1_sequence_length_hist.csv
artifacts/eda/figures/stage2_1_sequence_length.png
```

## 4.3 图形要求

- 使用 Matplotlib；
- 每张图单独保存；
- 标题、x/y 轴、单位清楚；
- Windows 中文字体不稳定时默认英文标题；
- `dpi >= 150`；
- 使用 `tight_layout()`；
- 保存后关闭 figure；
- 对高度偏斜分布同时考虑原始坐标、log1p/对数坐标、分位数。

## 4.4 日志要求

至少每处理一定数量 batch 输出一次：

```text
已处理 X 个批次
累计用户数 / 行为数 / 广告数
```

日志最后输出：

```text
阶段 2.x 完成
关键输出文件：
...
```

---

# 5. 阶段 2.1：用户序列长度与行为构成

## 当前问题

我们知道 seq 是变长序列，但不知道：

- 用户序列通常多长；
- 是否大量等于 100；
- 点击是否稀疏；
- 未知 action 是否集中在某类用户。

## 本步目的

判断序列建模是否必要，以及后续是否需要显式区分曝光/点击/未知行为。

## 输入

`seq`

## 必须统计

### A. 序列长度

统计：

- 用户数；
- min / max；
- mean / median；
- P25 / P50 / P75 / P90 / P95 / P99；
- 长度 = 100 的用户数和比例；
- 长度区间：1–10、11–20、21–50、51–80、81–99、100。

绘图：

```text
sequence_length_hist.png
sequence_length_cdf.png
```

### B. 行为总体构成

统计：

- action=0 数量与比例；
- action=1 数量与比例；
- action=None 数量与比例。

不要把 `None` 归入曝光。

### C. 用户级行为构成

对每个用户统计：

```text
seq_len
exposure_count
click_count
unknown_count
click_event_ratio = click_count / (exposure_count + click_count)
unknown_ratio = unknown_count / seq_len
```

注意使用 `click_event_ratio`，不要直接叫 CTR。

统计 click_event_ratio：mean / median / P25 / P50 / P75 / P90 / P95 / P99，以及 click_count=0、=1、>=2 的用户比例。

绘图：

```text
click_count_per_user.png
click_event_ratio_distribution.png
unknown_ratio_distribution.png
```

## 完成标准

能回答：

- SASRec 类序列模型是否有数据基础？
- 点击信号是否稀疏？
- 行为类型是否值得作为显式特征？
- 序列最大长度 100 是否可能是数据截断上限？

注意：若长度 100 的用户很多，只能先说“存在明显截断迹象”；除非官方说明明确，否则不要单靠 EDA 断言构造规则。

---

# 6. 阶段 2.2：时间行为模式

## 当前问题

推荐任务是“下一点击广告”，但还不知道用户历史覆盖多长时间，以及相邻行为之间的时间间隔。

## 本步目的

判断：

- 用户兴趣是否可能存在明显时效性；
- 是否值得在后续考虑 time-gap / recency 特征；
- 序列长度相同的用户，其时间跨度是否可能完全不同。

## 输入

`seq.timestamp`

## 必须统计

每名用户：

```text
first_timestamp
last_timestamp
history_span = last - first
```

每个相邻事件：

```text
delta_t = t_j - t_{j-1}
```

统计：

- history_span 分位数；
- delta_t 分位数；
- delta_t=0 比例；
- 按秒、分钟、小时、天换算后的可解释统计；
- 极端时间间隔样本数量。

绘图：

```text
history_span_distribution.png
event_time_gap_distribution.png
```

对 time gap 可用 `log1p(delta_t)` 作图，但必须保留原始分位数。

## 禁止

- 不进行未来信息切分；
- 不构造训练标签；
- 不根据 EDA 偷看后续测试答案。

## 完成标准

能够判断后续是否值得验证：

```text
普通 position embedding
vs
position + time-gap / recency encoding
```

阶段 2 不实现模型。

---

# 7. 阶段 2.3：广告热度与长尾分析

## 当前问题

不知道 478 万历史广告的曝光/点击是否高度集中。

## 本步目的

确认：

- 是否存在明显 popularity bias；
- 长尾广告比例；
- ID-only 模型是否可能更偏头部；
- 后续是否需要分头部/尾部评估。

## 输入

`seq`

## 必须统计

对每个 item RID：

```text
total_event_count
exposure_count
click_count
unknown_count
```

`unique_user_count` 只有在内存成本可控时才精确统计，不要为一个 EDA 指标造成巨大内存压力。

统计：

- item 交互次数 min / median / mean / P90 / P95 / P99 / max；
- 只出现 1 次、2–5 次、6–10 次、11–100 次、>100 次的广告数量和比例；
- Top 1%、5%、10%、20% 热门广告贡献多少历史行为；
- 点击热度分布；
- 曝光热度分布。

绘图：

```text
item_popularity_loglog.png
item_popularity_cdf.png
head_tail_contribution.png
```

## 热度分组

不要拍脑袋固定阈值。优先按真实历史事件数量分位数构造 head/mid/tail，并把阈值保存到 JSON。

## 完成标准

能回答：

- 是否存在明显长尾；
- Popularity baseline 为什么必要；
- 后续指标是否需要按 head/tail 分组报告。

---

# 8. 阶段 2.4：候选池、历史未见广告与 cold_start

这是阶段 2 的重点之一。

## 当前问题

阶段 1 已发现：

```text
660,000 candidate
148,971 candidate OID 无法映射历史 RID
约 22.57%
```

现在需要判断它意味着什么。

## 本步目的

验证：

1. “无法映射历史 RID”与 candidate 中的 `cold_start` 标记有什么关系；
2. 历史未见候选是否仍具有匿名侧信息；
3. 历史未见候选是否拥有多模态向量；
4. 后续是否值得专门设计 cold-start / unseen-item 分组实验。

## 输入

```text
candidate
indexer.pkl
mm_emb
```

## 必须统计

### A. seen / unseen

为 candidate 建立：

```text
is_history_seen
```

定义：

```text
1：OID 在 indexer["i"] 中
0：OID 不在 indexer["i"] 中
```

输出 seen/unseen 数量和比例，用于复现阶段 1 的 77.43% / 22.57%。

### B. cold_start 字段结构

candidate 匿名特征是 struct：

```text
{
    cold_start,
    feature_value
}
```

Codex 必须先阅读实际 Schema，再写通用逻辑。

统计：

- 每个匿名特征字段的 cold_start 取值集合；
- 每个取值数量；
- seen 与 unseen 候选下的 cold_start 分布；
- 不擅自推断数字值的业务含义。

如果不同字段的 cold_start 不一致，应保留字段级结果，不要强行合并成单一广告级状态。

### C. 候选侧信息可用性

分别对 seen / unseen candidate 统计：

- 每个匿名 feature_value 的非空率；
- 可用特征字段数量分布；
- 是否存在“历史未见但拥有侧信息”的广告。

### D. candidate × multimodal

将 candidate OID 与 mm_emb OID 对齐，分别统计：

```text
seen + valid_mm
seen + missing_mm
unseen + valid_mm
unseen + missing_mm
```

`valid_mm` 必须同时满足：

- 找到多模态 OID；
- emb 非空；
- emb 维度合法。

输出：

```text
candidate_seen_mm_matrix.csv
```

## 完成标准

必须明确回答：

```text
22.57% unseen candidate 是否与 cold_start 标记一致？
unseen candidate 中多少有匿名属性？
unseen candidate 中多少有有效多模态？
```

如果大量 unseen candidate 仍有侧信息/多模态，只形成假设：

> ID-only 模型缺乏 unseen item 历史交互表示，而侧信息/多模态可能提供可泛化表示。

不要在阶段 2 写成“已经解决冷启动”。

---

# 9. 阶段 2.5：多模态覆盖与缺失模式

## 当前问题

阶段 1 发现多模态覆盖较高但不完整，有少量 emb=None。单纯知道“有缺失”还不够。

## 本步目的

判断多模态缺失是否随机，还是集中在某些广告群体。

## 输入

```text
mm_emb
indexer.pkl
candidate
阶段 2.3 的 item popularity 聚合结果
```

## 必须统计

### A. 历史广告有效多模态覆盖

区分：

```text
ID coverage
valid embedding coverage
```

分别计算：

- 历史 RID 能找到 mm OID 的比例；
- 历史 RID 能找到非空 32D mm embedding 的比例。

### B. candidate 有效多模态覆盖

分别报告：

```text
all candidate
seen candidate
unseen candidate
```

的 valid-mm 覆盖率。

### C. 与广告热度关系

读取阶段 2.3 保存的 popularity 聚合表，不重新扫描 seq。

按 head/mid/tail 统计：

```text
valid_mm_ratio
missing_mm_ratio
```

### D. 缺失感知结论

阶段 2 只能回答：

```text
多模态缺失是否与 seen/unseen、head/tail 显著相关？
```

如果差异明显就报告；如果差异很小也如实记录。

## 完成标准

至少形成：

```text
mm_coverage_by_group.csv
```

为后续比较以下方案提供依据：

```text
zero-fill + mm_available mask
learnable missing embedding
gating
```

但阶段 2 不实现这些模型。

---

# 10. 阶段 2.6：匿名用户/广告特征 Profile

## 当前问题

存在匿名侧信息，但尚不知道各字段缺失程度、基数、单值/多值差异及是否存在极高基数字段。

## 本步目的

为后续 Feature Embedding 设计提供依据。

## 输入

```text
user_feat
item_feat
candidate
```

## 必须统计

每个字段：

```text
dtype
row_count
null_count
non_null_ratio
unique_count（可行时）
```

列表字段额外统计：

```text
list_length min/median/P90/P99/max
empty_list_ratio
```

candidate struct 字段分别统计：

```text
feature_value 非空率
cold_start 取值分布
```

## 注意

- 不解释匿名特征真实业务语义；
- 不因为基数高就立即 hash；
- 不在阶段 2 决定 Embedding 维度；
- 这里只为下一阶段提供统计基础。

## 输出

```text
user_feature_profile.csv
item_feature_profile.csv
candidate_feature_profile.csv
```

---

# 11. 阶段 2 执行顺序

Codex 按以下顺序编写代码：

```text
2.1 用户序列长度与行为构成
        ↓
2.2 时间模式
        ↓
2.3 广告长尾
        ↓
2.4 candidate seen/unseen + cold_start
        ↓
2.5 多模态覆盖与缺失模式
        ↓
2.6 匿名特征 profile
```

不要把全部 EDA 塞进一个巨大 Python 文件。

---

# 12. 用户运行方式

Codex 创建 `RUN_STAGE2.md`，命令保持类似：

```cmd
python -u scripts\eda\stage2_1_sequence_behavior.py > logs\stage2_1_sequence_behavior.log 2>&1
python -u scripts\eda\stage2_2_temporal_patterns.py > logs\stage2_2_temporal_patterns.log 2>&1
python -u scripts\eda\stage2_3_item_long_tail.py > logs\stage2_3_item_long_tail.log 2>&1
python -u scripts\eda\stage2_4_candidate_coldstart.py > logs\stage2_4_candidate_coldstart.log 2>&1
python -u scripts\eda\stage2_5_multimodal_coverage.py > logs\stage2_5_multimodal_coverage.log 2>&1
python -u scripts\eda\stage2_6_feature_profile.py > logs\stage2_6_feature_profile.log 2>&1
```

用户会自己逐个执行。因此 Codex：

- 不替用户运行；
- 不自动串行执行全部脚本；
- 不创建会自动跑完整阶段的 bat/shell；
- 不在 import 时启动耗时任务。

---

# 13. 每一步的用户学习卡片

每个脚本顶部 docstring 必须回答：

```text
所属阶段：
当前问题：
本步目的：
为什么现在做：
输入：
输出：
完成标准：
对后续模型的潜在影响：
面试时如何讲：
```

示例：

```python
"""
所属阶段：
阶段 2.1：用户序列长度与行为构成

当前问题：
尚不清楚用户序列长度、点击稀疏程度和未知行为分布。

本步目的：
量化用户行为序列结构，为后续序列模型和行为类型建模提供依据。

为什么现在做：
阶段 1 已证明 seq 数据可靠，现在才能对其真实分布做统计分析。

输入：
data/TencentGR-1M/seq

输出：
JSON 指标、CSV 汇总、PNG 图、终端日志。

完成标准：
能回答序列是否普遍较长、是否存在长度 100 聚集、点击是否稀疏。

对后续模型的潜在影响：
决定是否值得使用 SASRec、行为 Embedding、时间特征等。

面试时如何讲：
我先通过 EDA 确认序列结构和行为稀疏性，再决定是否引入序列建模，而不是直接堆 Transformer。
"""
```

---

# 14. 阶段 2 完成后的“问题发现”要求

阶段 2 不以“画完图”为完成标准。

所有脚本运行完后，应整理：

```text
数据现象 | 统计证据 | 潜在影响 | 可检验假设 | 后续是否验证
```

逻辑示例（示例数字不可伪造）：

```text
大量序列长度达到 100
→ 可能存在历史截断
→ 长期行为无法完整观察
→ 最近行为可能更重要
→ 后续验证 recency/time-aware

广告热度高度长尾
→ 头部广告贡献大量交互
→ ID 模型可能偏热门
→ tail item 表示较弱
→ 分 head/tail 报告 Recall/NDCG

大量候选历史未见
→ ID-only 无历史 embedding 学习信号
→ unseen retrieval 可能困难
→ side/mm features 可能提高泛化
→ seen/unseen 分组实验

多模态覆盖非 100%
→ 模态融合存在缺失输入
→ 无脑融合可能受影响
→ missing-aware 方法可能更稳
→ 有/无 mask 消融
```

这里只说明逻辑格式，不代表最终结果。

---

# 15. 阶段 2 禁止得出的结论

没有模型实验前，不允许写：

```text
“多模态解决了冷启动”
“时间特征一定提升 SASRec”
“长尾问题导致 NDCG 下降”
“unknown action 对模型有负面影响”
“某匿名特征非常重要”
```

阶段 2 最多写：

```text
“发现……现象”
“可能导致……”
“提出假设……”
“后续需要通过消融/分组实验验证……”
```

---

# 16. 阶段 2 完成标准

- [ ] 序列长度与行为构成完成；
- [ ] 时间跨度/time gap 完成；
- [ ] item 长尾完成；
- [ ] candidate seen/unseen 完成；
- [ ] candidate cold_start 与 unseen 关系完成；
- [ ] unseen candidate 的侧信息/多模态覆盖完成；
- [ ] 多模态有效覆盖及缺失分组完成；
- [ ] 用户/广告匿名特征 profile 完成；
- [ ] 所有数值有 JSON/CSV 保存；
- [ ] 所有关键分布有 PNG；
- [ ] 不存在明显的数据泄漏分析；
- [ ] 至少形成 2~4 个“数据事实 -> 建模假设”，但尚未声称模型有效。

---

# 17. Codex 本轮最终交付物

请直接在项目中完成：

```text
scripts/eda/common.py
scripts/eda/stage2_1_sequence_behavior.py
scripts/eda/stage2_2_temporal_patterns.py
scripts/eda/stage2_3_item_long_tail.py
scripts/eda/stage2_4_candidate_coldstart.py
scripts/eda/stage2_5_multimodal_coverage.py
scripts/eda/stage2_6_feature_profile.py
RUN_STAGE2.md
```

并确保：

- 路径适配当前项目根目录；
- Windows CMD 可运行；
- Python 3.10 可运行；
- 使用项目当前已有 pandas / pyarrow / numpy / matplotlib；
- 不依赖新安装的软件包；
- 全量统计尽量流式/批量完成；
- 不修改原始数据；
- **不要执行这些程序。**

最后只向用户总结：

1. 新建/修改了哪些文件；
2. 每个脚本负责什么；
3. 用户应该先运行哪一个命令；
4. 不分析任何尚未真实运行得到的结果。

**第一步只让用户先运行 `stage2_1_sequence_behavior.py`。用户检查结果后，再继续下一项。**
