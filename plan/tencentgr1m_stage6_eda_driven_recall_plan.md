# TencentGR-1M Stage 6：EDA-driven 召回模型迭代 — Codex 实施计划书

> 目标：在 **Stage 5 已冻结的 I2I ItemCF（Item-to-Item Item-based Collaborative Filtering，基于物品相似度的协同过滤召回）与 Pure-ID Vanilla Two-Tower（只使用广告 ID 的基础版双塔模型）** 之上，严格沿着 Stage 2 EDA（Exploratory Data Analysis，探索性数据分析）发现的问题进行逐步增强，并通过可归因的消融实验回答：
>
> 1. 用户长期/短期兴趣建模是否比“历史广告 ID 简单平均”更有效？
> 2. 显式区分曝光/点击行为是否有增益？
> 3. 显式使用真实时间间隔是否有增益？
> 4. Side Feature（结构化侧信息）与 MM（Multimodal，多模态信息）能否缓解 Tail（训练期历史很少）/Unseen（训练期从未出现）广告的纯 ID 弱信号问题？
> 5. History-strength-aware（历史强度感知）能否根据广告训练期历史强弱，自适应决定更相信 ID 还是非 ID 信息？
> 6. Enhanced Two-Tower（增强双塔）与 I2I 是否仍保持互补，并在真实多路召回融合后获得增量覆盖？

---

## 0. Stage 6 在整个项目中的位置

```text
阶段 0：项目与任务定义
        ↓
阶段 1：数据理解与数据预处理
        ↓
阶段 2：EDA 与问题发现
        ↓
阶段 3：样本构造、时间切分与评估协议
        ↓
阶段 4：特征工程
        ↓
阶段 5：召回 Baseline
        │
        ├── I2I ItemCF
        └── Pure-ID Vanilla Two-Tower
        ↓
阶段 6：EDA-driven 召回模型迭代
        │
        ├── User Tower（用户塔）
        │     SDM 长短期兴趣
        │       ↓
        │     + Action-aware（行为感知）
        │       ↓
        │     + Time-aware（时间感知）
        │
        ├── Item Tower（广告塔）
        │     ID
        │      ↓
        │     + Side/MM
        │      ↓
        │     + History-strength-aware
        │
        └── Enhanced Two-Tower
        ↓
      单路召回评估 + 互补性分析
        ↓
      I2I + Enhanced Two-Tower
        ↓
        多路召回融合
        ↓
阶段 7：排序模型与完整推荐链路
        ↓
阶段 8：评估、误差分析、收益归因、
        工程化检索与项目总结
```

Stage 6 的定位不是“追求一个复杂大模型”，而是：

$$
\boxed{
\text{Stage 5 暴露问题}
\rightarrow
\text{Stage 6 逐条验证 EDA 假设}
}
$$

---

# 1. Stage 6 必须继承、不能重新定义的事实

## 1.1 Stage 3 协议冻结

Stage 6 必须继续消费 Stage 3 已经确定的：

- next-click（下一次点击）监督样本；
- Train / Validation / Test 时间切分；
- `click_target_prefix_v2` 样本协议；
- Validation Primary / Test Primary；
- 正式 evaluation candidate pool（评估候选池）；
- Head / Mid / Tail / Unseen 分组。

禁止在 Stage 6：

- 重新造 target；
- 修改 Train cutoff；
- 因模型需要而换一套 Validation/Test；
- 从目标之后的行为构造用户特征；
- 用 Test 调参。

---

## 1.2 Stage 4 特征协议冻结

Stage 4 已经完成：

- 广告模型 ID 语义；
- Train-Unseen → `UNK`（Unknown，未知广告共享编号）；
- Action token（行为编号）；
- Sequence Store（序列存储）；
- Time Feature Utility（时间特征工具）；
- 13 个 Item Side Feature；
- 32 维 MM；
- Missing（缺失）/ OOV（Out-of-Vocabulary，训练词表外新值）；
- train-only item strength（只使用训练截止时间以前的广告历史强度）。

Stage 6 **直接读 Stage 4 store，不重新从原始 parquet 重建这些特征**。

重点复用：

```text
artifacts/stage4/
├── feature_store/
│   ├── item_side_tokens_by_rid.npy
│   ├── item_side_missing_by_rid.npy
│   ├── item_side_oov_by_rid.npy
│   ├── eval_candidate_side.parquet
│   ├── mm_by_rid.npy
│   ├── mm_valid_by_rid.npy
│   ├── eval_candidate_mm.npy
│   ├── eval_candidate_mm_valid.npy
│   ├── user_seq_offsets.npy
│   ├── seq_item_rid.npy
│   ├── seq_action_token.npy
│   └── seq_timestamp.npy
│
├── mappings/
│   ├── train_item_count_by_rid.npy
│   ├── rid_to_model_item_token.npy
│   ├── feature_vocab_manifest.json
│   └── vocab/
│
└── manifests/
    └── stage4_manifest.json
```

不得使用：

```text
candidate.cold_start
retrieval_id 作为模型广告 ID
Validation/Test 统计拟合词表
未来事件统计 item strength
```

---

## 1.3 `inspect_data.log` 对 Stage 6 的直接意义

原始 `seq` 每个行为本身包含：

```text
item_id
action_type
timestamp
```

所以用户侧真正拥有：

$$
(\text{广告},\text{行为},\text{真实时间})
$$

三类对齐信息，而 Stage 5 Pure-ID Two-Tower 只用了广告 ID。

广告侧 `item_feat` 有 13 个匿名结构化字段：

```text
100, 101, 102, 112, 114, 115, 116,
117, 118, 119, 120, 121, 122
```

MM 文件提供：

$$
\mathbf{x}_i^{MM}\in\mathbb{R}^{32}
$$

因此 Stage 6 的 Action-aware、Time-aware、Side/MM 都是已有真实数据支持的，不是为了堆模型临时造特征。

---

# 2. Stage 5 的结果必须作为 Stage 6 的固定起点

Stage 5 已经得到两个可信的基础召回。

## 2.1 I2I ItemCF

当前最优配置：

```text
click3_recent20
```

Validation：

$$
Recall@100 = 0.0773904771 \approx 7.74\%
$$

其中：

$$
Recall@100_{\text{Head}}\approx13.94\%
$$

$$
Recall@100_{\text{Mid}}\approx7.48\%
$$

$$
Recall@100_{\text{Tail}}\approx4.07\%
$$

$$
Recall@100_{\text{Unseen}}=0
$$

---

## 2.2 Pure-ID Vanilla Two-Tower

Validation：

$$
Recall@100 = 0.0154666971 \approx 1.55\%
$$

其中：

$$
Recall@100_{\text{Head}}\approx3.12\%
$$

$$
Recall@100_{\text{Mid}}\approx0.074\%
$$

$$
Recall@100_{\text{Tail}}=0
$$

$$
Recall@100_{\text{Unseen}}=0
$$

Stage 5 已经证明：

$$
\boxed{
\text{简单历史 ID 平均}
+
\text{纯 ID 广告表示}
\text{能力明显不足}
}
$$

---

## 2.3 两路召回又不是完全重复

Test 的 `@100` 命中互补性中，Two-Tower 仍存在：

```text
Two-Tower only hit = 1717
```

即有 1717 个目标：

```text
I2I 没命中
Two-Tower 命中
```

因此 Stage 6 的目标不是“用 Enhanced Two-Tower 替代 I2I”，而是：

$$
\boxed{
\text{先增强 U2I 双塔召回}
\rightarrow
\text{再与 I2I 做多路融合}
}
$$

其中 U2I（User-to-Item，用户到物品召回）指用用户向量直接检索广告向量。

---

# 3. Stage 4 给 Stage 6 的物品侧直接证据

Stage 4 正式结果：

```text
eval_candidate_count = 815,232
Train-Unseen eval candidate = 585,776
```

即候选池中大量广告没有训练期 ID 学习信号。

但 MM：

```text
eval candidate MM valid = 785,668
eval candidate MM missing = 29,564
```

即：

$$
\frac{785668}{815232}\approx96.37\%
$$

候选广告拥有有效 32 维 MM。

这直接支持：

$$
\boxed{
\text{Unseen 没有可靠 ID}
\;\text{不等于}\;
\text{Unseen 没有可用内容信息}
}
$$

Side Feature 也并非所有字段都同样可靠：

- `121` 在 eval candidate 上 OOV 约 `57.69%`；
- `115` 在 eval candidate 上 Missing 约 `70.44%`；
- `102` / `122` 的 eval OOV 约 `8.56%` / `8.61%`。

因此 Stage 6 **不能假设所有 Side 字段都同样有用**，必须保留 Side ablation（消融）能力。

---

# 4. Stage 6 的总实验原则

Stage 6 只允许遵循：

```text
问题
  ↓
假设
  ↓
只增加一个核心变量
  ↓
Validation 实验
  ↓
Head/Mid/Tail/Unseen 分组
  ↓
归因
  ↓
再进入下一步
```

不要直接做：

```text
SDM
+ Action
+ Time
+ Side
+ MM
+ Gate
一次全部加上
```

否则最终即使 Recall 提升，也无法回答：

> 到底是哪一项带来的？

---

# 5. 预注册实验矩阵

Stage 6 主线先固定以下实验，不允许 Codex 自己再加十几个模型。

| ID | User Tower | Item Tower | 核心问题 |
|---|---|---|---|
| B0 | Stage 5 历史 ID mean pooling | Pure-ID | Stage 5 固定 Baseline |
| U1 | SDM 长短期兴趣 | Pure-ID | 长短期建模是否有效 |
| U2 | SDM + Action-aware | Pure-ID | 曝光/点击语义是否有效 |
| U3 | SDM + Action-aware + Time-aware | Pure-ID | 真实时间是否进一步有效 |
| I1 | U3 | ID + Side | 结构化属性是否有效 |
| I2 | U3 | ID + MM | 多模态是否有效 |
| I3 | U3 | ID + Side + MM | 两类非 ID 信息是否互补 |
| E1 | U3 | ID + Side + MM + History-strength-aware | 最终 Enhanced Two-Tower |

这里：

- `B0` 不训练，直接读取 Stage 5 正式结果；
- `U1 → U2 → U3` 是用户侧递进链；
- `I1 / I2` 是并列消融；
- `I3` 验证 Side 与 MM 是否组合有效；
- `E1` 是最终 Enhanced Two-Tower。

必须优先保持：

```text
embedding_dim = 64
random_negatives = 20
seed = 42
Stage 3/4 protocol 不变
Recall K 不变
FAISS/HNSW 精度审计不变
```

从而尽量把收益归因到“表示信息变化”，而不是同时改训练任务。

---

# 6. 推荐新增目录

```text
configs/
└── stage6.yaml

scripts/
└── stage6/
    ├── stage6_0_contract_audit.py
    ├── stage6_1_sequence_adapter_smoke.py
    ├── stage6_2_train_user_variants.py
    ├── stage6_3_train_item_variants.py
    ├── stage6_4_train_enhanced_two_tower.py
    ├── stage6_5_build_indexes.py
    ├── stage6_6_evaluate_variants.py
    ├── stage6_7_compare_ablation.py
    ├── stage6_8_channel_complementarity.py
    ├── stage6_9_fuse_recall.py
    └── stage6_10_finalize.py

src/
├── models/
│   ├── sdm_user_tower.py
│   ├── action_time_encoder.py
│   ├── content_item_tower.py
│   ├── history_strength_gate.py
│   └── enhanced_two_tower.py
│
└── recall/
    ├── stage6_data.py
    ├── stage6_training.py
    ├── stage6_index.py
    ├── stage6_evaluation.py
    └── fusion.py

tests/
└── stage6/
    ├── test_stage6_contract.py
    ├── test_sequence_split.py
    ├── test_sdm_user_tower.py
    ├── test_action_time_encoder.py
    ├── test_content_item_tower.py
    ├── test_strength_gate.py
    ├── test_unseen_candidate.py
    ├── test_stage6_training.py
    ├── test_stage6_retrieval.py
    └── test_fusion.py

artifacts/
└── stage6/
    ├── audits/
    ├── checkpoints/
    ├── indexes/
    ├── metrics/
    ├── predictions/
    ├── reports/
    └── manifests/

logs/
└── stage6/

RUN_STAGE6.md
```

---

# 7. `configs/stage6.yaml` 建议结构

不要把所有参数散落在 Python 文件里。

建议：

```yaml
data_root: data/TencentGR-1M
stage3_root: artifacts/stage3
stage4_root: artifacts/stage4
stage5_root: artifacts/stage5
output_root: artifacts/stage6
log_root: logs/stage6

stage3_protocol_version: click_target_prefix_v2
stage4_protocol_version: stage4_train_only_features_v1
stage5_recall_protocol_version: stage5_recall_baseline_v2
stage6_protocol_version: stage6_eda_driven_recall_v1

seed: 42

recall_ks: [10, 50, 100, 500]
ndcg_ks: [10, 50, 100]

model:
  embedding_dim: 64

training:
  random_negatives: 20
  max_epochs: 12
  batch_size: 512
  sparse_learning_rate: 0.01
  dense_learning_rate: 0.001
  early_stopping_patience: 2
  early_stopping_min_delta: 0.002
  num_workers: 0
  pin_memory: false

user_tower:
  short_session:
    session_gap_seconds: null
    max_events: null
  long_history:
    max_events: null
  attention:
    num_heads: 4
    dropout: 0.1
  action_embedding_dim: 64
  time_feature_dim: 3

item_tower:
  side_fields:
    ["100","101","102","112","114","115","116",
     "117","118","119","120","121","122"]
  side_embedding_dim: 8
  mm_input_dim: 32
  mm_projection_dim: 64

strength_gate:
  force_unseen_id_weight_zero: true
  monotonic_id_weight: true

faiss:
  index_type: hnsw_flat_ip
  M: 32
  efConstruction: 200
  efSearch: 512
  embedding_batch_size: 16384
  query_batch_size: 2048

fusion:
  method: rrf
  rrf_c: 60
  validation_weights: [0.0, 0.25, 0.5, 0.75, 1.0]

debug:
  max_train_samples: 10000
  max_eval_samples: 1000
  epochs: 1
```

注意：

> `session_gap_seconds`、`max_events` 不能让 Codex 凭感觉随便写死。必须通过 Stage 6.1 审计确定，并写入最终 config。

---

# 8. Stage 6.0：合同审计与 Baseline 冻结

文件：

```text
scripts/stage6/stage6_0_contract_audit.py
```

## 8.1 目的

回答：

> Stage 6 是否真的建立在 Stage 3/4/5 的同一任务和同一候选池之上？

## 8.2 必查内容

检查：

```text
Stage 3 protocol
Stage 4 protocol
Stage 5 recall protocol
train cutoff
Validation/Test sample count
eval candidate count
Head/Mid/Tail/Unseen 阈值
Stage 5 B0 metrics
Stage 5 ItemCF metrics
```

正式应读取：

```text
artifacts/stage4/manifests/stage4_manifest.json
artifacts/stage5/two_tower/metrics.json
artifacts/stage5/itemcf/metrics.json
artifacts/stage5/reports/channel_complementarity.json
```

冻结 Baseline：

```json
{
  "itemcf_variant": "click3_recent20",
  "itemcf_validation_recall100": 0.0773904771090253,
  "two_tower_validation_recall100": 0.015466697062292034
}
```

将其保存：

```text
artifacts/stage6/manifests/baseline_freeze.json
```

## 8.3 绝对禁止

Stage 6 脚本不得覆盖：

```text
artifacts/stage5/
```

Stage 5 是只读 Baseline。

---

# 9. Stage 6.1：统一 Stage 6 数据适配层

文件：

```text
src/recall/stage6_data.py
scripts/stage6/stage6_1_sequence_adapter_smoke.py
```

## 9.1 直接复用 Stage 4

优先复用：

```python
FeatureStore
NextClickFeatureDataset
add_dynamic_time_features
```

不要重新写一套 ID mapping。

一个 Stage 6 样本至少应能得到：

```text
sample_id
user_id
target_item_rid
target_item_token
target_timestamp

hist_item_token
hist_action_token
hist_timestamp

hist_recency_log1p
hist_time_gap_log1p
hist_first_event_mask

target_item_side
target_item_side_missing
target_item_side_oov

target_mm
target_mm_valid

target_train_count
target_train_count_log1p
target_strength_group
```

---

## 9.2 长短期历史切分

SDM（Sequential Deep Matching，序列深度匹配）的原始思想是同时捕获短期 session 兴趣与长期行为兴趣。

本项目不要盲目复制论文里的“10 分钟 session”参数。

应先根据 Stage 2 的时间间隔 EDA 或 Train-only history 统计确定短期 session gap。

### 定义

目标前历史：

$$
H_u=(e_1,e_2,\ldots,e_m)
$$

其中：

$$
e_k=(i_k,a_k,t_k)
$$

最后一个行为所在短期 session 记为：

$$
S_u
$$

发生在该 session 之前的行为记为：

$$
L_u
$$

### session boundary

从最后一个事件向前扫描。

若：

$$
t_k-t_{k-1}>\tau_{\text{session}}
$$

则视为短期 session 边界。

其中：

$$
\tau_{\text{session}}
$$

必须写入 config 和 audit。

---

## 9.3 如果 Stage 2 没给出清晰阈值

不要直接照论文取 10 分钟。

只允许一个很小的 Validation-only 候选集，例如：

```text
10 min
30 min
60 min
```

只用 Validation 选择一次，然后冻结。

保存：

```text
artifacts/stage6/audits/session_definition.json
```

至少记录：

```text
selected_session_gap_seconds
source = stage2_eda / train_only_validation_selection
short_session_length p50/p90/p99
long_history_length p50/p90/p99
empty_long_ratio
single_event_short_ratio
history_truncation_ratio
```

---

## 9.4 序列截断

不能让最长用户把 GPU 显存拖垮，也不能凭感觉写 `max_len=50`。

应先统计 Train：

```text
short length p50/p90/p95/p99
long length p50/p90/p95/p99
```

再确定：

```text
short_max_events
long_max_events
```

并记录：

$$
\text{truncation rate}
=
\frac{\text{被截断样本数}}
{\text{总样本数}}
$$

如果截断比例很高，要在结果报告中明确。

---

# 10. Stage 6.2：U1 — SDM 长短期兴趣 User Tower

文件：

```text
src/models/sdm_user_tower.py
scripts/stage6/stage6_2_train_user_variants.py
```

## 10.1 这一实验只解决一个问题

Stage 5：

```text
所有历史广告 ID
      ↓
简单平均
      ↓
用户向量
```

U1 改成：

```text
长期历史
   ↓
长期兴趣编码
        \
         → 长短期融合 → User Vector
        /
短期 session
   ↓
短期兴趣编码
```

此阶段：

```text
不使用 Action
不使用显式 Time Feature
不使用 Side
不使用 MM
不使用 item strength
```

否则无法归因。

---

## 10.2 短期兴趣

历史广告 ID Embedding：

$$
\mathbf{e}_{i_k}\in\mathbb{R}^{64}
$$

短期 session：

$$
S_u=
(\mathbf{e}_{i_1},\ldots,\mathbf{e}_{i_s})
$$

使用轻量 Multi-Head Self-Attention（多头自注意力）建模短期内部关系。

输出：

$$
\mathbf{s}_u\in\mathbb{R}^{64}
$$

不要引入 Transformer 大堆叠。

建议：

```text
1 layer
4 heads
d_model = 64
```

Stage 6 的目标是验证假设，不是扩大网络规模。

---

## 10.3 长期兴趣

长期历史：

$$
L_u=
(\mathbf{e}_{j_1},\ldots,\mathbf{e}_{j_l})
$$

可用短期向量作为 query，对长期行为做 attention：

$$
\alpha_k
=
\operatorname{softmax}
\left(
\frac{
\mathbf{s}_u^\top W\mathbf{e}_{j_k}
}{
\sqrt{d}
}
\right)
$$

$$
\mathbf{l}_u
=
\sum_k
\alpha_k\mathbf{e}_{j_k}
$$

目的：

> 不是把长期所有行为平均，而是从长期历史中找到与当前短期兴趣更相关的部分。

---

## 10.4 长短期 Gate

使用可学习门控：

$$
\mathbf{g}_u
=
\sigma
\left(
W_g[\mathbf{s}_u;\mathbf{l}_u]+\mathbf{b}_g
\right)
$$

最终：

$$
\mathbf{u}
=
\mathbf{g}_u\odot\mathbf{s}_u
+
(1-\mathbf{g}_u)\odot\mathbf{l}_u
$$

最后：

$$
\mathbf{u}
\leftarrow
\frac{\mathbf{u}}
{\|\mathbf{u}\|_2}
$$

这就是本项目的 **SDM-style 长短期用户塔**。

注意：

> 这是针对 TencentGR-1M 的 SDM 风格实现，不应在文档中声称“完整复现原始 SDM 论文所有模块”。

因为本项目不会在主线里加入原 SDM 可能使用的其他用户画像模块。

---

# 11. Stage 6.3：U2 — + Action-aware

## 11.1 问题

Stage 2 已经发现：

```text
曝光
点击
未知行为
```

语义并不一样。

Stage 5 / U1 都只知道：

> 用户碰过这个广告。

但不知道：

> 是看过没点，还是主动点过。

---

## 11.2 实现

Stage 4 已定义：

```text
PAD = 0
Exposure = 1
Click = 2
Unknown = 3
```

为 Action 建立：

$$
\mathbf{e}_{a_k}\in\mathbb{R}^{64}
$$

事件表示改成：

$$
\mathbf{x}_k
=
\mathbf{e}_{i_k}
+
\mathbf{e}_{a_k}
$$

然后保持 U1 的：

```text
session 定义
长短期结构
attention
gate
loss
negative sampling
item tower
```

全部不变。

只有：

```text
item-only event
        ↓
item + action event
```

变化。

---

## 11.3 必做归因

输出：

```text
U1 vs U2
Overall Recall@K
Head/Mid/Tail/Unseen Recall@K
```

结论只能写：

> Action-aware 是否在相同长短期结构下提供额外收益。

不要把 U2 直接和 B0 的全部差值都归因于 Action。

正确：

$$
\Delta_{\text{Action}}
=
Metric(U2)-Metric(U1)
$$

---

# 12. Stage 6.4：U3 — + Time-aware

## 12.1 问题

序列位置相邻不代表真实时间相近。

例如：

```text
A → B
```

可能隔：

```text
2 分钟
```

也可能隔：

```text
3 天
```

所以只靠序列位置可能无法准确表示“当前兴趣”。

---

## 12.2 Stage 4 已准备的时间量

目标时间为：

$$
T
$$

历史第 $k$ 个行为时间：

$$
t_k
$$

Recency（距离目标多久）：

$$
r_k=T-t_k
$$

相邻行为时间差：

$$
\Delta t_k=t_k-t_{k-1}
$$

Stage 4 已提供：

$$
\log(1+r_k)
$$

与：

$$
\log(1+\Delta t_k)
$$

---

## 12.3 不建议第一版硬做很多时间桶

优先采用连续时间：

$$
\mathbf{q}_k
=
\left[
\log(1+r_k),
\log(1+\Delta t_k),
m_k^{first}
\right]
$$

其中：

$$
m_k^{first}\in\{0,1\}
$$

表示它是否为历史序列第一个事件。

Train-only 计算均值/标准差：

$$
\tilde{\mathbf q}_k
=
\frac{
\mathbf q_k-\boldsymbol\mu_{\text{train}}
}{
\boldsymbol\sigma_{\text{train}}+\epsilon
}
$$

再通过一个小 MLP（Multi-Layer Perceptron，多层感知机）：

$$
\mathbf{e}_{t_k}
=
MLP_{time}
(
\tilde{\mathbf q}_k
)
$$

事件表示：

$$
\mathbf{x}_k
=
\mathbf{e}_{i_k}
+
\mathbf{e}_{a_k}
+
\mathbf{e}_{t_k}
$$

这样不会因为手工随意选择几十个时间桶而增加额外超参数。

---

## 12.4 防泄漏要求

必须 assert：

$$
t_k<T
$$

若任何样本出现：

$$
t_k\ge T
$$

直接报错。

不得 clip 后继续训练。

---

## 12.5 归因

Time 增量：

$$
\Delta_{\text{Time}}
=
Metric(U3)-Metric(U2)
$$

不要用：

$$
Metric(U3)-Metric(B0)
$$

来宣称全部是 Time 收益。

---

# 13. User Tower 阶段冻结点

只有完成：

```text
B0
U1
U2
U3
```

Validation 比较后，才进入 Item Tower。

生成：

```text
artifacts/stage6/reports/user_tower_ablation.csv
```

至少包含：

```text
variant
split
group
K
Recall
NDCG
delta_vs_previous
```

并保存：

```text
artifacts/stage6/manifests/user_tower_selection.json
```

只能使用 Validation 选 User Tower。

Test 在主线结构冻结前不得用来反复调参。

---

# 14. Stage 6.5：Item Tower — Side / MM

这一阶段固定 User Tower 使用：

```text
U3 = SDM + Action-aware + Time-aware
```

然后只改 Item Tower。

---

# 15. I1 — ID + Side

## 15.1 Side 输入

13 个字段：

```text
100
101
102
112
114
115
116
117
118
119
120
121
122
```

每个字段已有独立 Train-only vocabulary（训练词表）。

其中：

```text
PAD = 0
MISSING = 1
OOV = 2
```

不要重新 fit。

---

## 15.2 Side Encoder

每个字段：

$$
z_{i,f}
=
Embedding_f(token_{i,f})
$$

建议第一版统一：

```text
side_embedding_dim = 8
```

理由：

- 13 个字段拼接只有 `13 × 8 = 104` 维；
- `121` vocabulary 超过 185 万，若直接每个字段 64 维会显著浪费显存/内存；
- Stage 6 第一目标是验证 Side 是否有增量，不是给每个 Side 字段做大模型。

拼接：

$$
\mathbf z_i^{side}
=
[
\mathbf z_{i,100};
\mathbf z_{i,101};
\ldots;
\mathbf z_{i,122}
]
$$

投影：

$$
\mathbf h_i^{side}
=
MLP_{side}
(
\mathbf z_i^{side}
)
\in\mathbb R^{64}
$$

---

## 15.3 ID + Side 第一版融合

先用简单 concat + projection：

$$
\mathbf v_i
=
Normalize
\left(
MLP
[
\mathbf e_i^{ID};
\mathbf h_i^{side}
]
\right)
$$

此时**还不要做 history strength gate**。

否则无法回答：

> Side 本身有没有帮助？

---

# 16. I2 — ID + MM

Stage 4 MM：

$$
\mathbf x_i^{MM}\in\mathbb R^{32}
$$

且没有 L2 normalize。

第一版：

$$
\mathbf h_i^{MM}
=
MLP_{MM}
(
\mathbf x_i^{MM}
)
\in\mathbb R^{64}
$$

必须使用：

```text
mm_valid
```

不能因为缺失 MM 存成零向量，就假装它是真实零向量。

可实现：

$$
\tilde{\mathbf h}_i^{MM}
=
m_i\mathbf h_i^{MM}
+
(1-m_i)\mathbf e_{MM\_missing}
$$

其中：

$$
m_i\in\{0,1\}
$$

为 `mm_valid`。

ID + MM：

$$
\mathbf v_i
=
Normalize
\left(
MLP
[
\mathbf e_i^{ID};
\tilde{\mathbf h}_i^{MM}
]
\right)
$$

---

# 17. I3 — ID + Side + MM

用于回答：

> Side 与 MM 是重复信息，还是能继续互补？

构造：

$$
\mathbf h_i^{nonID}
=
MLP
[
\mathbf h_i^{side};
\tilde{\mathbf h}_i^{MM}
]
$$

然后：

$$
\mathbf v_i
=
Normalize
\left(
MLP
[
\mathbf e_i^{ID};
\mathbf h_i^{nonID}
]
\right)
$$

此时仍然是固定融合。

---

# 18. Side 的针对性消融

以下不是 Stage 6 主实验编号，不要一开始全部跑。

只有当 `I1` 或 `I3` 结果异常时再触发：

```text
drop_f121
drop_f115
drop_high_oov_group
```

原因：

- `121` eval OOV ≈ 57.69%；
- `115` eval Missing ≈ 70.44%。

优先原则：

```text
先验证全部 Side 有没有价值
↓
再针对异常字段做诊断
```

不要一开始就在 13 个字段上做 13 次 leave-one-out（逐字段删除）实验。

---

# 19. Stage 6.6：History-strength-aware Item Tower

这是 Stage 6 物品侧最重要的“归因型”设计。

## 19.1 EDA 假设

广告训练期历史越多：

> ID Embedding 越可靠。

训练期历史越少：

> ID 越不可靠，应更多依赖 Side/MM。

训练期完全没有：

> 不应该相信广告专属 ID。

---

## 19.2 输入

Stage 4 已有：

$$
n_i^{train}
=
\text{广告 }i\text{ 在 Train cutoff 前出现次数}
$$

和：

$$
s_i
=
\log(1+n_i^{train})
$$

禁止重新从 Validation/Test 统计。

---

## 19.3 建议采用可解释的单调 scalar gate

定义：

$$
g_i
=
\mathbb{1}(n_i^{train}>0)
\cdot
\sigma
\left(
softplus(a)\cdot\log(1+n_i^{train})+b
\right)
$$

其中：

$$
g_i\in[0,1]
$$

表示“相信 ID 的程度”。

由于：

$$
softplus(a)>0
$$

所以广告历史越强，ID 权重天然单调不减。

并且当：

$$
n_i^{train}=0
$$

强制：

$$
g_i=0
$$

因此 Unseen 广告完全不依赖共享 `UNK` ID。

---

## 19.4 最终 Item 表示

ID：

$$
\mathbf h_i^{ID}
$$

非 ID：

$$
\mathbf h_i^{nonID}
$$

最终：

$$
\mathbf v_i
=
Normalize
\left(
g_i\mathbf h_i^{ID}
+
(1-g_i)\mathbf h_i^{nonID}
\right)
$$

这比一个不可解释的黑盒 concat 更适合本项目的研究假设。

---

## 19.5 Gate 审计

必须输出：

```text
Head mean/p10/p50/p90 gate
Mid mean/p10/p50/p90 gate
Tail mean/p10/p50/p90 gate
Unseen mean/p10/p50/p90 gate
```

Unseen 必须满足：

$$
g_i=0
$$

如果不是，单元测试失败。

保存：

```text
artifacts/stage6/audits/history_strength_gate.json
```

---

# 20. Stage 6.7：Enhanced Two-Tower

最终：

```text
User Tower
=
SDM
+ Action-aware
+ Time-aware

Item Tower
=
ID
+ Side
+ MM
+ History-strength-aware
```

记为：

```text
E1 = Enhanced Two-Tower
```

相似度仍使用：

$$
score(u,i)
=
\mathbf u^\top\mathbf v_i
$$

两侧都先 L2 normalize，因此为 cosine-equivalent inner product（等价余弦相似度的内积）。

不要在 Stage 6 再加入：

```text
Cross Network
DIN
DeepFM
GBDT
复杂 ranker
```

这些属于 Stage 7 排序。

---

# 21. Stage 6 训练目标

为保证与 Stage 5 尽可能可比，继续使用 sampled softmax（采样 Softmax）形式。

一个正样本：

$$
i^+
$$

以及 $N$ 个随机负样本：

$$
i_1^-,\ldots,i_N^-
$$

其中：

$$
N=20
$$

第一版保持不变。

Loss：

$$
\mathcal L
=
-
\log
\frac{
\exp(score(u,i^+))
}{
\exp(score(u,i^+))
+
\sum_{j=1}^{N}
\exp(score(u,i_j^-))
}
$$

---

# 22. Negative Sampling（负采样）必须保持干净

训练负样本池继续使用：

> `train_item_count > 0` 的 Train-Seen item。

不要把：

```text
Validation/Test eval candidate membership
```

变成训练负样本选择依据。

原因：

> 即使 Stage 6 能在推理时给 Unseen candidate 计算 Side/MM 向量，也不能在训练时利用“它恰好属于未来 eval candidate pool”这个事实。

因此：

```text
Train negative pool
=
Train-Seen items
```

与：

```text
Inference index
=
Stage 6 可表示的 eval candidates
```

必须分开。

---

# 23. Stage 6 与 Stage 5 最大的索引差异

Stage 5 Pure-ID Two-Tower：

```text
Train-Unseen candidate
没有独立 ID
→ 无法建立可靠专属向量
→ 不进入 Pure-ID FAISS index
```

Stage 6 `I1/I2/I3/E1`：

```text
Train-Unseen candidate
虽然 ID = UNK
但拥有 Side / MM
→ 可以生成专属 non-ID representation
→ 应进入 Enhanced FAISS index
```

因此 Enhanced Two-Tower 的正式索引目标应是：

$$
815232
$$

个正式 evaluation candidates，除非该 candidate 经审计后完全无法形成有限有效向量。

必须输出：

```text
indexed_candidate_count
seen_indexed_count
unseen_indexed_count
zero_vector_count
nan_inf_count
duplicate_vector_count
```

不要因为沿用了 Stage 5 FAISS builder，就继续偷偷排除所有 Unseen。

这是 Stage 6 最重要的工程检查之一。

---

# 24. Dense + Sparse 参数优化

Stage 5 只有巨大 ID Embedding，使用 `SparseAdam`。

Stage 6 增加：

```text
Self-Attention
MLP
Gate
MM projection
```

因此不能简单把所有参数都扔给同一个 sparse optimizer。

建议：

```text
Sparse parameters:
- item ID embedding
- side categorical embeddings

→ SparseAdam

Dense parameters:
- SDM attention
- long-short gate
- action/time dense modules
- side projection
- MM projection
- strength gate

→ AdamW
```

一个 batch：

```text
zero_grad(sparse)
zero_grad(dense)

forward
loss.backward()

sparse_optimizer.step()
dense_optimizer.step()
```

检查：

> dense optimizer 中不能混入 sparse-gradient Embedding 参数。

---

# 25. Checkpoint 策略

Stage 6 checkpoint 会比 Stage 5 更大。

继续采用：

```text
best.pt
=
model-only

resume.pt
=
model
+ sparse optimizer
+ dense optimizer
+ RNG state
```

不要每个 epoch 都写多 GB 的完整 resume。

建议：

```text
model-only best:
每次需要保存 best 时写

full resume:
每 3 epoch
+ early stop
+ max epoch
```

必须保持 atomic write（原子写入临时文件再替换），避免中途断电得到损坏 checkpoint。

---

# 26. 不再单纯用 Validation Loss 代表 Recall

Stage 5 已经观察到：

```text
Validation Loss 继续下降
但全量 Recall 没有同步提高
```

所以 Stage 6 不能写：

> “Loss 最低就是召回最好的 checkpoint。”

建议策略：

1. Training 过程中仍用固定 Validation Loss 判断是否进入平台；
2. 保留：
   - best-loss checkpoint；
   - final checkpoint；
   - 最近 1～2 个 plateau checkpoint；
3. 训练结束后对这几个候选 checkpoint 运行正式 Validation Recall；
4. **最终 checkpoint 只按 Validation Recall@100 选择**；
5. 选择完成后冻结；
6. 再跑 Test。

模型选择指标：

$$
\boxed{
Validation\ Recall@100
}
$$

分组 Recall 用于归因，不用于偷偷针对 Test 调参。

---

# 27. Stage 6.8：统一离线评估

所有 Stage 6 模型必须沿用 Stage 5：

```text
Recall@10
Recall@50
Recall@100
Recall@500

NDCG@10
NDCG@50
NDCG@100
```

其中 NDCG（Normalized Discounted Cumulative Gain，归一化折损累计增益）用于辅助观察正确广告排得是否更靠前。

---

## 27.1 Recall@K

每个样本只有一个 target 时：

$$
Recall@K
=
\frac{
\sum_{u}
\mathbb 1
(
rank_u(target_u)\le K
)
}{
N
}
$$

必须保持 Unseen target 在 denominator（分母）中。

不能：

```text
模型召不出来
→ 把该样本从分母删掉
```

否则结果会虚高。

---

## 27.2 分组

严格使用 Stage 3/4：

$$
\text{Unseen}:n_i^{train}=0
$$

$$
\text{Tail}:1\le n_i^{train}\le2
$$

$$
\text{Mid}:3\le n_i^{train}\le22
$$

$$
\text{Head}:n_i^{train}>22
$$

不要 Stage 6 自己重新算 percentile。

---

# 28. Stage 6 每一步必须回答的归因问题

## U1

$$
U1-B0
$$

回答：

> 长短期兴趣建模本身是否优于历史 ID 简单平均？

## U2

$$
U2-U1
$$

回答：

> 显式区分行为是否有额外收益？

## U3

$$
U3-U2
$$

回答：

> 真实时间信息是否有额外收益？

## I1

$$
I1-U3
$$

回答：

> Side 是否帮助 Item 表示？

## I2

$$
I2-U3
$$

回答：

> MM 是否帮助 Item 表示？

## I3

$$
I3-\max(I1,I2)
$$

回答：

> Side 与 MM 组合后是否存在互补？

## E1

$$
E1-I3
$$

回答：

> History-strength-aware 是否比固定融合更合理？

---

# 29. 特别关注 Tail / Unseen

Stage 6 的 Item Tower 不能只看 Overall。

尤其记录：

```text
Tail Recall@100
Unseen Recall@100
```

Stage 5 Pure-ID：

$$
Recall@100_{\text{Tail}}=0
$$

$$
Recall@100_{\text{Unseen}}=0
$$

因此 `I1/I2/I3/E1` 的关键问题之一就是：

$$
\boxed{
\text{非 ID 特征能否首次产生真正的 Unseen Recall}
}
$$

但不要把“Unseen 从 0 变成非 0”自动解释为成功。

还需要同时看：

```text
Overall
Head
Mid
Tail
```

防止：

```text
Unseen 上升很多
但 Head 大幅崩掉
```

---

# 30. Stage 6.9：与 I2I 做互补性分析

Enhanced Two-Tower 冻结后，再与 Stage 5 最优：

```text
I2I ItemCF = click3_recent20
```

比较。

对每个 K 输出：

```text
both_hit
i2i_only_hit
enhanced_only_hit
neither_hit
oracle_union_recall
enhanced_incremental_recall
```

理论命中并集：

$$
Recall^{oracle}_{union}@K
=
\frac{
|\text{hit}_{I2I}\cup\text{hit}_{Enhanced}|
}{
N
}
$$

注意命名：

```text
oracle_union
```

或：

```text
ideal_hit_union
```

不能直接叫：

```text
fusion_recall
```

因为这还没有解决两路候选排序问题。

---

# 31. Stage 6.10：真实多路召回融合

只有完成互补性分析后再做。

输入：

```text
I2I Top-N
Enhanced Two-Tower Top-N
```

第一版不要训练复杂融合模型。

采用 RRF（Reciprocal Rank Fusion，倒数排名融合）。

---

## 31.1 RRF

某广告 $i$ 在 I2I 和 Enhanced 两路中的 rank 为：

$$
r_{i,I2I}
$$

和：

$$
r_{i,TT}
$$

融合分数：

$$
score_{fusion}(i)
=
\frac{\alpha}{c+r_{i,I2I}}
+
\frac{1-\alpha}{c+r_{i,TT}}
$$

其中：

$$
c=60
$$

作为固定平滑常数第一版使用。

只在 Validation 上搜索很小的：

$$
\alpha\in
\{0,0.25,0.5,0.75,1\}
$$

然后冻结。

不要在 Test 看完结果以后再改 $\alpha$。

---

## 31.2 为什么优先 rank-based fusion

I2I 分数和 Two-Tower cosine score：

```text
数值范围
分布
尺度
```

不一致。

直接：

$$
score_{I2I}+score_{TT}
$$

没有明确意义。

RRF 只依赖各自 rank，更适合作为第一版工程融合 baseline。

---

# 32. 融合必须报告两种指标

## 32.1 Candidate Union Upper Bound

回答：

> 两路本身理论上能覆盖多少？

## 32.2 Actual Fused Top-K

回答：

> 真正把两路候选压回固定 Top-K 后，最终能留下多少正确目标？

因此：

$$
Recall_{fused}@100
\le
Recall^{oracle}_{union}@100
$$

通常应成立。

如果实际 fused 反而比 oracle union 高，说明计算或定义有 bug。

---

# 33. 多路融合后的分组分析

最终必须再次输出：

```text
Overall
Head
Mid
Tail
Unseen
```

因为有可能：

```text
I2I 更擅长 Head/Mid/Tail
Enhanced 更擅长 Unseen / 非协同部分
```

融合的价值应由分组解释，而不只是一个 Overall 数字。

---

# 34. HNSW / FAISS 检索

FAISS（Facebook AI Similarity Search，向量相似度检索库）继续使用：

```text
HNSW Flat Inner Product
```

HNSW（Hierarchical Navigable Small World，分层可导航小世界图）参数尽量继承 Stage 5：

```text
M = 32
efConstruction = 200
efSearch = 512
```

但由于 Stage 6 index 从约 22.9 万 Train-Seen candidate 扩大到最多 81.5 万 candidate，必须重新做 exact-vs-HNSW accuracy audit。

至少：

```text
1000 validation queries
K = 10/50/100/500
```

重点保证：

```text
mean recall@100 >= 0.95
```

若不达标：

> 先调 HNSW search 参数，不要误判为模型 Recall 差。

---

# 35. Stage 6 单元测试最低要求

## 35.1 Contract

- Stage 3/4 protocol 不一致必须失败；
- candidate count 不一致必须失败；
- Stage 6 不得写 Stage 5 目录。

## 35.2 Sequence

- item/action/timestamp 长度始终一致；
- history 事件严格早于 target；
- session boundary 正确；
- padding mask 正确；
- truncation 不改变末端短期 session 顺序。

## 35.3 Action

- PAD / Exposure / Click / Unknown token 不混；
- PAD event 不贡献用户表示。

## 35.4 Time

- recency 必须大于 0；
- gap 不得为负；
- Train-only normalization stats；
- Validation/Test 不允许重新 fit。

## 35.5 Side

- 13 个字段顺序严格来自 manifest；
- Missing 与 OOV 不混；
- `candidate.cold_start` 不进模型；
- `retrieval_id` 不作为模型 ID。

## 35.6 MM

- 固定 32 维；
- NaN/Inf 失败；
- missing zero vector 必须同时带 `valid=false`；
- candidate row alignment 不得用 retrieval_id 直接当 numpy 行号。

## 35.7 Strength

- count 来自 Train-only；
- Unseen gate 必须为 0；
- 单调 gate 对更大的 count 不应给出更小 ID 权重。

## 35.8 Retrieval

- ID-only variant 不得伪造 Unseen 专属 ID；
- Side/MM variant 必须能 index Train-Unseen candidate；
- zero/nan item vector 必须审计。

## 35.9 Fusion

- duplicate candidate 正确去重；
- 同一广告两路命中时正确累加 rank score；
- Test 不得参与 weight selection；
- oracle union 不得小于任一单路 Recall。

---

# 36. Smoke Test 顺序

先 Debug：

```bat
conda activate tencent_rec

python -X utf8 -m unittest discover -s tests\stage6 -v

python -X utf8 scripts\stage6\stage6_0_contract_audit.py --config configs\stage6.yaml --debug

python -X utf8 scripts\stage6\stage6_1_sequence_adapter_smoke.py --config configs\stage6.yaml --debug

python -X utf8 -u scripts\stage6\stage6_2_train_user_variants.py --config configs\stage6.yaml --debug

python -X utf8 -u scripts\stage6\stage6_3_train_item_variants.py --config configs\stage6.yaml --debug

python -X utf8 -u scripts\stage6\stage6_4_train_enhanced_two_tower.py --config configs\stage6.yaml --debug

python -X utf8 -u scripts\stage6\stage6_5_build_indexes.py --config configs\stage6.yaml --debug

python -X utf8 -u scripts\stage6\stage6_6_evaluate_variants.py --config configs\stage6.yaml --debug

python -X utf8 scripts\stage6\stage6_7_compare_ablation.py --config configs\stage6.yaml --debug

python -X utf8 scripts\stage6\stage6_8_channel_complementarity.py --config configs\stage6.yaml --debug

python -X utf8 scripts\stage6\stage6_9_fuse_recall.py --config configs\stage6.yaml --debug
```

Debug 只验证链路。

Debug 数字不能写进项目结论。

---

# 37. Full Run 推荐顺序

不要一次让 Codex 跑完全部 Stage 6。

## Step A：合同 + 数据适配

```bat
python -X utf8 scripts\stage6\stage6_0_contract_audit.py --config configs\stage6.yaml
python -X utf8 scripts\stage6\stage6_1_sequence_adapter_smoke.py --config configs\stage6.yaml
```

人工审查后再继续。

---

## Step B：User Tower

依次：

```text
U1
U2
U3
```

每完成一个：

```text
Train
↓
Validation retrieval
↓
记录 Recall
↓
再进入下一项
```

不要等三个都跑完以后才发现第一个实现有 bug。

---

## Step C：Item Tower

固定 U3 后：

```text
I1 = ID + Side
I2 = ID + MM
I3 = ID + Side + MM
```

先看 Validation。

---

## Step D：History Strength

只在 `I3` 正确后做：

```text
E1 = I3 + History-strength-aware
```

---

## Step E：最终 Test

当：

```text
User Tower 结构冻结
Item Tower 结构冻结
gate 冻结
checkpoint 冻结
FAISS 参数冻结
```

之后，统一跑 Test。

---

## Step F：I2I + Enhanced

最后：

```text
互补性
↓
oracle union
↓
RRF validation weight
↓
freeze
↓
Test fusion
```

---

# 38. Stage 6 正式输出

建议最终：

```text
artifacts/stage6/

manifests/
├── baseline_freeze.json
├── session_definition.json
├── u1_training_manifest.json
├── u2_training_manifest.json
├── u3_training_manifest.json
├── i1_training_manifest.json
├── i2_training_manifest.json
├── i3_training_manifest.json
├── enhanced_training_manifest.json
└── stage6_manifest.json

metrics/
├── B0.json
├── U1.json
├── U2.json
├── U3.json
├── I1.json
├── I2.json
├── I3.json
├── E1.json
└── fused.json

predictions/
├── i2i_test_hits.parquet
├── enhanced_test_hits.parquet
└── fused_test_rank.parquet

audits/
├── contract.json
├── session_definition.json
├── history_truncation.json
├── time_feature_stats.json
├── side_feature_usage.json
├── mm_usage.json
├── history_strength_gate.json
├── enhanced_candidate_index.json
└── hnsw_accuracy.json

reports/
├── user_tower_ablation.csv
├── item_tower_ablation.csv
├── full_ablation.csv
├── group_recall_comparison.csv
├── channel_complementarity.json
├── fusion_validation.csv
└── stage6_summary.md
```

---

# 39. `stage6_manifest.json` 最低字段

```json
{
  "stage": 6,
  "protocol_version": "stage6_eda_driven_recall_v1",
  "stage3_protocol": "click_target_prefix_v2",
  "stage4_protocol": "stage4_train_only_features_v1",
  "stage5_baseline_frozen": true,

  "user_variants_completed": ["U1", "U2", "U3"],
  "item_variants_completed": ["I1", "I2", "I3", "E1"],

  "test_used_for_model_selection": false,
  "candidate_cold_start_used_as_model_feature": false,
  "retrieval_id_used_as_model_item_id": false,
  "future_item_strength_used": false,

  "unseen_candidates_indexed_with_non_id_features": true,

  "hnsw_accuracy_passed": true,
  "all_stage6_tests_passed": true,
  "fusion_completed": true
}
```

---

# 40. Stage 6 完成标准

Stage 6 只有同时满足以下条件才能打 ✅。

## 工程闭环

- [ ] Stage 5 Baseline 冻结且未被覆盖
- [ ] Stage 3/4 contract 一致
- [ ] U1 / U2 / U3 全量完成
- [ ] I1 / I2 / I3 全量完成
- [ ] E1 Enhanced Two-Tower 完成
- [ ] Enhanced candidate index 支持 Train-Unseen
- [ ] HNSW accuracy 审计通过
- [ ] Validation/Test Recall 完成
- [ ] Head/Mid/Tail/Unseen 分组完成
- [ ] 单路互补性分析完成
- [ ] I2I + Enhanced 实际融合完成
- [ ] Stage 6 单测全部通过
- [ ] leakage audit 全部通过

## 实验闭环

至少能明确回答：

1. SDM 长短期是否有效？
2. Action-aware 是否有效？
3. Time-aware 是否有效？
4. Side 是否有效？
5. MM 是否有效？
6. Side + MM 是否互补？
7. History-strength-aware 是否比固定融合有效？
8. Enhanced Two-Tower 在 Tail/Unseen 上做了什么？
9. Enhanced 与 I2I 是否仍有互补性？
10. 实际多路融合是否获得 Recall 增量？

不要求所有模块一定正收益。

如果某个模块无收益，必须：

```text
保留真实结果
↓
分析原因
↓
不要删除实验假装没做
```

这本身就是项目迭代能力的证据。

---

# 41. Stage 6 明确禁止 Codex 做的事情

## 禁止 1：重新做 Stage 3/4

不要：

```text
重新切 Train/Val/Test
重新造 target
重新 fit Side vocab
重新统计 strength
```

---

## 禁止 2：把 Test 当 Validation

任何：

```text
看 Test
→ 改结构
→ 再 Test
```

都算污染 Test。

---

## 禁止 3：直接用 candidate `cold_start`

Stage 4 已明确：

```text
candidate_cold_start_used_as_model_feature = false
```

Stage 6 保持。

History Strength 只由：

$$
n_i^{train}
$$

决定。

---

## 禁止 4：给 Unseen 偷一个独立 ID

当：

$$
n_i^{train}=0
$$

仍必须：

```text
ID → UNK
```

Unseen 能被召回，必须来自：

```text
Side
MM
```

而不是未来 ID 泄漏。

---

## 禁止 5：把所有增强一次性加上

必须保留：

```text
B0
U1
U2
U3
I1
I2
I3
E1
```

否则不接受 Stage 6 完成。

---

## 禁止 6：提前做排序模型

Stage 6 是召回。

不要引入：

```text
CTR ranker
Cross Feature
DIN
DeepFM
XGBoost ranker
```

---

## 禁止 7：只汇报 Overall

必须：

```text
Overall
Head
Mid
Tail
Unseen
```

---

# 42. Codex 每完成一个小阶段必须生成的“实验说明”

每个 variant 都生成一个：

```text
artifacts/stage6/reports/<variant>_experiment.md
```

固定四段：

## 问题

例如：

> Stage 5 用户历史只做平均，可能无法区分长期兴趣和当前短期兴趣。

## 假设

> 使用 SDM 风格长短期建模后，用户向量能更准确表达当前兴趣。

## 方案

> 只把 mean pooling 改成 SDM-style encoder，其余训练、Item Tower、负采样和评估保持不变。

## 实验与归因

例如：

```text
Validation Recall@100:
B0 → U1

Head:
...

Mid:
...

Tail:
...

Unseen:
...
```

最后写：

> 当前数据支持 / 不支持该假设。

这样 Stage 8 汇总时不需要重新回忆每次为什么做。

---

# 43. 最终 Stage 6 的项目叙事应该形成

不是：

> 我用了 SDM、Action、Time、多模态、Gate。

而应该是：

```text
Stage 5
Pure-ID Two-Tower 明显弱于 I2I
且 Tail/Unseen 几乎失效
        ↓
用户侧问题：
简单平均历史 ID 无法表达兴趣变化
        ↓
SDM
        ↓
Action-aware
        ↓
Time-aware
        ↓
逐步验证用户表示增强来源
        ↓
物品侧问题：
Tail/Unseen 缺少可靠 ID 学习信号
        ↓
Side
MM
        ↓
验证非 ID 信息是否能补足
        ↓
History-strength-aware
        ↓
根据训练期历史强弱自适应融合
        ↓
Enhanced Two-Tower
        ↓
与 I2I 做互补性分析
        ↓
实际多路召回融合
```

最终形成：

$$
\boxed{
\text{EDA 问题}
\rightarrow
\text{建模假设}
\rightarrow
\text{单变量实验}
\rightarrow
\text{分组归因}
\rightarrow
\text{多路召回融合}
}
$$

---

# 44. Codex 的最终执行要求

给 Codex 的核心要求：

1. **先阅读现有仓库，不重写 Stage 3/4/5。**
2. 优先复用 `src/features/` 和 `src/recall/` 已有 API。
3. 每个脚本先有 Debug/Smoke，再允许 Full Run。
4. 每个新增模型都必须有单元测试。
5. 所有实验参数进入 `configs/stage6.yaml`。
6. 所有输出都写 `artifacts/stage6/`。
7. 所有日志写 `logs/stage6/`。
8. 每个结果必须保存机器可读 JSON/CSV，而不只 print。
9. Test 只能在结构/超参数冻结后运行。
10. 不得为了让结果“好看”删除负结果或改评估口径。
11. 所有英文缩略语在文档首次出现时必须给出全称和中文含义。
12. Full Run 前必须暂停，让人工先审查 Debug artifact。

---

# 45. 建议提交节奏

不要一个 commit 把 Stage 6 全做完。

推荐：

```text
commit 1
Stage 6.0 + 6.1
合同、数据 adapter、session/time smoke

commit 2
U1 SDM

commit 3
U2 Action-aware

commit 4
U3 Time-aware

commit 5
I1/I2/I3 Side/MM

commit 6
E1 History-strength-aware

commit 7
Enhanced FAISS + 全量评估

commit 8
I2I complementarity + fusion

commit 9
Stage 6 final audit + docs
```

这样每次 commit 都可以单独审查：

```text
代码具体怎么实现？
为什么这样做？
结果怎样？
下一步由结果推出什么？
```

---

# 46. 参考依据

## 当前项目仓库

- https://github.com/Bumblebee121121/tencent_results
- `logs/inspect_data.log`
- `artifacts/stage4/manifests/stage4_manifest.json`
- `artifacts/stage4/manifests/multimodal_store_manifest.json`
- `artifacts/stage4/audits/feature_vocab_coverage.csv`
- `artifacts/stage5/itemcf/metrics.json`
- `artifacts/stage5/two_tower/metrics.json`
- `artifacts/stage5/reports/channel_complementarity.json`
- `src/features/feature_store.py`
- `src/features/next_click_dataset.py`
- `src/features/time_features.py`
- `src/models/vanilla_two_tower.py`

## SDM 结构参考

Lv et al., **SDM: Sequential Deep Matching Model for Online Large-scale Recommender System**, CIKM 2019.

本项目只借鉴其：

```text
短期兴趣
长期兴趣
长短期 gated fusion
```

思想，并根据 TencentGR-1M 的 Stage 3/4 协议重新实现，不宣称逐模块完全复现原论文。
