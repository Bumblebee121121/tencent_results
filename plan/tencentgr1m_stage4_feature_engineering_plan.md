# TencentGR-1M 判别式推荐项目
# Stage 4：特征工程计划书（供 Codex 编写代码）

> 目标：在 Stage 3 已固定的 **next-click 样本、时间切分、候选池与评估协议** 之上，建立一套可被 Stage 5～Stage 7 复用的统一特征层。  
> 原则：**Model-aware，但不 Model-locked**。Stage 4 决定“给模型什么信息、如何无泄漏地表示和读取”，不决定 Transformer 层数、Embedding 维度、SDM 结构、Fusion Gate 结构等模型超参数。

---

## 0. 当前项目阶段

```text
阶段 0：项目与任务定义                     ✅
        ↓
阶段 1：数据理解与数据预处理               ✅
        ↓
阶段 2：EDA 与问题发现                     ✅
        ↓
阶段 3：样本构造、时间切分与评估协议       ✅
        ↓
阶段 4：特征工程                           ← CURRENT
        ↓
阶段 5：召回 Baseline
        │
        ├── ItemCF / I2I
        └── Vanilla Two-Tower
        ↓
阶段 6：EDA-driven 召回模型迭代
        │
        ├── User Tower
        │     SDM 长短期兴趣
        │       ↓
        │     + Action-aware
        │     + Time-aware
        │
        ├── Item Tower
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

Stage 4 **不训练推荐模型**，也**不重新定义 Stage 3 的样本和切分**。

---

# 1. Stage 4 的输入依据

Codex 在实现之前，必须先阅读并以这些文件为准，不允许凭经验自行猜 TencentGR-1M 的字段语义。

## 1.1 原始数据结构

重点阅读：

```text
logs/inspect_data.log
logs/inspect_indexer.log
logs/inspect_mapping_examples.log
```

当前已经确认的数据结构如下。

### `seq`

```text
user_id: int64
seq: list<
    struct<
        item_id: int64,
        action_type: int32,
        timestamp: int64
    >
>
```

这里的 `seq.item_id` 是历史 RID 空间中的广告 ID。

每个序列 token 是：

$$
e_k=(i_k,a_k,t_k)
$$

其中：

- $i_k$：历史广告 RID；
- $a_k$：行为反馈，当前合法值为 `0/1/null`；
- $t_k$：秒级时间戳。

**不要重新解释成“曝光表 + 点击表”两条日志。**

---

### `user_feat`

共有 8 个匿名字段。

单值字段：

```text
103
104
105
109
```

多值 List 字段：

```text
106
107
108
110
```

当前 EDA 显示多值字段长度都较短，但 Stage 4 不应假设它们具有某种业务语义。

---

### `item_feat`

共有 13 个匿名结构化字段：

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

均为 `int64` 单值字段，但：

- 基数差异很大；
- 缺失率差异很大；
- `115` 缺失尤其明显；
- `121`、`102`、`122` 等字段基数很高。

Stage 4 只建立可复用编码接口，**不因为“高基数”就直接决定 Hashing，也不在此阶段决定 Embedding 维度**。

---

### `candidate`

官方 candidate 共 660000 条，包含：

```text
item_id
retrieval_id
100 ... 122
```

每个结构化字段不是普通标量，而是：

```text
struct<
    cold_start: int64,
    feature_value: string
>
```

因此，candidate 的 `feature_value` 与 `item_feat` 的匿名字段在“字段编号”上对应，但**物理存储类型不同**。

Stage 4 必须先做一致性审计，不能直接假设：

```text
candidate["102"].feature_value
==
item_feat["102"]
```

---

### `mm_emb_81_32`

```text
anonymous_cid: string
emb: list<double>
```

多模态向量维度固定为：

$$
d_{\mathrm{MM}}=32
$$

`anonymous_cid` 是 OID 字符串，需要经过类型规范化后和 item OID 对齐。

---

## 1.2 ID 映射事实

`indexer.pkl` 至少包含：

```text
indexer["i"]    # OID -> RID
indexer["u"]    # 原始 user -> RID
indexer["a"]    # action 映射
indexer["f"]    # 各匿名 feature namespace 的映射
```

当前规模约为：

```text
item mapping: 4,783,154
user mapping: 1,001,845
action mapping: 2
feature namespaces: 22
```

但是 Stage 4 **不得把所有现有 indexer ID 直接等同于“训练期可学习 ID”**。

原因见后文的 ID Contract。

---

# 2. Stage 3 不可破坏的合同

Stage 4 必须消费 Stage 3 的结果，而不是重新定义它们。

重点读取：

```text
configs/stage3.yaml

artifacts/stage3/samples/sample_manifest.json
artifacts/stage3/splits/split_manifest.json
artifacts/stage3/candidates/eval_candidate_manifest.json
artifacts/stage3/item_strength/item_strength_thresholds.json
artifacts/stage3/evaluation/evaluation_protocol.json
```

---

## 2.1 Target 定义

正式 target 固定为：

```text
action_type == 1 interaction
```

对于 target 时刻 $T$，历史必须满足：

$$
H_u(T)=\{e_k:t_k<T\}
$$

也就是：

```text
seq[:history_end_position]
```

Stage 4 不允许：

- 把 target 自己加入历史；
- 使用 target 之后的事件；
- 改成 next-exposure；
- 改成任意 next-item；
- 重做标签归因。

---

## 2.2 时间切分

Stage 3 当前正式切分：

```text
Train:      80%
Validation: 10%
Test:       10%
```

并且相同 timestamp 不跨 split。

训练期 raw event cutoff 必须直接从：

```text
artifacts/stage3/splits/split_manifest.json
```

读取。

Stage 4 不允许硬编码 cutoff。

---

## 2.3 Primary Evaluation

正式主评估使用：

```text
samples/val_primary.parquet
samples/test_primary.parquet
```

每条记录只有一个 ground truth。

正式候选池：

```text
candidates/eval_candidates.parquet
```

由：

```text
official candidates
∪ validation targets
∪ test targets
```

组成。

不要在 Stage 4 改动候选池。

---

## 2.4 Train-only History Strength

训练强度定义必须继续使用：

$$
n_i^{\mathrm{train}}
=
\sum_{e\in \mathcal D,\ t_e<T_{\mathrm{train\ cutoff}}}
\mathbf 1(i_e=i)
$$

正式分组：

$$
\text{Unseen}: n_i^{\mathrm{train}}=0
$$

$$
\text{Tail}:0<n_i^{\mathrm{train}}\le P_{50}^{\mathrm{train}}
$$

$$
\text{Mid}:P_{50}^{\mathrm{train}}<n_i^{\mathrm{train}}\le P_{90}^{\mathrm{train}}
$$

$$
\text{Head}:n_i^{\mathrm{train}}>P_{90}^{\mathrm{train}}
$$

当前 Stage 3 得到：

$$
P_{50}^{\mathrm{train}}=2,\qquad
P_{90}^{\mathrm{train}}=22
$$

**Stage 4 不重新拟合 Head/Mid/Tail 阈值。**

---

# 3. Stage 4 的核心目标

Stage 4 最终要提供 6 类信息。

```text
① User History
   item / action / timestamp

② User Static Side Features
   103~110

③ Item ID Signal
   严格区分 train-seen 与 train-unseen

④ Item Structured Side Features
   100~122

⑤ Item Multimodal Feature
   32D MM + valid mask

⑥ Train-only History Strength
   item_train_count / strength group
```

对应后续模型路线：

| Stage 4 信息 | 后续用途 |
|---|---|
| History Item | Vanilla Two-Tower、SDM、ItemCF |
| History Action | Action-aware User Tower |
| History Timestamp | Time-aware User Tower |
| User Side | 后续召回/排序可选输入 |
| Item ID | Vanilla Item Tower |
| Item Side | Side-enhanced Item Tower、排序 |
| MM | Multimodal Item Tower、排序 |
| Train Count | History-strength-aware Fusion |
| Missing/OOV Mask | 防止把缺失值误当正常类别/正常 MM |

最终特征层应该满足：

$$
q_u=f(H_u,x_u)
$$

以及：

$$
v_i=g(i,x_i^{\mathrm{side}},x_i^{\mathrm{MM}},n_i^{\mathrm{train}})
$$

但 Stage 4 **不实现** $f(\cdot)$ 和 $g(\cdot)$ 的具体神经网络。

---

# 4. 最重要的 ID Contract

这是 Stage 4 第一优先级的工程约束。

TencentGR-1M 当前至少同时存在 4 种“Item ID”。

| 名称 | 含义 | 是否可以直接做 learned ID Embedding |
|---|---|---:|
| `item_oid` | 原始匿名广告 ID，跨 candidate/MM 对齐 | ❌ 不直接使用 |
| `item_rid` | 历史 indexer 映射后的 ID | △ 仅作为 join key |
| `retrieval_id` | evaluation candidate-local 编号 | ❌ 严禁作为 learned item ID |
| `model_item_token` | Stage 4 新建的训练期模型 ID | ✅ |

---

## 4.1 为什么不能直接使用 `retrieval_id`

`retrieval_id` 的作用是：

```text
evaluation candidate pool 内部索引
```

它不是训练历史中的 Item Embedding ID。

因此禁止：

```python
item_embedding(retrieval_id)
```

作为 Vanilla Two-Tower 的广告 ID 表示。

---

## 4.2 Train-Unseen 必须映射到共享 UNK

即使一个广告存在于全局 `indexer["i"]` 中，只要：

$$
n_i^{\mathrm{train}}=0
$$

那么从当前训练协议看，它就没有训练期 ID 学习信号。

因此 Vanilla ID 分支必须满足：

$$
\operatorname{IDToken}(i)=
\begin{cases}
\operatorname{SeenToken}(i), & n_i^{\mathrm{train}}>0\\
\operatorname{UNK}, & n_i^{\mathrm{train}}=0
\end{cases}
$$

官方 candidate 中 `item_rid=null` 的广告也必须映射到 `UNK`。

---

## 4.3 推荐的特殊 Token

### Item ID

```text
0 = PAD_ITEM
1 = UNK_ITEM
2+ = train-seen item token
```

为了避免额外构建巨型 Python Dict，可以优先采用：

```text
train-seen RID -> RID + offset
train-unseen RID -> UNK_ITEM
RID is null -> UNK_ITEM
```

前提是 Stage 4.1 先验证 RID 的取值范围和唯一性。

---

### Action

不要直接把 `null` 当 exposure。

建议：

```text
0 = PAD_ACTION
1 = EXPOSURE_ACTION
2 = CLICK_ACTION
3 = UNKNOWN_ACTION
```

满足：

```text
raw 0    -> EXPOSURE_ACTION
raw 1    -> CLICK_ACTION
raw null -> UNKNOWN_ACTION
padding  -> PAD_ACTION
```

---

### Side Categorical Feature

每一个 feature namespace 独立编码。

建议保留：

```text
0 = PAD
1 = MISSING
2 = OOV
3+ = train-known feature value
```

必须区分：

```text
缺失值 MISSING
```

和：

```text
训练期没见过的合法新值 OOV
```

---

# 5. 不使用 candidate.cold_start 作为主模型特征

candidate 每个匿名字段里存在：

```text
cold_start
```

但是当前项目已经有自己的严格时间切分。

因此正式的历史强度必须来自：

```text
Stage 3 train-only item count
```

而不是直接使用：

```text
candidate.<feature>.cold_start
```

Stage 4 的默认策略：

```text
candidate.cold_start
→ 保留在 audit / metadata
→ 不进入 Vanilla Two-Tower
→ 不进入 Enhanced Two-Tower 的 history-strength gate
```

除非后续官方文档能够明确证明其业务语义并且确定不存在协议冲突，再单独做附加实验。

**Codex 不要擅自把它变成模型输入。**

---

# 6. Stage 4 子任务设计

---

# Stage 4.1：Feature Contract 与跨表一致性审计

## 目的

在真正编码之前，先把：

```text
seq
user_feat
item_feat
candidate
mm_emb
indexer
Stage 3 artifacts
```

之间的 ID 和字段关系一次性确认清楚。

## 新增脚本

```text
scripts/stage4/stage4_1_feature_contract_audit.py
```

## 必做检查

### A. Schema 校验

确认：

```text
seq:
  user_id
  item_id
  action_type
  timestamp

user_feat:
  103~110

item_feat:
  100,101,102,112,114,115,116,117,118,119,120,121,122

candidate:
  item_id
  retrieval_id
  100~122 struct

mm:
  anonymous_cid
  emb
```

字段缺失时直接 fail-fast。

---

### B. RID/OID/retrieval_id 审计

输出：

```text
RID min/max
RID unique count
candidate OID -> RID coverage
eval candidate RID null count
retrieval_id uniqueness
retrieval_id range
```

并明确写入 manifest：

```text
retrieval_id_is_model_embedding_id = false
```

---

### C. candidate 与 item_feat 的同字段一致性

对：

```text
candidate OID 可映射 RID
```

的重叠广告，逐字段比较：

```text
candidate[field].feature_value
vs
item_feat[field]
```

需要先规范类型，再计算每个字段：

```text
both_non_null_count
equal_count
mismatch_count
match_ratio
candidate_null_item_non_null
candidate_non_null_item_null
```

输出：

```text
artifacts/stage4/audits/candidate_item_feature_consistency.csv
```

**不要提前假定 100% 一致。**

如果存在 mismatch：

1. 不自动覆盖；
2. 报告具体字段；
3. 在 manifest 中保存；
4. 使用后文规定的 deterministic source precedence。

---

### D. indexer feature namespace 类型审计

对 `indexer["f"]` 的相关 namespace 检查：

```text
key type
value type
size
```

尤其确认 candidate `feature_value: string` 应如何规范化后进入同一 namespace。

---

### E. MM 审计复用

确认：

```text
emb dim == 32
NaN/Inf == 0
wrong dimension == 0
```

不要重新解释 MM 的业务含义。

---

# Stage 4.2：构建 Train-only Item ID / Strength 基础层

## 目的

为：

```text
Vanilla Two-Tower ID Branch
History-strength-aware Fusion
Train-Unseen 处理
```

提供统一基础变量。

## 新增脚本

```text
scripts/stage4/stage4_2_build_train_item_base.py
```

## 输入

必须从 Stage 3 读取：

```text
train_raw_event_cutoff_exclusive
P50/P90
```

禁止自己重新算 cutoff。

---

## 4.2.1 构建全历史 RID 对齐的 train count

Stage 3 当前正式 `item_train_counts.parquet` 主要服务 evaluation candidate pool。

Stage 4 为了支持 **所有 Train target / history item**，需要再构建一个 RID 对齐的完整 count store：

```text
train_item_count_by_rid
```

统计定义必须与 Stage 3 完全一致：

$$
n_i^{\mathrm{train}}
=
\sum_{e:t_e<T_{\mathrm{train\ cutoff}}}
\mathbf 1(i_e=i)
$$

建议保存：

```text
artifacts/stage4/mappings/train_item_count_by_rid.npy
```

数据类型优先：

```text
int32
```

如果最大频次超出范围再升级 `int64`。

---

## 4.2.2 与 Stage 3 做一致性测试

对 evaluation candidate pool 中能够映射 RID 的广告：

```text
Stage4 full train count
```

必须与：

```text
Stage3 item_train_counts.parquet
```

逐项一致。

若不一致：

```text
Stage 4 立即失败
```

不要“以 Stage 4 为准”。

---

## 4.2.3 构建 model item token

输出：

```text
rid_to_model_item_token.npy
```

规则：

```text
PAD -> 0
train-unseen -> 1
train-seen RID -> stable seen token
```

同时提供函数：

```python
encode_item_rid(rid, train_count) -> model_item_token
encode_candidate(item_oid, item_rid) -> model_item_token
```

---

## 4.2.4 暴露 History Strength 特征

至少提供：

```text
item_train_count
item_train_count_log1p
item_strength_group
```

其中：

$$
c_i=\log(1+n_i^{\mathrm{train}})
$$

Stage 6 的 gate 以后可以消费 $c_i$，但 Stage 4 **不实现 gate**。

---

# Stage 4.3：User Static Feature Store

## 目的

把 `user_feat` 变成可被召回和排序复用的稳定输入，但不提前决定 pooling/Embedding 结构。

## 新增脚本

```text
scripts/stage4/stage4_3_build_user_features.py
```

## 单值字段

```text
103
104
105
109
```

输出每个字段的：

```text
encoded_token
missing_mask
```

---

## 多值字段

```text
106
107
108
110
```

必须保留：

```text
list of encoded tokens
missing_mask
```

Stage 4 **不要决定**：

```text
mean pooling
sum pooling
attention pooling
```

这些属于模型阶段。

也不要因为 EDA 中 list 很短，就把它们错误地当成单值字段。

---

## Vocabulary Fit Scope

所有 categorical namespace 的“训练期已知值”必须基于 train scope 定义。

推荐：

```text
User feature:
使用训练期存在有效历史/训练样本的用户确定 train-known value

Item feature:
使用 train-seen item 确定 train-known value
```

验证/测试中第一次出现的 category：

```text
→ OOV
```

缺失：

```text
→ MISSING
```

---

## 输出

推荐：

```text
artifacts/stage4/feature_store/user_features.parquet
```

字段示意：

```text
user_id
f103_token
f103_missing
...
f106_tokens
f106_missing
...
```

同时输出：

```text
artifacts/stage4/manifests/user_feature_manifest.json
```

记录：

```text
row_count
field_type
train_known_vocab_size
missing_count
OOV coverage
list length distribution
```

---

# Stage 4.4：统一 Item Structured Side Feature Store

这是 Stage 4 的重点之一。

## 目的

统一处理：

```text
历史广告：item_feat
官方 candidate：candidate.feature_value
added val/test targets：item_feat
```

使后续 Item Tower 不需要自己理解三套来源。

## 新增脚本

```text
scripts/stage4/stage4_4_build_item_side_features.py
```

---

## 4.4.1 训练/历史广告

对于拥有 RID 的历史广告：

```text
source = item_feat
key = RID
```

构建：

```text
item_side_tokens_by_rid
item_side_missing_mask_by_rid
```

建议使用可 memory-map 的二维数组：

```text
[N_RID + special_rows, 13]
```

数据类型：

```text
int32
```

---

## 4.4.2 Evaluation Candidate

对 Stage 3：

```text
eval_candidates.parquet
```

逐广告构建统一 13 字段 Side Feature。

推荐 source precedence：

### 情况 A：属于 official candidate

优先：

```text
candidate[field].feature_value
```

因为这是官方 candidate 在预测时直接提供的特征。

### 情况 B：Stage 3 为保证 target coverage 新加入的 val/test target

若有 RID：

```text
item_feat[field]
```

### 情况 C：两者都存在

不要静默混合。

保存：

```text
source_kind
```

用于 audit，但 `source_kind` 默认不进入模型。

---

## 4.4.3 Missing / OOV

每个字段都必须输出：

```text
token
missing_mask
oov_mask
```

严禁：

```text
null -> 0
```

然后不告诉模型 0 是缺失还是正常值。

---

## 4.4.4 输出

```text
artifacts/stage4/feature_store/item_side_tokens_by_rid.npy
artifacts/stage4/feature_store/item_side_missing_by_rid.npy

artifacts/stage4/feature_store/eval_candidate_side.parquet
```

候选表至少包含：

```text
retrieval_id
item_oid
item_rid
model_item_token

f100_token ... f122_token
f100_missing ... f122_missing
f100_oov ... f122_oov

item_train_count
item_train_count_log1p
strength_group

source_kind
```

`source_kind` 仅审计，不作为默认模型特征。

---

# Stage 4.5：Multimodal Feature Store

## 目的

为 Item Tower 提供：

```text
32D MM vector
+
valid mask
```

而不是简单把缺失 MM 当作真实全零向量。

## 新增脚本

```text
scripts/stage4/stage4_5_build_multimodal_store.py
```

---

## 4.5.1 统一 dtype

原始：

```text
list<double>
```

训练存储转换为：

```text
float32
```

维度必须保持：

$$
x_i^{\mathrm{MM}}\in\mathbb R^{32}
$$

---

## 4.5.2 历史 RID 对齐

构建：

```text
mm_by_rid.npy
mm_valid_by_rid.npy
```

对于有效 MM：

```text
mask = 1
```

对于缺失 MM：

```text
vector = zeros(32)
mask = 0
```

这里的零向量只是**存储占位符**。

模型必须同时拿到 mask。

---

## 4.5.3 Eval Candidate OID 对齐

另外构建：

```text
eval_candidate_mm.npy
eval_candidate_mm_valid.npy
```

严格按照：

```text
eval_candidates.parquet
```

的候选顺序对齐。

不要依赖 RID，因为 official history-unseen candidate 的 RID 可以为 null。

---

## 4.5.4 Stage 4 不做的事情

暂时不要：

```text
训练 MM encoder
PCA
learned projection
与 ID concat
Fusion Gate
```

这些属于 Stage 6。

也不要默认 L2 normalize 后覆盖原始向量。

如后续模型需要归一化，应在模型/配置层明确做。

---

# Stage 4.6：Sequence Feature Store

这是 User Tower 的基础。

## 目的

高效支持：

```text
Vanilla history encoder
SDM long/short
Action-aware
Time-aware
Ranking
```

同时避免为 868 万条样本重复物化完整 history。

## 新增脚本

```text
scripts/stage4/stage4_6_build_sequence_store.py
```

---

## 4.6.1 不要按 sample 重复保存 history list

Stage 3 已经采用：

```text
materialize_history = false
```

并保存：

```text
user_id
history_end_position
history_length
target_timestamp
```

Stage 4 必须继续复用这个设计。

禁止生成：

```text
868 万条 sample
×
每条复制一份最长约 100 的 history
```

这会制造大量重复存储。

---

## 4.6.2 推荐 CSR / Memmap 式序列存储

把原始每用户 list 序列展平：

```text
user_seq_offsets.npy
seq_item_rid.npy
seq_action_token.npy
seq_timestamp.npy
```

其中：

```text
user_seq_offsets[user]
```

定位用户序列起点。

某个 sample 的历史仍由：

```text
user_id
history_end_position
```

切片得到。

示意：

```python
start = user_seq_offsets[user_id]
end = start + history_end_position

hist_item_rid = seq_item_rid[start:end]
hist_action = seq_action_token[start:end]
hist_timestamp = seq_timestamp[start:end]
```

---

## 4.6.3 Sequence Item 同时暴露 RID 和 Model Token

建议不要只保存 model token。

保留：

```text
seq_item_rid
```

然后由：

```text
rid_to_model_item_token
```

得到：

```text
seq_item_token
```

这样 Stage 6 如果需要通过 RID 读取 Side/MM，仍然可以做到。

---

## 4.6.4 Action 对齐

必须始终满足：

$$
|I_u|=|A_u|=|T_u|
$$

并且同一位置：

$$
(i_k,a_k,t_k)
$$

来自同一个原始 sequence element。

---

# Stage 4.7：Time Feature Utility

## 目的

为 Stage 6 的 Time-aware 实验保留统一、无歧义的时间变量。

## 新增模块

```text
src/features/time_features.py
```

Stage 4 不需要提前决定最终用 Bucket Embedding 还是 MLP，只提供基础量。

---

## 4.7.1 Recency

对于 target 时刻 $T$：

$$
r_k=T-t_k
$$

必须满足：

$$
r_k>0
$$

因为 Stage 3 已保证：

$$
t_k<T
$$

可额外提供连续变换：

$$
\tilde r_k=\log(1+r_k)
$$

---

## 4.7.2 Inter-event Gap

$$
\Delta t_k=t_k-t_{k-1}
$$

并提供：

$$
\widetilde{\Delta t}_k
=
\log(1+\Delta t_k)
$$

第一个历史事件可以定义：

```text
gap = 0
```

并配套：

```text
first_event_mask
```

---

## 4.7.3 为什么 Recency 不应全局预计算

同一个历史事件 $t_k$ 对不同 target $T_1,T_2$：

$$
T_1-t_k\neq T_2-t_k
$$

因此：

```text
recency
```

应该在 Dataset/Collator 根据当前 sample 的 target timestamp 动态计算。

不要给每个用户事件永久保存一个唯一 recency。

---

## 4.7.4 Time Bucket

Stage 4 只实现通用工具：

```python
bucketize_time(seconds, boundaries)
```

但默认不硬编码最终 bucket 边界。

Stage 6 再根据实验决定：

```text
fixed semantic buckets
or
train-quantile buckets
or
continuous time encoder
```

---

# Stage 4.8：统一 FeatureDataset / FeatureStore API

## 目的

Stage 5～Stage 7 的模型代码不能再各自重新读原始 Parquet、各自重新解释 missing/OOV/ID。

统一通过：

```text
src/features/
```

访问。

## 推荐新增文件

```text
src/features/__init__.py
src/features/id_semantics.py
src/features/categorical_encoder.py
src/features/time_features.py
src/features/sequence_store.py
src/features/item_feature_store.py
src/features/multimodal_store.py
src/features/feature_store.py
src/features/next_click_dataset.py
```

---

## 4.8.1 单条样本建议返回

```python
{
    "sample_id": ...,
    "user_id": ...,

    "target_item_oid": ...,
    "target_item_rid": ...,
    "target_item_token": ...,
    "target_timestamp": ...,

    "hist_item_rid": ...,
    "hist_item_token": ...,
    "hist_action_token": ...,
    "hist_timestamp": ...,

    "hist_length": ...,

    "user_features": ...,

    "target_item_side": ...,
    "target_item_side_missing": ...,
    "target_item_side_oov": ...,

    "target_mm": ...,
    "target_mm_valid": ...,

    "target_train_count": ...,
    "target_train_count_log1p": ...,
    "target_strength_group": ...
}
```

Time-aware 实验启用时，再由 collator 追加：

```text
hist_recency
hist_recency_log1p
hist_time_gap
hist_time_gap_log1p
```

---

## 4.8.2 Stage 4 不负责 Negative Sampling

Stage 4 只提供特征读取。

不要在这里决定：

```text
random negative
in-batch negative
hard negative
popularity negative
```

这些属于 Stage 5/Stage 6 的模型训练协议。

---

# 7. Feature Source Precedence

必须写成代码中的明确规则，不能靠“哪个 join 成功就用哪个”。

## 7.1 User

```text
user_id
→ user_feat
```

---

## 7.2 History Item

```text
RID
→ item_feat
→ mm_by_rid
```

---

## 7.3 Official Candidate

```text
OID
→ candidate side feature_value
→ MM by OID

OID -> RID 成功时：
    RID 仅用于历史映射/Train Count/ID Token
```

---

## 7.4 Added Validation/Test Target

如果不属于 official candidate：

```text
target RID
→ item_feat
→ MM by RID/OID
```

---

# 8. Train-only Vocabulary 与 Leakage Guardrail

Stage 4 的词表/已知类别判断必须遵守：

```text
Split First
↓
Fit Train-known State
↓
Transform Train / Validation / Test
```

禁止：

```text
先扫描全部 validation/test feature values
↓
创建完整 category vocabulary
↓
再训练模型
```

对于 Side Feature：

$$
\operatorname{Token}(x)=
\begin{cases}
\mathrm{MISSING}, & x\text{ 缺失}\\
\mathrm{KnownToken}(x), & x\text{ 在 train scope 出现}\\
\mathrm{OOV}, & x\text{ 合法但 train 未见}
\end{cases}
$$

---

# 9. Stage 4 配置文件

新增：

```text
configs/stage4.yaml
```

建议至少包含：

```yaml
data_root: data/TencentGR-1M
stage3_root: artifacts/stage3
output_root: artifacts/stage4
log_root: logs/stage4

timestamp_unit: seconds
mm_dim: 32
mm_dtype: float32

user_scalar_features:
  - "103"
  - "104"
  - "105"
  - "109"

user_list_features:
  - "106"
  - "107"
  - "108"
  - "110"

item_features:
  - "100"
  - "101"
  - "102"
  - "112"
  - "114"
  - "115"
  - "116"
  - "117"
  - "118"
  - "119"
  - "120"
  - "121"
  - "122"

special_tokens:
  item:
    pad: 0
    unk: 1

  categorical:
    pad: 0
    missing: 1
    oov: 2

  action:
    pad: 0
    exposure: 1
    click: 2
    unknown: 3

use_candidate_cold_start_as_model_feature: false

sequence_store:
  backend: numpy_memmap

materialize_history_per_sample: false
random_seed: 42
```

所有 feature list 都需要在 Stage 4.1 和真实 schema 对照。

配置与 schema 不一致时 fail-fast。

---

# 10. 输出目录建议

```text
artifacts/stage4/
│
├── audits/
│   ├── feature_contract.json
│   ├── candidate_item_feature_consistency.csv
│   ├── feature_vocab_coverage.csv
│   ├── multimodal_alignment.json
│   └── leakage_audit.json
│
├── mappings/
│   ├── action_token_map.json
│   ├── rid_to_model_item_token.npy
│   ├── train_item_count_by_rid.npy
│   └── feature_vocab_manifest.json
│
├── feature_store/
│   ├── user_features.parquet
│   ├── item_side_tokens_by_rid.npy
│   ├── item_side_missing_by_rid.npy
│   ├── eval_candidate_side.parquet
│   │
│   ├── mm_by_rid.npy
│   ├── mm_valid_by_rid.npy
│   ├── eval_candidate_mm.npy
│   ├── eval_candidate_mm_valid.npy
│   │
│   ├── user_seq_offsets.npy
│   ├── seq_item_rid.npy
│   ├── seq_action_token.npy
│   └── seq_timestamp.npy
│
└── manifests/
    └── stage4_manifest.json
```

大型 `.npy/.parquet` 不应提交 GitHub。

GitHub 主要保留：

```text
代码
配置
tests
manifest
小型 audit csv/json
日志
```

---

# 11. 必须新增的单元测试

目录：

```text
tests/stage4/
```

至少包含以下测试。

---

## 11.1 ID Semantics Test

### Test A

```text
train_count > 0
→ model_item_token != UNK
```

### Test B

```text
train_count == 0
→ model_item_token == UNK
```

### Test C

```text
item_rid is null
→ model_item_token == UNK
```

### Test D

断言任何 FeatureDataset 都没有：

```text
retrieval_id -> learned item ID
```

这种路径。

---

## 11.2 Action Test

```text
0    -> EXPOSURE
1    -> CLICK
null -> UNKNOWN
pad  -> PAD
```

四者不能混。

---

## 11.3 Sequence Alignment Test

对于任意 sample：

$$
|I|=|A|=|T|
$$

并且：

$$
\max(T_{\mathrm{history}})<T_{\mathrm{target}}
$$

同时：

```text
len(history) == history_length
```

---

## 11.4 Time Feature Test

验证：

$$
r_k=T-t_k>0
$$

以及：

$$
\Delta t_k\ge 0
$$

并验证 `log1p` 无 NaN/Inf。

---

## 11.5 Categorical Missing/OOV Test

人工构造：

```text
train-known value
validation-only value
null value
```

必须分别编码成：

```text
known token
OOV
MISSING
```

---

## 11.6 List Feature Test

验证：

```text
null list
```

与：

```text
non-null list
```

区分。

同时保证 token 顺序和原始 list 一致，不擅自排序或去重。

---

## 11.7 Candidate vs Item Feature Test

对重叠广告验证：

```text
source normalization
comparison
source precedence
```

当人工制造 mismatch 时，audit 必须发现。

---

## 11.8 Multimodal Test

验证：

```text
valid MM -> 32D float32 + mask=1
missing MM -> zeros(32) + mask=0
```

并确保：

```text
wrong_dim / NaN / Inf
```

会 fail-fast。

---

## 11.9 Stage 3 Count Consistency Test

Stage 4 构建的 full train count 在 evaluation candidate subset 上必须与 Stage 3 完全一致。

这是 Stage 4 最重要的 leakage regression test 之一。

---

## 11.10 Dataset Smoke Test

随机取：

```text
Train 100 samples
Val Primary 100 samples
Test Primary 100 samples
```

验证：

```text
所有必需字段可读取
history slice 正确
target side feature 可读取
MM mask 正确
train-unseen target 的 ID token 为 UNK
```

---

# 12. Stage 4 日志要求

每个脚本都必须使用 Python logging 自动写日志。

目录：

```text
logs/stage4/
```

建议：

```text
stage4_1_feature_contract_audit.log
stage4_2_build_train_item_base.log
stage4_3_build_user_features.log
stage4_4_build_item_side_features.log
stage4_5_build_multimodal_store.log
stage4_6_build_sequence_store.log
stage4_8_feature_dataset_smoke.log
```

不要只依赖终端输出。

每份日志至少输出：

```text
输入路径
输出路径
处理行数
缺失数
OOV 数
映射失败数
耗时
peak/approx memory（能获得则记录）
```

---

# 13. Smoke Test → Full Run 顺序

不要直接全量跑。

## 第一步：单元测试

```bat
python -X utf8 -m unittest discover -s tests\stage4 -v
```

要求：

```text
OK
```

---

## 第二步：Stage 4.1 审计

先只跑审计，不生成大型 store。

```bat
python -X utf8 scripts\stage4\stage4_1_feature_contract_audit.py --config configs\stage4.yaml
```

重点人工检查：

```text
candidate/item_feat 同字段一致性
indexer feature key 类型
RID/retrieval_id 语义
MM 维度
```

如果这里异常，先停止。

---

## 第三步：Debug / Smoke

建议所有 builder 支持：

```text
--debug
--max-users
--max-items
--max-candidates
```

例如：

```bat
python -X utf8 scripts\stage4\stage4_2_build_train_item_base.py --config configs\stage4.yaml --debug
python -X utf8 scripts\stage4\stage4_3_build_user_features.py --config configs\stage4.yaml --debug
python -X utf8 scripts\stage4\stage4_4_build_item_side_features.py --config configs\stage4.yaml --debug
python -X utf8 scripts\stage4\stage4_5_build_multimodal_store.py --config configs\stage4.yaml --debug
python -X utf8 scripts\stage4\stage4_6_build_sequence_store.py --config configs\stage4.yaml --debug
```

然后：

```bat
python -X utf8 scripts\stage4\stage4_8_feature_dataset_smoke.py --config configs\stage4.yaml --debug
```

---

## 第四步：再次跑测试

```bat
python -X utf8 -m unittest discover -s tests\stage4 -v
```

---

## 第五步：全量构建

审查 Debug 结果无误后，再去掉 `--debug`。

建议按：

```text
4.2
↓
4.3
↓
4.4
↓
4.5
↓
4.6
↓
4.8
```

顺序执行。

---

# 14. Stage 4 Manifest

最终必须生成：

```text
artifacts/stage4/manifests/stage4_manifest.json
```

至少记录：

```text
schema_version
feature_protocol_version

stage3_protocol_version
stage3_train_cutoff

user_count
item_rid_count
eval_candidate_count

user_feature_fields
item_feature_fields

special_token_definitions

train_seen_item_count
train_unseen_eval_candidate_count

feature_vocab_sizes
missing_rates
oov_rates

mm_dim
mm_valid_count
mm_missing_count

sequence_event_count
sequence_store_backend

candidate_cold_start_used_as_model_feature: false
retrieval_id_used_as_model_item_id: false
materialize_history_per_sample: false

all_stage4_tests_passed
```

---

# 15. Stage 4 完成标准

只有同时满足以下条件，才能进入 Stage 5。

## 数据合同

- [ ] 没有修改 Stage 3 target 定义。
- [ ] 没有修改 Stage 3 split。
- [ ] 没有修改 Stage 3 candidate pool。
- [ ] 所有 history 都严格早于 target。

## ID

- [ ] OID / RID / retrieval_id / model token 已明确区分。
- [ ] Train-Unseen ID 进入共享 UNK。
- [ ] `retrieval_id` 未被用于 learned ID Embedding。

## Action / Time

- [ ] Exposure / Click / Unknown / PAD 四种 action 明确区分。
- [ ] item/action/timestamp 三序列严格对齐。
- [ ] Recency / Gap 工具已实现并通过测试。

## Side

- [ ] user_feat 单值/多值均可读取。
- [ ] item 13 个 Side field 已统一编码。
- [ ] MISSING 和 OOV 分开。
- [ ] candidate 与 item_feat 的跨源一致性已经审计。
- [ ] candidate `cold_start` 默认没有进入模型特征。

## MM

- [ ] 32D float32 store 可读取。
- [ ] valid mask 存在。
- [ ] 缺失 MM 不会和真实零向量混淆。

## 工程

- [ ] 不按 868 万样本重复物化历史。
- [ ] FeatureDataset 能读取 Train/Val/Test 样本。
- [ ] Stage 4 tests 全部通过。
- [ ] Debug 与 Full manifest 均生成。
- [ ] 大型 feature store 已被 `.gitignore` 排除。

---

# 16. Stage 4 明确“不做”的事情

Codex 不要越界实现以下内容：

```text
❌ ItemCF 训练/评估
❌ Two-Tower 训练
❌ SDM
❌ Action Embedding 网络
❌ Time Encoder 网络
❌ MM Projection 网络
❌ History-strength Gate
❌ Negative Sampling
❌ FAISS
❌ 多路召回融合
❌ DIN / DeepFM / DCN
❌ 排序
```

Stage 4 的职责只有：

```text
正确、统一、无泄漏、高效地把信息交给后续模型
```

---

# 17. Stage 4 与后续模型的接口检查

完成 Stage 4 后，下面这些模型应该都不需要再修改底层数据语义。

## Stage 5：ItemCF / I2I

需要：

```text
train-only sequence/events
item RID
action
timestamp
```

---

## Stage 5：Vanilla Two-Tower

最小输入：

```text
hist_item_token
target_item_token
```

Train-Unseen：

```text
target_item_token = UNK
```

---

## Stage 6：SDM

直接复用完整有序 History：

```text
hist_item_rid
hist_item_token
```

模型层自行决定：

```text
short-term window
long-term encoder
```

Stage 4 不提前切死。

---

## Stage 6：Action-aware

直接增加：

```text
hist_action_token
```

---

## Stage 6：Time-aware

动态派生：

```text
hist_recency
hist_time_gap
```

---

## Stage 6：Side/MM Item Tower

读取：

```text
item_side
item_side_missing
item_side_oov

item_mm
item_mm_valid
```

---

## Stage 6：History-strength-aware

读取：

```text
item_train_count
item_train_count_log1p
strength_group
```

后续模型可研究：

$$
v_i
=
g_i v_i^{\mathrm{ID}}
+
(1-g_i)v_i^{\mathrm{nonID}}
$$

其中：

$$
g_i=f\!\left(\log(1+n_i^{\mathrm{train}})\right)
$$

但这两个公式属于 Stage 6 的模型假设，Stage 4 只负责把 $n_i^{\mathrm{train}}$ 正确提供出来。

---

# 18. 给 Codex 的最终执行要求

1. **先阅读现有仓库代码和上述输入文件，再写代码。**
2. 尽量复用 Stage 1～Stage 3 已有的：
   - path utility；
   - parquet scanner；
   - indexer loader；
   - Stage 3 cutoff；
   - Stage 3 item count semantics；
   - logging 风格。
3. 不复制一套新的、含义不同的 ID mapping。
4. 任何可能引入 future information 的统计量，都必须明确写出 fit scope。
5. 任何 missing value 都不能静默变成普通类别。
6. 任何 train-unseen item 都不能因为全局 indexer 中存在 RID，就自动获得独立 learned ID。
7. `candidate.cold_start` 默认只做 audit，不做模型输入。
8. `retrieval_id` 只做候选池定位，不做 learned ID。
9. 先 Unit Test，再 Audit，再 Debug，再 Full。
10. 代码完成后不要开始 Stage 5；先输出 Stage 4 的：
    - 文件变更列表；
    - 单元测试结果；
    - Debug 结果；
    - Full run 命令；
    - manifest / audit 解释。

---

# 19. Stage 4 最终一句话定义

> **Stage 4 的目标是在 Stage 3 已固定的 next-click 时间协议上，建立训练期可学习 ID、用户序列 Action/Time、用户 Side、广告 Side/MM、缺失/OOV 语义以及 Train-only History Strength 的统一特征层，使 ItemCF、Vanilla Two-Tower、SDM/Action/Time 增强双塔和后续排序模型能够在完全相同、无泄漏的数据语义上公平比较。**
