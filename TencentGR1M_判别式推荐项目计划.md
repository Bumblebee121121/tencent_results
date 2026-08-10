# TencentGR-1M 判别式推荐项目计划书

> 当前状态：TencentGR-1M 必要数据已经下载完成，尚未开始正式建模。  
> 当前范围：先完成判别式推荐；生成式推荐留到第二阶段。  
> 核心原则：先从数据中发现问题，再选择与问题匹配的模型，而不是按模型列表堆模型。

---

## 1. 项目定位

### 1.1 项目名称

**基于行为序列与多模态特征的工业广告推荐系统**

英文名称可暂定为：

```text
Behavior-Aware and Multimodal Sequential Advertising Recommendation on TencentGR-1M
```

### 1.2 项目要证明什么

项目需要向面试官证明三件事：

1. 能独立完成数据读取、样本构造、模型训练、检索和评估；
2. 理解推荐系统中数据、召回、向量检索和效果评估的完整流程；
3. 能通过数据分析发现问题，提出假设，设计迭代，并验证收益来源。

项目的核心论证链应当始终保持为：

$$
\boxed{
\text{数据现象}
\rightarrow
\text{建模问题}
\rightarrow
\text{原因假设}
\rightarrow
\text{模型改进}
\rightarrow
\text{消融与分组验证}
}
$$

### 1.3 当前任务定义

对于用户 $u$，历史行为序列记为：

$$
S_u=
\left[
(i_1,a_1,t_1),
(i_2,a_2,t_2),
\ldots,
(i_T,a_T,t_T)
\right]
$$

其中：

- $i_t$：第 $t$ 次交互的广告；
- $a_t$：行为类型，数据中区分曝光和点击；
- $t_t$：时间戳。

判别式推荐的基本目标是：给定用户历史和候选广告 $i$，计算匹配分数：

$$
s(u,i)=f(S_u,x_u,x_i,m_i)
$$

然后从候选库中选出分数最高的广告：

$$
R_u^K
=
\underset{i\in\mathcal{C}_u}{\operatorname{arg\,topK}}
\ s(u,i)
$$

其中：

- $x_u$：用户匿名特征；
- $x_i$：广告匿名特征；
- $m_i$：广告多模态向量；
- $\mathcal{C}_u$：候选广告集合。

### 1.4 主任务与辅助任务

#### 主任务：下一物品推荐

使用用户历史行为预测下一个交互广告。

#### 辅助任务：下一点击推荐

EDA 证明点击样本数量足够后，可以增加：

> 使用曝光和点击历史，预测用户下一次点击的广告。

两个任务必须分开报告：

- 下一物品任务用于复现和比较序列推荐模型；
- 下一点击任务用于研究强反馈和弱反馈差异。

---

## 2. 对数据集的已知认识

根据 TencentGR-1M 官方数据卡：

- 用户数约为 100 万；
- 每个用户最多保留 100 个历史行为；
- 每个行为包含 `item_id`、`action_type` 和 `timestamp`；
- `action_type=0` 表示曝光，`action_type=1` 表示点击；
- `item_feat` 包含约 478 万个历史广告的匿名特征；
- `candidate` 包含约 66 万个候选广告；
- 官方 `candidate` 不提供 Ground Truth，不能直接作为离线验证集；
- 部分候选广告没有出现在历史序列中；
- `mm_emb_81_32` 提供 32 维多模态广告表示；
- 多模态表使用 OID，训练数据使用 RID，二者需要通过 `indexer.pkl` 对齐。

官方 Baseline：

- 基于 SASRec/Transformer 建模用户行为序列；
- 使用正样本和负样本的向量内积进行判别训练；
- 推理阶段输出用户向量与广告向量；
- 通过向量检索完成候选召回。

因此，虽然该数据集原本服务于生成式推荐比赛，但完全可以建立严谨的判别式序列推荐项目。

---

## 3. 项目总路线

```text
原始 Parquet 数据
        ↓
数据完整性校验与 ID 对齐
        ↓
EDA 与数据问题发现
        ↓
统一任务定义、时间切分和泄漏测试
        ↓
Popularity / ItemCF 传统基线
        ↓
Two-Tower 向量召回
        ↓
SASRec-ID
        ↓
官方多特征 SASRec 复现
        ↓
行为类型与时间信息增强
        ↓
多模态与长尾/冷启动增强
        ↓
FAISS 检索、效率与扩展性实验
        ↓
消融、分组评估和误差分析
        ↓
README、实验报告和简历表述
```

当前不做：

- RQ-VAE；
- Semantic ID；
- 自回归生成；
- 生成推荐列表；
- 将大量 FunRec 模型机械地全部加入；
- 将自行构造的离线样本宣称为真实线上 CTR 日志。

---

# 4. 第一阶段：数据完整性与项目骨架

## 4.1 下载完成后的第一件事

不是立即训练模型，而是：

> 确认数据文件完整、字段正确、ID 能够对齐，并建立可重复的数据读取入口。

## 4.2 建议项目目录

```text
tencent_rec/
├── README.md
├── configs/
│   ├── data.yaml
│   ├── popularity.yaml
│   ├── itemcf.yaml
│   ├── two_tower.yaml
│   └── sasrec.yaml
├── data/
│   └── TencentGR-1M/
├── src/
│   ├── data/
│   │   ├── schema.py
│   │   ├── reader.py
│   │   ├── id_mapping.py
│   │   ├── build_splits.py
│   │   └── negative_sampling.py
│   ├── eda/
│   │   ├── profile_dataset.py
│   │   └── plot_distributions.py
│   ├── models/
│   │   ├── popularity.py
│   │   ├── itemcf.py
│   │   ├── two_tower.py
│   │   ├── sasrec.py
│   │   └── feature_sasrec.py
│   ├── retrieval/
│   │   ├── exact_search.py
│   │   └── faiss_search.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── grouped_metrics.py
│   │   └── evaluator.py
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── paths.py
├── scripts/
│   ├── verify_data.py
│   ├── run_eda.py
│   ├── build_splits.py
│   ├── train.py
│   ├── retrieve.py
│   └── evaluate.py
├── tests/
│   ├── test_schema.py
│   ├── test_id_mapping.py
│   ├── test_temporal_leakage.py
│   ├── test_negative_sampling.py
│   └── test_metrics.py
├── reports/
├── artifacts/
│   ├── data_profile/
│   ├── splits/
│   ├── retrieval/
│   └── metrics/
├── checkpoints/
├── logs/
└── third_party/
    └── baseline_2025/
```

原则：

- 原始数据只读，不覆盖；
- 官方代码放入 `third_party/`，不要直接无记录修改；
- 自己实现的代码放入 `src/`；
- 每次实验固定配置、随机种子和 `run_id`；
- 原始数据、权重、缓存和日志不提交到 Git。

## 4.3 数据校验任务

创建 `scripts/verify_data.py`，至少检查：

- 必要目录和文件是否存在；
- Parquet 文件数量和 Schema；
- 序列时间戳是否基本单调；
- `action_type` 是否只出现合法值；
- 用户和物品 ID 是否存在非法值；
- `indexer.pkl` 能否加载；
- OID 与 RID 是否能够正确映射；
- 多模态向量是否为 32 维；
- 用户表、物品表和序列表主键是否重复；
- 序列广告在 `item_feat` 中的覆盖率；
- 多模态向量与广告的匹配覆盖率。

输出：

```text
artifacts/data_profile/data_integrity_report.json
artifacts/data_profile/data_integrity_report.md
```

### 第一阶段验收标准

- 数据读取不需要一次性加载全部表到内存；
- 数据校验脚本可以重复执行；
- ID 对齐规则有单元测试；
- 能稳定读取固定的 1000 个用户；
- 此阶段不修改模型结构。

---

# 5. 第二阶段：EDA 与问题发现

EDA 的目标不是画很多图，而是决定后续做什么模型。

## 5.1 基础统计

统计：

- 用户数、历史广告数和总行为数；
- 序列长度分布；
- 每个用户的曝光数和点击数；
- 广告交互频次分布；
- 时间戳范围；
- 用户特征、广告特征与多模态覆盖率；
- `candidate` 中历史广告与新广告的数量。

## 5.2 行为类型分析

定义用户在该数据序列中的点击行为比例：

$$
\operatorname{ClickRatio}_u
=
\frac{N_{u,\mathrm{click}}}
{N_{u,\mathrm{exposure}}+N_{u,\mathrm{click}}}
$$

注意：它只是数据序列中的行为比例，不直接等同于腾讯线上真实 CTR。

需要分析：

- 曝光和点击的总体比例；
- 用户点击行为比例的分布；
- 点击是否比曝光更能预测下一次交互；
- 仅曝光、低点击和高点击用户的占比；
- 相同广告的曝光和点击是否对应不同后续行为。

这一步决定是否值得加入：

- 行为类型 Embedding；
- 点击加权；
- 行为条件注意力；
- 下一点击辅助任务。

## 5.3 时间分析

相邻行为时间间隔：

$$
\Delta t_k=t_k-t_{k-1}
$$

分析：

- 时间戳是否单调；
- 时间间隔分布；
- 是否存在大量同时刻行为；
- 用户会话间隔；
- 最近行为对目标广告的预测能力；
- 不同时间间隔下广告重复交互概率。

这一步决定是否值得加入：

- 时间桶 Embedding；
- 相对时间偏置；
- 短期/长期兴趣融合；
- Session 划分。

## 5.4 长尾分析

训练期广告热度：

$$
\operatorname{pop}(i)
=
\sum_{u,t}
\mathbb{I}(i_{u,t}=i)
$$

分析：

- 广告热度分布；
- Top 1%、5%、10% 广告贡献的交互比例；
- 验证目标属于热门或长尾广告的比例；
- 用户历史中热门和长尾广告的比例。

建议按照训练集分位数划分：

```text
Head
Mid
Tail
Unseen-in-train
```

## 5.5 特征缺失与多模态覆盖

字段覆盖率：

$$
\operatorname{Coverage}(f)
=
\frac{
\#\{x:f(x)\neq \mathrm{None}\}
}{
\#\{x\}
}
$$

分析：

- 每个用户和广告匿名字段的覆盖率；
- 多模态向量覆盖率；
- 特征缺失是否与广告热度相关；
- 特征缺失广告是否更多出现在长尾组；
- 低覆盖字段是否值得特殊处理。

## 5.6 EDA 输出

```text
reports/eda_report.md
artifacts/data_profile/dataset_summary.json
artifacts/data_profile/feature_coverage.csv
artifacts/data_profile/user_groups.parquet
artifacts/data_profile/item_groups.parquet
```

EDA 报告最后必须回答：

1. 数据中最明显的三个问题是什么？
2. 哪些问题可以由现有字段验证？
3. 哪些问题缺少标签，不能做？
4. 第一条模型迭代应解决什么问题？
5. 预期改善哪个用户或广告子集？

至少形成两个待验证假设，例如：

$$
H_1:
\text{区分曝光和点击能够改善高曝光低点击用户的推荐}
$$

$$
H_2:
\text{侧信息和多模态信息主要改善长尾或训练期未见广告}
$$

---

# 6. 第三阶段：统一切分、样本与评估

## 6.1 不直接用官方 candidate 做验证

官方 `candidate` 没有 Ground Truth，因此：

- 可以用于候选格式、冷启动字段分析和检索实验；
- 不能直接用于计算离线 Recall/NDCG；
- 必须从 `seq` 中构建训练、验证和测试目标。

## 6.2 主切分：用户内时间留出

对于长度至少为 3 的用户序列：

$$
S_u=(i_1,i_2,\ldots,i_T)
$$

构造：

$$
S_u^{\mathrm{train}}
=(i_1,\ldots,i_{T-2})
$$

$$
y_u^{\mathrm{val}}=i_{T-1}
$$

$$
y_u^{\mathrm{test}}=i_T
$$

测试时历史为：

$$
S_u^{\mathrm{test\ history}}
=(i_1,\ldots,i_{T-1})
$$

优点：适合下一物品推荐，每个用户都有历史。

限制：不能严格模拟全局时间上线，也不能天然定义严格的全局新品冷启动。

## 6.3 辅助切分：全局时间切分

EDA 确认时间戳可比较后，再考虑：

$$
\mathcal{D}_{\mathrm{train}}
=
\{e:t(e)<\tau_1\}
$$

$$
\mathcal{D}_{\mathrm{val}}
=
\{e:\tau_1\leq t(e)<\tau_2\}
$$

$$
\mathcal{D}_{\mathrm{test}}
=
\{e:t(e)\geq\tau_2\}
$$

用途：时间外推和 temporal cold-start。该方案不是第一版必须项。

## 6.4 负采样

至少比较：

1. 随机负样本；
2. 热度采样；
3. Batch 内负样本；
4. 检索得到的困难负样本。

负样本必须满足：

$$
i^{-}\notin S_u^{+}
$$

后续可使用混合负样本：

$$
\mathcal{N}_u
=
\mathcal{N}_u^{\mathrm{random}}
\cup
\mathcal{N}_u^{\mathrm{popular}}
\cup
\mathcal{N}_u^{\mathrm{hard}}
$$

## 6.5 两套评估

### 小候选集评估

例如：

```text
1 个正样本 + 99 个负样本
```

用于快速开发，但不能作为唯一最终结果。

### 全库或大候选库评估

使用训练物品库或可映射的大候选广告库进行检索。这是最终简历结果应优先使用的评估。

## 6.6 核心指标

当每个用户只有一个目标广告时：

### Recall@K

$$
\operatorname{Recall@K}
=
\frac{1}{|\mathcal{U}|}
\sum_{u\in\mathcal{U}}
\mathbb{I}(y_u\in R_u^K)
$$

### NDCG@K

$$
\operatorname{NDCG@K}_u
=
\begin{cases}
\dfrac{1}{\log_2(r_u+1)}, & r_u\leq K\\
0, & r_u>K
\end{cases}
$$

### MRR@K

$$
\operatorname{MRR@K}
=
\frac{1}{|\mathcal{U}|}
\sum_u
\frac{\mathbb{I}(r_u\leq K)}{r_u}
$$

至少报告：

```text
Recall@10
Recall@50
Recall@100
NDCG@10
NDCG@50
MRR@10
```

## 6.7 分组指标

主要模型都应报告：

- 短/中/长序列用户；
- 低/中/高点击用户；
- Head/Mid/Tail 广告；
- 训练期已见/未见广告；
- 有/无多模态广告；
- 特征完整/缺失广告。

## 6.8 泄漏测试

自动化检查：

- 验证和测试目标没有进入对应历史；
- 广告热度和统计特征只从训练交互计算；
- 负样本不包含正样本和用户已交互广告；
- 测试目标不参与训练负采样；
- 模型选择只看验证集；
- 静态侧信息可以使用，但未来交互统计不能使用；
- 全局时间切分时，训练特征不能读取未来窗口的交互统计。

输出：

```text
artifacts/splits/split_manifest.json
tests/test_temporal_leakage.py
```

---

# 7. 第四阶段：逐级判别式基线

## 7.1 P0：Popularity

$$
s_{\mathrm{pop}}(u,i)=\operatorname{pop}(i)
$$

作用：

- 建立非个性化下界；
- 验证指标和切分；
- 判断数据是否被热门广告支配。

## 7.2 P1：行为加权 Popularity

$$
\operatorname{pop}(i)
=
N_{\mathrm{exposure}}(i)
+
\lambda N_{\mathrm{click}}(i)
$$

$\lambda$ 必须由验证集选择。

目的：快速验证点击是否比曝光具有更强预测价值。

## 7.3 P2：ItemCF

$$
\operatorname{sim}(i,j)
=
\frac{C_{ij}}{\sqrt{C_iC_j}}
$$

$$
s(u,j)
=
\sum_{i\in S_u}
w(i,u)\operatorname{sim}(i,j)
$$

逐步加入：

- 行为权重；
- 时间衰减；
- 每个广告只保留 Top-N 邻居；
- 降低超热门广告导致的虚假共现。

时间衰减示例：

$$
w(i_t,u)
=
w_{a_t}\exp(-\gamma\Delta t_t)
$$

## 7.4 P3：Two-Tower

$$
e_u=f_{\mathrm{user}}(S_u,x_u)
$$

$$
e_i=f_{\mathrm{item}}(i,x_i,m_i)
$$

$$
s(u,i)=e_u^\top e_i
$$

逐级实验：

```text
Two-Tower-ID
Two-Tower-ID + user/item feature
Two-Tower-ID + 32D multimodal
Two-Tower + in-batch negatives
Two-Tower + hard negatives
```

## 7.5 P4：SASRec-ID

只使用广告 ID、位置编码和因果自注意力：

$$
x_t=e_{i_t}+p_t
$$

$$
s(u,i)=h_T^\top e_i
$$

目的：单独验证序列顺序与自注意力的收益。

## 7.6 P5：官方多特征 SASRec

先复现官方 Baseline，不立即修改。

固定并记录：

- 官方 commit；
- 配置；
- 随机种子；
- 数据子集；
- 日志与 checkpoint；
- 验证指标；
- 显存和训练时间。

必须进行代码审计：

- 实际使用了哪些用户和广告特征；
- 负采样方式；
- `action_type` 是否真正进入用户表示或损失；
- 时间戳是否被模型使用；
- 正负样本损失如何计算；
- 推理如何导出用户和广告向量。

### 基线阶段结果表

| Model | Recall@10 | Recall@50 | NDCG@10 | NDCG@50 | Train Time | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| Popularity |  |  |  |  |  |  |
| Weighted Popularity |  |  |  |  |  |  |
| ItemCF |  |  |  |  |  |  |
| Two-Tower-ID |  |  |  |  |  |  |
| SASRec-ID |  |  |  |  |  |  |
| Official Feature-SASRec |  |  |  |  |  |  |

这些基线稳定前，不进入复杂改进阶段。

---

# 8. 第五阶段：问题驱动的模型迭代

最终只选择一到两个主问题，不建议同时铺开很多方向。

## 8.1 主方向 A：行为感知与时间感知

### 假设

$$
H_A:
\text{曝光和点击表达的兴趣强度不同}
$$

$$
H_B:
\text{相同位置间隔不代表相同真实时间间隔}
$$

### A1：行为类型 Embedding

$$
x_t=e_{i_t}+e_{a_t}+p_t
$$

### A2：时间间隔 Embedding

$$
\Delta t_t=t_t-t_{t-1}
$$

先对时间间隔进行对数分桶：

$$
b_t=\operatorname{bucket}\left(\log(1+\Delta t_t)\right)
$$

然后：

$$
x_t=e_{i_t}+e_{a_t}+e_{\Delta t_t}+p_t
$$

### A3：行为加权损失

$$
\mathcal{L}
=
\sum_t w(a_{t+1})\mathcal{L}_t
$$

其中：

$$
w(\mathrm{click})>w(\mathrm{exposure})
$$

权重由验证集确定。

### A4：相对时间注意力

$$
A_{pq}
=
\frac{Q_pK_q^\top}{\sqrt d}
+
b_{\operatorname{bucket}(|t_p-t_q|)}
$$

### 必须验证的收益来源

- 点击活跃用户是否收益更明显；
- 长时间间隔序列是否收益更明显；
- 近期兴趣主导用户是否收益更明显；
- 行为和时间模块是否各自有效；
- 收益是否只是热门广告偏置。

### 消融

```text
SASRec-ID
+ action embedding
+ time-gap embedding
+ action-weighted loss
+ relative-time bias
Full behavior-time model
```

## 8.2 主方向 B：多模态与长尾/冷启动增强

### 假设

$$
H_C:
\text{纯 ID 模型难以学习低频或训练期未见广告}
$$

$$
H_D:
\text{多模态收益主要出现在长尾和冷启动子集}
$$

### B1：直接拼接

$$
e_i
=
\operatorname{MLP}
\left(
[e_i^{\mathrm{ID}};e_i^{\mathrm{feat}};m_i]
\right)
$$

### B2：门控融合

$$
g_i
=
\sigma
\left(
W_g[e_i^{\mathrm{ID}};e_i^{\mathrm{feat}};m_i]
\right)
$$

$$
e_i
=
e_i^{\mathrm{ID}}
+
g_i\odot
\operatorname{MLP}([e_i^{\mathrm{feat}};m_i])
$$

### B3：缺失感知

$$
r_i^{(m)}
=
\mathbb{I}(m_i\ \text{available})
$$

模型显式输入：

$$
[m_i;r_i^{(m)}]
$$

避免把缺失补零与真实零向量混淆。

### B4：模态 Dropout

训练时随机丢弃部分侧信息，降低模型对单一模态的过度依赖。

### 必须验证的收益来源

比较：

$$
\Delta\operatorname{Recall@K}_{\mathrm{head}},
\quad
\Delta\operatorname{Recall@K}_{\mathrm{tail}}
$$

$$
\Delta\operatorname{Recall@K}_{\mathrm{seen}},
\quad
\Delta\operatorname{Recall@K}_{\mathrm{unseen}}
$$

只有长尾或未见广告获得更明显改善时，才能说模型缓解了对应问题。

---

# 9. 第六阶段：向量检索与扩展性

## 9.1 精确检索与 FAISS

先在小候选库进行精确内积检索，再接入 FAISS。

比较：

```text
Exact Inner Product
FAISS FlatIP
FAISS IVF
可选 HNSW
```

## 9.2 ANN 检索质量

$$
\operatorname{ANNRecall@K}
=
\frac{|R_{\mathrm{ANN}}^K\cap R_{\mathrm{Exact}}^K|}{K}
$$

注意：该指标衡量近似检索对精确 Top-K 的逼近，不等同于推荐任务的 `Recall@K`。

## 9.3 扩展性实验

候选广告规模：

$$
N\in\{10^4,10^5,6.6\times10^5\}
$$

候选数量：

$$
K\in\{10,50,100,500\}
$$

记录：

- 平均检索延迟；
- P95 延迟；
- QPS；
- 索引构建时间；
- 索引大小；
- ANNRecall@K；
- 推荐 Recall@K。

可以表述为：

> 完成离线广告向量生成、FAISS 索引构建和 Top-K 候选召回。

不能表述为：

> 部署了腾讯线上推荐系统。

---

# 10. 排序模块是否要做

## 10.1 当前不作为核心阶段

TencentGR-1M 没有提供完整的线上请求候选集、展示位置、召回来源和真实 CTR 训练表。

因此暂时不强行加入：

```text
DeepFM
DIN
DIEN
MMoE
PLE
```

这些模型需要与任务标签匹配，不能为了模型数量而使用。

## 10.2 可选离线重排

召回阶段完成后，可以构造：

- 正样本：用户下一真实交互广告；
- 负样本：召回 Top-K 中未成为目标的广告；
- 标签：下一物品是否为该广告。

必须命名为：

> 基于下一物品任务构造的离线候选重排。

不能宣称为真实线上 CTR 精排。

由于已有新闻推荐召回—精排项目，本项目更应该突出：

- 行为异质性；
- 时间信息；
- 序列建模；
- 长尾与冷启动；
- 多模态融合；
- 大规模向量检索。

---

# 11. 实验管理与复现

## 11.1 三档数据规模

### Smoke

```text
1,000～10,000 用户
```

只用于代码正确性检查，不用于最终结论。

### Fixed Dev

```text
50,000～100,000 用户
```

使用固定哈希或固定随机种子抽样，用于模型迭代和消融。

### Full

```text
全部约 100 万用户
```

用于最终结果和效率分析。

## 11.2 每个实验记录

```json
{
  "run_id": "",
  "model": "",
  "dataset_version": "",
  "split_version": "",
  "seed": 2026,
  "config": {},
  "git_commit": "",
  "train_time_seconds": 0,
  "peak_vram_mb": 0,
  "metrics": {}
}
```

## 11.3 实验命名

```text
P0_popularity
P1_weighted_popularity
P2_itemcf
P3_two_tower_id
P4_sasrec_id
P5_official_feature_sasrec
A1_sasrec_action
A2_sasrec_action_time
B1_sasrec_mm_concat
B2_sasrec_mm_gate
```

## 11.4 多随机种子

正式神经模型至少运行 3 个随机种子：

$$
\mu=\frac{1}{S}\sum_{s=1}^{S}m_s
$$

$$
\sigma
=
\sqrt{
\frac{1}{S-1}
\sum_{s=1}^{S}(m_s-\mu)^2
}
$$

报告：

$$
\mu\pm\sigma
$$

---

# 12. 开发里程碑

## Milestone 0：数据可用

- [ ] 数据完整性校验通过
- [ ] ID 映射通过
- [ ] 小批量读取通过
- [ ] 原始数据只读
- [ ] 项目目录建立

## Milestone 1：可信数据协议

- [ ] EDA 完成
- [ ] 任务定义固定
- [ ] 用户内时间切分完成
- [ ] 泄漏测试通过
- [ ] 指标单元测试通过
- [ ] Fixed Dev 固定

## Milestone 2：传统基线

- [ ] Popularity
- [ ] Weighted Popularity
- [ ] ItemCF
- [ ] 统一评估

## Milestone 3：神经召回

- [ ] Two-Tower-ID
- [ ] SASRec-ID
- [ ] 官方 Feature-SASRec
- [ ] FAISS 检索

## Milestone 4：问题驱动迭代

优先完成：

- [ ] 行为与时间增强

随后根据结果决定：

- [ ] 多模态与长尾/冷启动增强

不要同时开工两条复杂主线。

## Milestone 5：完整证据链

- [ ] 消融实验
- [ ] 分组指标
- [ ] 多随机种子
- [ ] 效率与扩展性
- [ ] 失败实验记录
- [ ] README 和实验报告
- [ ] 简历表述与面试问答

---

# 13. 下载完数据后，现在立即做什么

按以下顺序执行，不要跳步。

## 第 1 步：建立项目仓库和目录

原始数据放在：

```text
data/TencentGR-1M/
```

`.gitignore` 至少加入：

```gitignore
data/
checkpoints/
logs/
artifacts/
*.fbin
*.u64bin
```

## 第 2 步：引入官方 Baseline

将官方仓库放入：

```text
third_party/baseline_2025/
```

记录 commit，不直接大改。

## 第 3 步：编写数据校验脚本

先检查目录、Schema、样例、ID 映射、向量维度和覆盖率，不进行全量训练。

## 第 4 步：生成 1000 用户 Smoke 数据

使用固定种子或用户 ID 哈希生成：

```text
artifacts/splits/smoke_users.json
```

以后所有程序先在 Smoke 数据上跑通。

## 第 5 步：完成第一版 EDA

优先输出五项：

1. 序列长度分布；
2. 曝光/点击比例；
3. 时间间隔分布；
4. 广告热度长尾分布；
5. 用户、广告和多模态特征覆盖率。

## 第 6 步：固定评估协议

实现：

- 用户内时间留出；
- 负样本过滤；
- Recall/NDCG/MRR；
- 泄漏测试。

## 第 7 步：先跑 Popularity

用它验证：

- 切分是否正确；
- 指标是否正确；
- 推荐列表是否排除历史物品；
- 数据是否被极少数热门广告支配。

## 第 8 步：再跑 ItemCF

检验共现关系、行为权重、时间衰减和长尾表现。

## 第 9 步：跑通官方 SASRec Smoke

只使用 Smoke 数据，确认数据路径、batch、loss、checkpoint、用户向量、广告向量和检索输出。

## 第 10 步：建立 Fixed Dev

Smoke 全部通过后，再固定 50,000～100,000 用户进行正式迭代。

---

# 14. 第一轮 Codex 任务边界

第一轮只允许 Codex 完成：

```text
项目目录
+ 数据校验
+ 小样本读取
+ EDA
+ 时间切分
+ 泄漏测试
+ Popularity
```

第一轮不要让 Codex：

- 直接跑全量 SASRec；
- 随意修改官方 Baseline；
- 加入 RQ-VAE；
- 加入 DeepFM/DIN/DIEN；
- 一次设计十几个模型；
- 根据预期伪造实验结果；
- 使用测试集调参。

第一轮应生成：

```text
reports/eda_report.md
artifacts/data_profile/data_integrity_report.json
artifacts/splits/split_manifest.json
artifacts/metrics/popularity.json
tests/test_temporal_leakage.py
tests/test_metrics.py
```

---

# 15. 预期的最终项目故事

在完成实验前不能提前宣称结果，但可以暂定以下框架：

> 首先对 TencentGR-1M 真实广告行为序列进行数据审计，分析曝光与点击反馈差异、行为时间间隔和广告长尾结构。随后建立 Popularity、ItemCF、Two-Tower 和 SASRec 等逐级判别式基线，在统一时间切分与大候选检索协议下比较协同过滤、向量召回和序列推荐。基于 EDA 结果，在 SASRec 中加入行为类型与时间信息，并结合广告匿名侧信息及轻量多模态向量改善长尾或训练期未见广告的表示。最后通过消融、用户/广告分组实验和 FAISS 扩展性测试验证收益来源与工程可用性。

最终叙事必须根据真实实验结果调整。

---

# 16. 判别式阶段停止条件

满足以下条件后即可暂停，不必无限增加实验：

- 数据完整性和泄漏测试完善；
- 至少 3 个逐级基线；
- 官方 SASRec 已复现；
- 至少 1 个问题驱动改进；
- 有完整消融；
- 有分组收益归因；
- 有全库或大候选检索；
- 有效率和资源指标；
- 所有结果可从配置复现；
- README 能解释“为什么做”，而不只是“做了什么”。

完成后，再决定是否进入生成式推荐阶段。

---

# 17. 外部依据

## 腾讯官方数据与代码

- TencentGR-1M 数据卡：  
  `https://huggingface.co/datasets/TAAC2025/TencentGR-1M`

- 2025 腾讯广告算法大赛官方 Baseline：  
  `https://github.com/TencentAdvertisingAlgorithmCompetition/baseline_2025`

- 数据集与比赛论文：  
  `https://arxiv.org/abs/2604.04976`

## 项目方法论依据

用户上传的“小红书博主项目建议”强调：

- 项目需要证明代码能力、全链路理解和迭代能力；
- 创新应来自场景和数据中的实际问题；
- 需要解释收益来源、替代方案和验证方法；
- 完整的思维故事比简单罗列模型更重要。

本计划因此把 EDA、问题假设、对照实验、消融和分组归因放在项目核心位置。
