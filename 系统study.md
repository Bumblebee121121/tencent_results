# 任务：根据用户在目标广告曝光之前的曝光、点击历史，以及用户、广告和多模态侧信息，从候选广告集合中召回并排序用户下一次最可能点击的广告。
# tencent数据集
本地路径	                    是什么	                                        推荐系统里的作用
seq/	                    用户行为序列	                                    最核心的训练数据
user_feat/	                用户匿名特征（属性）	                            描述“这个用户是什么样的人”
item_feat/	                历史广告匿名特征（属性）	                        描述“历史广告是什么样的广告”
candidate/	                候选广告集合（历史出现过的广告和历史没出现过的广告）    模型最后要从这里选广告
mm_emb/emb_81_32_parquet/	广告 32 维多模态向量	                            描述广告内容语义
indexer.pkl	ID/特征映射表	把原始 ID 转成训练使用的内部                            ID

1. seq用户行为序列：包含用户Rid，用户交互过的广告 RID，交互行为（曝光和点击）记录，交互行为发生的时间。长度n代表该用户有n个交互过的广告。
    item_id	用户交互过的广告 ID
    action_type=0	曝光，但没有点击
    action_type=1	点击
    timestamp	该行为发生的时间
如：
user_id = 953188

seq = [
    {
        item_id: 2905777,
        action_type: 0,
        timestamp: 1745919240
    },
    {
        item_id: 728757,
        action_type: 0,
        timestamp: 1745919475
    },
    ...
]
1. user_feat用户匿名特征：包含用户Rid，匿名特征（不知道某个特征具体对应什么信息（如：性别，年龄））。
   模型以后可以利用这些特征学习：某类用户可能更喜欢某类广告。
user_id = 19
103 = 36
104 = 2
105 = 7
106 = [6]
107 = [15]
108 = null
109 = 2
110 = null
1. item_feat历史广告匿名特征:包含历史广告Rid, 匿名特征（不知道某个特征具体对应什么信息（如：广告类别，广告商））。
item_id = 26
100 = 4
101 = 5
102 = 81603
112 = 14
114 = 3
115 = null
116 = 8
117 = 382
118 = 692
119 = 3077
120 = 640
121 = 2011258
122 = 67985
1. candidate候选广告集合：包含广告原始id（OID）和为了检索方便创建的连续编号,匿名特征以及该特征对应的冷启动信息。 
   模型最终要从这些广告中选出 Top-K。candidate 大约有 66 万个广告

2. mm_emb广告 32 维多模态向量:包含广告原始id（OID）和这个广告对应的 embedding （语义）向量。
   完整官方数据实际上提供 6 套广告多模态 embedding,这些 embedding 来自广告的文本和视觉等多模态内容表示。
一个广告：广告图片 + 广告文字 + 内容信息
                 ↓
          多模态模型
                 ↓
    [0.154, -0.211, ..., 0.083] 32维语义向量
                 ↑
              32维
    
1. indexer.pkl：包含用户 ID、广告 ID、匿名特征映射。键是OID,值是RID。
   {
    "u": {
        OID: RID,
        ...
    },

    "i": {
        OID: RID,
        ...
    },

    "f": {
        101: {
            原始特征值: 映射后的特征值,
            ...
        },

        102: {
            原始特征值: 映射后的特征值,
            ...
        },

        ...
    }
}
生成式推荐：EDA 表明，当前任务存在两类核心困难：用户侧，曝光与点击行为语义不同，同时交互行为间存在时间差异，如果模型只依赖 item 顺序和位置编码，可能无法准确刻画用户当前点击兴趣；物品侧，大量 Tail/Unseen（历史只出现 1～2 次以及历史出现0 次） 广告缺乏可靠的 ID 学习信号（假设传统模型给每个广告一个 ID Embedding）。官方 Baseline 已利用 Transformer 建模基本序列依赖，并融合结构化和多模态特征缓解弱 ID 问题，但对用户侧没有显式利用历史行为（action）和真实时间，对物品侧也没有根据不同历史信号强度显式区分 ID 与非 ID 信息的可靠程度。因此，我在用户侧准备验证 Action-aware、Time-aware 以及更贴合 next-click 的训练方式；在物品侧则验证 History-strength-aware 的 ID/Side/MM 自适应融合，并通过 Head/Mid/Tail/Unseen 分组实验分析收益来源。

Action-aware 可以翻译成-行为感知的。aware表示：模型显式知道并利用某类信息。Action-aware = 模型显式利用 action type。这里的 action 就是：exposure、click、unknown
Time-aware 可以翻译成-时间感知的。Time-aware = 模型不仅知道行为先后顺序，还显式利用交互行为间的时间信息。
time信息：绝对时间Absolute time，相对时间（时间差）time-gap，recency（预测用户下一次点击的时间与历史某个交互行为的时间的差值，Session（时间上连续、可能具有相似短期意图的一组行为，如：10:01 点击手机 10:03 点击手机壳 10:06 点击充电器 10:09 曝光耳机）
 
# EDA（探索性数据分析） 与问题发现
## EDA 表明用户侧存在长序列和明显的时间异质性，而 item 侧存在严重长尾和大量 history-unseen candidate；更重要的是这些弱历史 item 仍然拥有丰富的结构化与多模态信息，因此项目后续最值得验证的是如何利用非 ID 信息增强弱历史 item 的泛化表示，同时通过 Head/Mid/Tail/Unseen 分组评估确认收益究竟来自哪里。
长序列、时间异质性、长尾、unseen 本身不是模型问题，它们是数据事实；真正的问题是这些事实会导致模型无法准确形成用户表示或候选广告表示，从而把真实的下一点击广告排不到 Top-K。
## 生成式推荐：
EDA 表明，当前任务存在两类核心困难：用户侧，曝光与点击行为语义不同，同时交互行为间存在时间差异，如果模型只依赖 item 顺序和位置编码，可能无法准确刻画用户当前点击兴趣；物品侧，大量 Tail/Unseen（历史只出现 1～2 次以及历史出现0 次） 广告缺乏可靠的 ID 学习信号（假设传统模型给每个广告一个 ID Embedding）。官方 Baseline 已利用 Transformer 建模基本序列依赖，并融合结构化和多模态特征缓解弱 ID 问题，但对用户侧没有显式利用历史行为（action）和真实时间，对物品侧也没有根据不同历史信号强度显式区分 ID 与非 ID 信息的可靠程度。因此，我在用户侧准备验证 Action-aware、Time-aware 以及更贴合 next-click 的训练方式；在物品侧则验证 History-strength-aware 的 ID/Side/MM 自适应融合，并通过 Head/Mid/Tail/Unseen 分组实验分析收益来源。

Action-aware 可以翻译成-行为感知的。aware表示：模型显式知道并利用某类信息。Action-aware = 模型显式利用 action type。这里的 action 就是：exposure、click、unknown
Time-aware 可以翻译成-时间感知的。Time-aware = 模型不仅知道行为先后顺序，还显式利用交互行为间的时间信息。
time信息：绝对时间Absolute time，相对时间（时间差）time-gap，recency（预测用户下一次点击的时间与历史某个交互行为的时间的差值，Session（时间上连续、可能具有相似短期意图的一组行为，如：10:01 点击手机 10:03 点击手机壳 10:06 点击充电器 10:09 曝光耳机）
## 判别式推荐
EDA 表明，用户侧，曝光与点击行为语义不同，同时交互行为间存在时间差异。物品侧，大量 Tail/Unseen（历史只出现 1～2 次以及历史出现0 次） 广告缺乏可靠的 ID 学习信号（假设传统模型给每个广告一个 ID Embedding），但这些弱历史广告仍具有较完整的结构化和多模态信息。因此后续判别式建模重点验证行为/时间感知的用户表示，以及利用非 ID 信息增强弱历史 Item 表示，并在严格时间切分后通过 Head/Mid/Tail/Unseen 的 target 分组评估确认实际收益来源。

我们在 EDA 中暂时根据历史交互次数划分：
Tail:ni≤2
Mid:2< ni​ ≤23
Head:ni >23

EDA 发现：
item 历史交互次数中位数只有 2；
约 73% 的 item 历史交互不超过 5 次；
但 Top 1% item 却贡献了接近一半的历史行为。

1%广告被曝光/点击的次数大约占总交互历史次数的一半。约 73% 的 item 历史交互不超过 5 次。

横轴：广告id按照交互次数从高到低排列；
纵轴：广告交互次数；

画出来，会表现为：

少数 Head
频率极高
│\
│ \
│  \
│   \____________________________
│                 大量 Tail
└───────────────────────────────

这就是“长尾”。

EDA：

大量 Tail / Unseen
       ↓
问题：
ID 表示学不好
       ↓
继续 EDA：
它们有没有别的信息？
       ↓
发现：
Side/MM 仍然大量存在
       ↓
假设：
能否利用非 ID 信息弥补 ID 表示不足？

2.1：序列有多长、点击是否稀疏
2.2：序列跨多久、相邻行为隔多久
2.3：广告有多热门、交互是否集中在少量头部广告
2.4：候选广告有多少历史未见，以及未见广告还有没有侧信息/多模态信息可利用
2.5：多模态到底覆盖了多少广告，以及缺失是否集中在特定广告群体
2.6：用户/广告侧特征是什么结构、缺多少、取值空间有多大

## 2.1 stage2_1_sequence_behavior.py 的唯一核心输入就是 seq 用户行为序列，它主要统计两件事：
① 每个用户的行为序列有多长->判断序列建模有没有意义；② 每个用户的序列里曝光、点击、未知行为是怎么构成的->判断点击是否稀疏、正负反馈是否失衡。

统计对象	统计内容	文字公式
数据规模	用户总数	seq 中用户行的总数量
数据规模	行为总数	所有用户序列长度相加
序列长度	每用户序列长度	一个用户 seq 中事件的数量
序列长度	长度分位数	对所有用户序列长度排序后统计不同分位位置
序列长度	长度区间	统计处于 1–10、11–20、21–50 等区间的用户数及占比
长度上限	长度恰好 100	长度等于 100 的用户数及其占总用户的比例
行为构成	曝光数	action_type=0 的事件数
行为构成	点击数	action_type=1 的事件数
行为构成	unknown 数	action_type 为空的事件数
行为构成	各行为占比	某类行为数量除以全部行为数量
用户点击	每用户点击数	某用户序列中 action_type=1 的数量
用户点击	0/1/≥2 点击用户占比	对用户按照点击次数分组，再除以总用户数
点击稀疏性	每用户点击行为比例	用户点击数除以该用户曝光数与点击数之和
缺失情况	每用户 unknown 比例	用户 unknown 数除以该用户序列总长度
## 2.2 stage2_2_temporal_patterns.py 主要统计两件事：
① 每个用户的整段历史行为跨了多长时间（history span）
→ 判断用户序列虽然长度相近，但真实时间覆盖范围是否差异很大，从而判断仅用序列位置建模是否足够。
② 同一用户相邻两次行为之间隔了多久（delta_t）
→ 判断用户行为主要发生在秒级、分钟级、小时级还是天级，以及是否存在大量长时间间隔，从而判断后续是否值得加入 time-gap、recency 等时间特征或时间编码。

统计变量	含义	文字形式的计算方法	为什么统计
user_count	用户总数	统计 seq 中一共有多少个用户	确认分析覆盖的用户规模
event_count	行为事件总数	将所有用户 seq 中的行为条数全部相加	确认分析覆盖的行为规模
history_span	每个用户的历史时间跨度	该用户最后一个事件的 timestamp 减去第一个事件的 timestamp	判断一条用户历史实际覆盖几小时、几天甚至几个月
history_span 分位数	用户历史跨度的分布	将所有用户的 history_span 排序，统计最小值、25%、50%、75%、90%、95%、99%、最大值以及平均值	判断典型用户和长时间跨度用户之间差异有多大
history_span_days	以“天”为单位的历史跨度	把以秒为单位的 history_span 除以一天的秒数	让结果更容易解释
delta_t	同一用户两条相邻行为之间的时间间隔	后一条事件 timestamp 减去前一条相邻事件 timestamp	判断连续行为是几秒、几分钟、几小时还是几天发生一次
delta_t 数量	总共有多少对相邻事件	对每个用户，如果有若干条行为，就产生“行为数减一”个相邻间隔，再对所有用户求和	确认时间间隔样本规模
delta_t 分位数	相邻事件间隔的整体分布	将所有 delta_t 排序，统计最小值、25%、50%、75%、90%、95%、99% 和最大值	判断普通相邻行为与极端长间隔行为的差异
delta_t_zero.count	零时间间隔数量	统计 delta_t 恰好等于 0 的相邻事件对数量	判断是否存在多个行为拥有完全相同时间戳
delta_t_zero.ratio	零时间间隔比例	零间隔事件对数量除以全部相邻事件对数量	衡量相同时间戳现象有多普遍
0_seconds	间隔恰好 0 秒	统计相邻事件间隔等于 0 秒的数量	分析极短时间行为
1_to_59_seconds	间隔 1～59 秒	统计相邻事件时间间隔落在 1～59 秒的数量	判断秒级连续行为多少
1_to_59_minutes	间隔 1～59 分钟	统计间隔在 60 秒到 1 小时以内的数量	判断分钟级行为多少
1_to_23_hours	间隔 1～23 小时	统计间隔在 1 小时到 1 天以内的数量	判断小时级行为多少
1_to_6_days	间隔 1～6 天	统计间隔在 1 天到 7 天以内的数量	判断跨日行为多少
7_to_29_days	间隔 7～29 天	统计间隔在 7 天到 30 天以内的数量	判断明显长期间隔行为多少
30_days_or_more	间隔至少 30 天	统计间隔达到或超过 30 天的数量	判断极端长时间间隔是否存在
每个 gap_band 的 gap_ratio	某时间区间占全部相邻事件的比例	某时间区间内的相邻事件数量除以全部相邻事件数量	判断用户行为主要集中在哪种时间尺度
at_least_7_days	至少相隔 7 天的事件数量	将 7～29 天和至少 30 天两类事件数量相加	单独衡量较长期行为间隔
at_least_30_days	至少相隔 30 天的事件数量	直接统计 delta_t 达到 30 天及以上的数量	衡量极端长期依赖是否存在
negative_gap_count	负时间间隔数量	统计后一条行为 timestamp 小于前一条行为 timestamp 的情况	检查用户序列是否按时间正确排序
## 2.3 stage2_3_item_long_tail.py
主要统计三件事：
① 每个广告被交互了多少次，包括总交互数、曝光数、点击数、未知行为数
→ 判断广告热度分布是否存在明显的长尾现象 / popularity bias，以及是否有必要设置 Popularity baseline。
② 按照广告总交互次数，把广告划分为 tail / mid / head，并统计各组广告数量和贡献的行为量
→ 建立可复用的热门/中部/长尾广告分组，方便后续分别评估模型在 head、tail 上的效果，而不是只看一个总体指标。当前代码使用广告总交互次数的 p50、p90 作为分界。
③ 统计最热门的 Top 1%、5%、10%、20% 广告贡献了多少历史交互
→ 判断用户行为是否高度集中在少量热门广告上，从而量化热门偏置到底有多严重。

统计变量	含义	文字公式 / 计算方法
user_count	用户总数	seq 中用户记录的总数量
event_count	行为事件总数	所有用户行为序列中的事件数量相加
history_span	单个用户整段历史跨度	该用户最后一次行为的 timestamp − 第一次行为的 timestamp
history_span_seconds	所有用户历史跨度的秒级分布	对所有用户的 history_span 统计最小值、P25、P50、P75、P90、P95、P99、最大值和平均值
history_span_days	历史跨度的天级表示	history span 秒数 ÷ 86400
delta_t	同一用户两次相邻行为的时间间隔	后一条行为的 timestamp − 前一条相邻行为的 timestamp
gap_count	相邻行为对总数	对每个非空用户统计“行为数 − 1”，再把所有用户相加
delta_t_seconds 分位数	相邻行为时间间隔分布	对全部 delta_t 统计最小值、P25、P50、P75、P90、P95、P99、最大值；当前代码没有计算均值
delta_t_zero.count	时间间隔为 0 的相邻行为对数量	统计所有满足 delta_t = 0 的相邻行为对
delta_t_zero.ratio	零时间间隔比例	零时间间隔行为对数量 ÷ 全部相邻行为对数量
0_seconds	同时间戳行为	统计 delta_t = 0 的相邻行为对
1_to_59_seconds	秒级间隔	统计时间间隔在 1～59 秒的相邻行为对
1_to_59_minutes	分钟级间隔	统计时间间隔在 1 分钟～不足 1 小时的相邻行为对
1_to_23_hours	小时级间隔	统计时间间隔在 1 小时～不足 1 天的相邻行为对
1_to_6_days	短期跨日间隔	统计时间间隔在 1 天～不足 7 天的相邻行为对
7_to_29_days	较长期间隔	统计时间间隔在 7 天～不足 30 天的相邻行为对
30_days_or_more	超长期间隔	统计时间间隔 大于等于 30 天的相邻行为对
gap_ratio	某时间区间占比	该时间区间内的相邻行为对数量 ÷ 全部相邻行为对数量
at_least_7_days	至少相隔 7 天的数量	7～29 天行为对数量 + 至少 30 天行为对数量
at_least_30_days	至少相隔 30 天的数量	统计所有 delta_t ≥ 30 天 的相邻行为对
negative_gap_count	负时间间隔数量	统计所有满足后一条 timestamp < 前一条 timestamp的相邻行为对
## 2.4 stage2_4_candidate_coldstart.py

主要统计四件事：

① 候选广告中有多少是历史 seen、有多少是历史 unseen
→ 判断候选池中历史未见广告的规模有多大，也就是 ID-only 模型面临多严重的冷启动/泛化问题；同时说明“candidate 无法映射历史 RID”不等于数据错误。代码把 candidate 的 OID 能在 indexer['i'] 中找到定义为 seen，找不到定义为 unseen。

② 分别在 seen / unseen 广告中，统计各匿名字段的 cold_start 取值分布
→ 判断“历史是否见过这个广告”和数据集提供的字段级 cold_start 标记是否一致，避免直接把 unseen 等同于某一个 cold_start 值。源码明确要求只报告各字段的 cold-start 分布，不擅自赋予业务含义。

③ 分别统计 seen / unseen 广告还有多少匿名侧信息 feature_value 可用
→ 判断即使一个广告没有历史 ID 行为，是否仍然能够利用广告侧属性特征进行表示，从而为后续“ID-only vs ID+side feature”的冷启动实验提供依据。

④ 分别统计 seen / unseen 广告有没有有效的 32D 多模态 embedding
→ 判断历史未见广告是否仍可以依靠多模态内容表示进行泛化，从而为后续比较 ID-only、side-feature、multimodal 表示提出实验假设；EDA 本身并不证明多模态一定能解决冷启动。

统计变量	含义	文字形式的计算方法
candidate_count	候选广告总数	统计 candidate 表中的广告记录总数量
seen_count	历史 seen 候选广告数量	统计 candidate 的 OID 能够在 indexer['i'] 中找到的广告数量
seen_ratio	历史 seen 广告占比	seen 广告数量 ÷ candidate 总广告数量
unseen_count	历史 unseen 候选广告数量	统计 candidate 的 OID 无法在 indexer['i'] 中找到的广告数量
unseen_ratio	历史 unseen 广告占比	unseen 广告数量 ÷ candidate 总广告数量
feature_count	candidate 匿名特征字段数	除 item_id、retrieval_id 外，统计具有 cold_start 和 feature_value 子字段的匿名特征字段数量
cold_start_value candidate_count	某字段、某 seen/unseen 组内某个 cold_start 取值出现多少次	在指定匿名字段和 seen/unseen 分组中，统计 cold_start 等于某个值的 candidate 数量；字段为空则单独记为 null
cold_start candidate_ratio_within_group	某个 cold_start 值在该组的占比	该 cold_start 值的广告数量 ÷ 当前 seen 或 unseen 组的广告总数
non_null_feature_value_count	某字段中存在有效侧信息的广告数量	在指定 seen/unseen 组中，统计该匿名字段本身非空、且 feature_value 也非空的 candidate 数量
non_null_feature_value_ratio	某字段侧信息可用率	该字段 feature_value 非空的广告数量 ÷ 当前 seen 或 unseen 组广告总数
available_feature_count	单个 candidate 拥有多少个可用匿名特征	对一个广告逐字段检查，只要字段非空且 feature_value 非空就计 1，最后把所有可用字段数量相加
available_feature_count candidate_count	拥有某一数量可用侧特征的广告数	分别统计“有 0 个、1 个、2 个……可用字段”的 seen/unseen 广告数量
available_feature_count candidate_ratio_within_group	某可用字段数量的广告占组内比例	拥有该数量可用字段的广告数 ÷ 当前 seen 或 unseen 组广告总数
unseen_with_any_side_feature.count	至少拥有一个侧特征的 unseen 广告数量	在所有 unseen 广告中，统计 available_feature_count > 0 的广告数量
unseen_with_any_side_feature.ratio_within_unseen	unseen 中至少有一个侧特征的比例	至少具有一个有效侧特征的 unseen 广告数量 ÷ unseen 广告总数
valid_mm candidate_count	有效多模态 embedding 的广告数量	在 seen / unseen 各组中，统计能够匹配到且通过有效性检查的 32D 多模态 embedding 的 candidate 数量
valid_mm candidate_ratio_within_history_group	有效多模态覆盖率	具有有效多模态 embedding 的广告数量 ÷ 当前 seen 或 unseen 组广告总数
missing_mm candidate_count	缺失有效多模态 embedding 的广告数量	在 seen / unseen 各组中，统计没有有效多模态 embedding 的 candidate 数量
missing_mm candidate_ratio_within_history_group	多模态缺失率	没有有效多模态 embedding 的广告数量 ÷ 当前 seen 或 unseen 组广告总数
## 2.5 stage2_5_multimodal_coverage.py

主要统计三件事：
① 历史广告中有多少广告存在多模态记录，以及其中多少真正拥有合法的 32 维多模态向量
→ 判断 mm_emb 的总体覆盖程度和真实可用程度；不能把“存在一条 mm 记录”直接当成“这个 embedding 可以用于模型”。当前代码把 OID 合法、emb 非空且维度恰好为 32 的记录定义为有效多模态向量。
② 按照 2.3 的 tail / mid / head 广告热度分组，分别统计有效多模态覆盖率
→ 判断多模态缺失是否与广告热度有关，例如长尾广告是不是比热门广告更容易缺失多模态信息；如果不同组覆盖率差异明显，后续评估模型时就不能简单假设多模态缺失是随机的。脚本沿用 2.3 的 p50、p90 分界。
③ 按照 candidate 的 seen / unseen 分组，分别统计有效多模态覆盖率
→ 判断历史未见广告是否仍拥有可用于泛化的多模态信息，以及 unseen 广告是否比 seen 广告存在更严重的多模态缺失；这为后续 zero-fill + mask、可学习 missing embedding、gating 等缺失处理消融提供依据。

统计变量	含义	文字形式的计算方法
item_count	当前分组的广告总数	统计当前 all / tail / mid / head / seen / unseen 分组中的广告数量
mm_id_covered_count	存在多模态记录的历史广告数	对历史广告先通过 indexer.pkl 将 OID 与 RID 对齐，再统计其 OID 能在 mm_emb 中找到记录的广告数量
mm_id_coverage_ratio	多模态 ID 覆盖率	存在 mm_emb 记录的广告数 ÷ 当前分组广告总数
valid_mm_count	拥有有效多模态向量的广告数	统计能够匹配到 mm_emb，并且 OID 合法、embedding 非空、embedding 维度恰好为 32 的广告数量
valid_mm_ratio	有效多模态覆盖率	拥有有效 32D embedding 的广告数 ÷ 当前分组广告总数
missing_mm_count	缺失有效多模态信息的广告数	当前分组广告总数 − 有效多模态广告数
missing_mm_ratio	有效多模态缺失率	缺失有效多模态广告数 ÷ 当前分组广告总数
tail valid_mm_ratio	长尾广告有效多模态覆盖率	tail 组中具有有效 embedding 的广告数 ÷ tail 广告总数
mid valid_mm_ratio	中部广告有效多模态覆盖率	mid 组中具有有效 embedding 的广告数 ÷ mid 广告总数
head valid_mm_ratio	头部广告有效多模态覆盖率	head 组中具有有效 embedding 的广告数 ÷ head 广告总数
history_head_mid_tail_max_minus_min	不同热度组之间最大的多模态覆盖差距	tail、mid、head 三组有效多模态覆盖率中的最大值 − 最小值，最终以百分点表示
candidate all valid_mm_ratio	candidate 整体有效多模态覆盖率	candidate 中拥有有效 embedding 的广告数 ÷ candidate 总广告数
candidate seen valid_mm_ratio	seen candidate 的有效多模态覆盖率	历史 seen candidate 中拥有有效 embedding 的广告数 ÷ seen candidate 总数
candidate unseen valid_mm_ratio	unseen candidate 的有效多模态覆盖率	历史 unseen candidate 中拥有有效 embedding 的广告数 ÷ unseen candidate 总数
candidate_seen_minus_unseen	seen 与 unseen 的多模态覆盖差距	seen candidate 有效多模态覆盖率 − unseen candidate 有效多模态覆盖率，最终以百分点表示
mm_scan_audit.row_count	mm_emb 总记录数	扫描 mm_emb 时累计所有数据行数量
mm_scan_audit.invalid_oid_count	非法 OID 记录数	统计无法形成正整数 OID，即 OID 小于等于 0 的记录数量
mm_scan_audit.null_embedding_count	embedding 为空的记录数	统计 emb 本身为 null 的记录数量
mm_scan_audit.wrong_dimension_count	embedding 维度错误的记录数	统计 embedding 非空但长度不等于 32 的记录数量
mm_scan_audit.valid_embedding_count	合法 embedding 记录总数	统计同时满足 OID > 0、embedding 非空、embedding 长度 = 32 的记录数量

## 2.6 stage2_6_feature_profile.py

主要统计四件事。它的输入已经不再是 seq，而是 user_feat、item_feat 和 candidate 三类结构化侧信息。源码对本阶段的定义就是：逐字段统计数据类型、缺失程度、基数以及列表长度，为后续 Feature Embedding、列表聚合和缺失处理提供依据。
① 每个匿名用户/广告特征是什么数据结构，是单值特征还是列表特征
→ 判断后续应该如何编码特征：单值离散特征可以直接做 Embedding；列表特征则还需要考虑 pooling / aggregation 等聚合方式。
② 每个匿名特征有多少缺失、多少非缺失，覆盖率是多少
→ 判断哪些侧信息比较可靠、哪些字段缺失严重，从而决定后续是否需要缺失值处理、missing mask 或缺失感知建模。
③ 每个匿名特征有多少种不同取值，即特征基数 cardinality
→ 判断离散特征的取值空间有多大，为后续Embedding 表规模、特征编码方式提供依据；不过当前阶段明确不因为“高基数”就直接决定必须 hashing，也不直接确定 Embedding 维度。
④ 对于列表特征，统计每条样本包含多少个特征值，以及空列表有多少；对于 candidate，还统计 feature_value 与 cold_start 的分布
→ 判断列表特征到底是“通常只有一个值”还是“真正的多值特征”，从而决定后续是否值得做列表聚合；同时进一步了解候选广告结构化侧信息的可用程度。

统计变量	含义	文字形式的计算方法
dtype	该字段的数据类型	直接读取 Parquet Schema 中该匿名字段的数据类型，用于判断是单值字段还是 list 字段
row_count	该字段对应的总样本数	将该字段所有数据批次中的记录数量相加
null_count	缺失值数量	统计该字段值为 null 的样本数量
non_null_count	非缺失值数量	总样本数量减去 null 样本数量
non_null_ratio	特征覆盖率 / 非缺失率	非缺失样本数量除以总样本数量
unique_count（单值字段）	单值特征的基数	对该字段所有非 null 单值去重，统计不同取值数量
unique_count（列表字段）	列表特征的元素基数	先将所有非空列表中的元素展开，再对所有非 null 元素去重，统计不同元素数量
list_length_min	列表最小长度	对所有非 null list 样本计算列表元素数量，再取最小值
list_length_median	列表长度中位数	将所有非 null list 的长度排序，取 50% 分位数
list_length_p90	列表长度 90% 分位数	将所有非 null list 长度排序，取 90% 分位位置对应的长度
list_length_p99	列表长度 99% 分位数	将所有非 null list 长度排序，取 99% 分位位置对应的长度
list_length_max	列表最大长度	对所有非 null list 的长度取最大值
empty_list_count	空列表数量	在字段本身不是 null 的样本中，统计列表长度等于 0的样本数
empty_list_ratio	空列表比例	空列表数量除以该字段非 null 样本数量

1.ItemCF / I2I
2.Vanilla Two-Tower
3.序列召回
4.Action/Time-aware
5.MIND/ComiRec 多兴趣
6.Side/MM Item Tower
7.History-strength-aware
8.MIND/ComiRec 多兴趣
9.多路召回融合


