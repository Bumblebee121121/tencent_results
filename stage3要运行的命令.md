结果是好的：**12 个 Stage 3 单元测试全部通过，`Ran 12 tests ... OK`**。这说明我们最担心的几类协议错误——target history 泄漏、最近 preceding exposure、timestamp tie、Train-only item strength、Candidate coverage、OID/RID 和 Recall/NDCG——至少在人工构造的测试样例上全部成立。

Codex 当前 `RUN_STAGE3.md` 也明确规定先跑 smoke，再逐步跑正式数据；6 个 Stage 3 入口与你仓库中的脚本一致。([GitHub][1])

下面就是**从现在开始所有后续运行命令**。建议保存下来，但仍然按顺序逐个执行，不要把正式全量任务一次性全启动。

1. **先进入项目和环境**

```bat
D:
cd D:\AI-file\tecent
conda activate tencent_rec

if not exist logs\stage3 mkdir logs\stage3
```

你已经完成单元测试，所以接下来先跑完整的 **Smoke Test 链路**。仓库官方运行说明也是按照 `3.1 → 3.6` 这个顺序。([GitHub][1])

### Stage 3.1：Click → Exposure Attribution，小样本

```bat
python -X utf8 -u scripts\stage3\stage3_1_click_exposure_attribution.py --max-users 1000
```
# 在这里
完成后重点查看：

```text
artifacts\stage3_debug\attribution\attribution_report.json
```

**这一项我建议你跑完先发给我看。**

主要检查：

```text
click_count
attributed_click_count
attribution_failure_count
attribution_coverage
same_timestamp_ratio
multiple_preceding_exposure_ratio
gap 的 median / p90 / p95 / p99 / max
```

如果 3.1 正常，再继续。

### Stage 3.2：构造 next-click 样本，小样本

```bat
python -X utf8 -u scripts\stage3\stage3_2_build_next_click_samples.py --max-users 1000
```

主要检查：

```text
artifacts\stage3_debug\samples\
```

确认有样本产物，并关注：

[
\max(t_{\text{history}})<t_{\text{target exposure}}
]

不应该出现大量 attribution 后却没有合法 history 的异常情况。

### Stage 3.3：时间切分，小样本

```bat
python -X utf8 -u scripts\stage3\stage3_3_temporal_split.py --debug
```

主要查看：

```text
artifacts\stage3_debug\splits\split_manifest.json
```

重点检查：

```text
train sample count
validation sample count
test sample count

train 时间范围
validation 时间范围
test 时间范围
```

以及：

[
\max(T_{\text{train}})
<
\min(T_{\text{val}})
<
\min(T_{\text{test}})
]

### Stage 3.4：Evaluation Candidate Pool，小样本

```bat
python -X utf8 -u scripts\stage3\stage3_4_build_eval_candidates.py --debug
```

查看：

```text
artifacts\stage3_debug\candidates\eval_candidate_manifest.json
```

重点关注：

```text
official_candidate_count
added_validation_target_count
added_test_target_count
final_candidate_count
target coverage
```

最终必须保证：

[
Coverage_{\text{Val}}=100%
]

[
Coverage_{\text{Test}}=100%
]

### Stage 3.5：Train-only Item Strength，小样本

```bat
python -X utf8 -u scripts\stage3\stage3_5_build_item_strength.py --max-users 1000
```

重点查看：

```text
artifacts\stage3_debug\item_strength\
```

尤其是：

```text
item_strength_thresholds.json
val_target_strength_distribution.csv
test_target_strength_distribution.csv
```

不过这里提醒一下：

> **1000 用户的 Head/Mid/Tail/Unseen 比例只能验证代码能不能跑，不能作为项目结论。**

真正有意义的是后面**全量 Stage 3.5**。

### Stage 3.6：Evaluation Protocol，小样本

```bat
python -X utf8 -u scripts\stage3\stage3_6_build_eval_protocol.py --debug
```

检查：

```text
artifacts\stage3_debug\evaluation\evaluation_protocol.json
```

确认至少包含统一：

```text
Recall@10
Recall@50
Recall@100
Recall@500

NDCG@10
NDCG@50
NDCG@100

Overall
Head
Mid
Tail
Unseen
```

Smoke 全部完成后，再跑一次测试：

```bat
python -X utf8 -m unittest discover -s tests\stage3 -v
```

如果此时仍然：

```text
OK
```

就说明：

[
\boxed{\text{Stage 3 Smoke Pipeline = Pass}}
]

然后进入**正式全量 Stage 3**。Codex 的 `RUN_STAGE3.md` 同样要求正式运行必须逐项执行。([GitHub][1])

### 正式 Stage 3.1

```bat
python -X utf8 -u scripts\stage3\stage3_1_click_exposure_attribution.py
```

结果：

```text
artifacts\stage3\attribution\attribution_report.json
```

**这里必须停下来检查。**

正式数据的 attribution coverage 决定：

> 我们自行构造的 `click → preceding exposure` 协议是否可靠。

不要一口气继续跑。

如果结果合理，再：

### 正式 Stage 3.2

```bat
python -X utf8 -u scripts\stage3\stage3_2_build_next_click_samples.py
```

### 正式 Stage 3.3

```bat
python -X utf8 -u scripts\stage3\stage3_3_temporal_split.py
```

查看：

```text
artifacts\stage3\splits\split_manifest.json
```

### 正式 Stage 3.4

```bat
python -X utf8 -u scripts\stage3\stage3_4_build_eval_candidates.py
```

查看：

```text
artifacts\stage3\candidates\eval_candidate_manifest.json
```

### 正式 Stage 3.5

```bat
python -X utf8 -u scripts\stage3\stage3_5_build_item_strength.py
```

这一项是**Stage 3 最值得关注的结果之一**。

你最终要得到：

| Target | Validation | Test |
| ------ | ---------: | ---: |
| Head   |          ? |    ? |
| Mid    |          ? |    ? |
| Tail   |          ? |    ? |
| Unseen |          ? |    ? |

因为 Stage 2 只告诉我们 Candidate 有很多 history-unseen；Stage 3 才真正回答：

[
P(y_{\text{next-click}}\in Unseen)
]

到底是多少。

它会决定后面的：

[
\boxed{\text{History-strength-aware}}
]

到底是项目主线，还是一个次要改进。

### 正式 Stage 3.6

```bat
python -X utf8 -u scripts\stage3\stage3_6_build_eval_protocol.py
```

最后再次：

```bat
python -X utf8 -m unittest discover -s tests\stage3 -v
```

---

如果中间某一步提示：

```text
output already exists
```

而你确认就是要重建，才追加：

```bat
--overwrite
```

例如：

```bat
python -X utf8 -u scripts\stage3\stage3_3_temporal_split.py --debug --overwrite
```

Codex 的运行说明也明确要求已有产物时只有确认需要重建才显式使用 `--overwrite`，并且不要混用 debug 和正式产物。([GitHub][1])

### 你现在实际只需要执行下一条

虽然上面我已经把所有命令给全了，**现在先不要连续粘贴全部命令**。

你下一步执行：

```bat
python -X utf8 -u scripts\stage3\stage3_1_click_exposure_attribution.py --max-users 1000
```

然后把：

```text
artifacts\stage3_debug\attribution\attribution_report.json
```

内容发给我。

当前项目状态可以更新成：

```text
Stage 3 Code       Done
Stage 3 Unit Tests Done（12/12）
Stage 3 Smoke      ← 现在开始
Stage 3 Full       Todo
Stage 4 Features   暂不开始
```

等 `3.1 Smoke` 看完，我们再判断 attribution 规则在**真实 TencentGR-1M 数据**上是不是符合预期。

[1]: https://github.com/Bumblebee121121/tencent_results/blob/main/RUN_STAGE3.md "tencent_results/RUN_STAGE3.md at main · Bumblebee121121/tencent_results · GitHub"
