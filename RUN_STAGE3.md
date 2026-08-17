# TencentGR-1M Stage 3 运行说明

所有可直接执行的 Stage 3 入口都位于 `scripts/stage3/`。共享实现位于
`src/data/` 和 `src/evaluation/`，配置位于 `configs/stage3.yaml`，测试位于
`tests/stage3/`。

请在项目根目录 `D:\AI-file\tecent` 激活已有环境：

```cmd
conda activate tencent_rec
if not exist logs\stage3 mkdir logs\stage3
```

先运行小样本 smoke 链路（输出固定进入 `artifacts/stage3_debug/`）：

```cmd
python -X utf8 -u scripts\stage3\stage3_1_click_exposure_attribution.py --max-users 1000
python -X utf8 -u scripts\stage3\stage3_2_build_next_click_samples.py --max-users 1000
python -X utf8 -u scripts\stage3\stage3_3_temporal_split.py --debug
python -X utf8 -u scripts\stage3\stage3_4_build_eval_candidates.py --debug
python -X utf8 -u scripts\stage3\stage3_5_build_item_strength.py --max-users 1000
python -X utf8 -u scripts\stage3\stage3_6_build_eval_protocol.py --debug
```

若文件已存在且确认需要重建，请显式追加 `--overwrite`。不要混用 debug 与正式
产物。

正式任务必须逐步执行，不要一次性启动整条 pipeline。首先运行 3.1：

```cmd
python -X utf8 -u scripts\stage3\stage3_1_click_exposure_attribution.py
```

检查 `artifacts/stage3/attribution/attribution_report.json` 中的 attribution
coverage 和 gap 分布，确认协议可接受后，再逐个执行：

```cmd
python -X utf8 -u scripts\stage3\stage3_2_build_next_click_samples.py
python -X utf8 -u scripts\stage3\stage3_3_temporal_split.py
python -X utf8 -u scripts\stage3\stage3_4_build_eval_candidates.py
python -X utf8 -u scripts\stage3\stage3_5_build_item_strength.py
python -X utf8 -u scripts\stage3\stage3_6_build_eval_protocol.py
```

最后运行测试：

```cmd
python -X utf8 -m unittest discover -s tests\stage3 -v
```
