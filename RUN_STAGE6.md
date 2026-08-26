# Stage 6 运行说明

Stage 6 沿用 Stage 3 的 next-click、时间切分和候选池，直接读取 Stage 4 Feature Store，并把 Stage 5 视为只读基线。主实验固定为 `B0 → U1 → U2 → U3 → I1/I2/I3 → E1`；Debug 产物写入 `artifacts/stage6_debug/`，不得用于实验结论。

本机项目环境：

```bat
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 -m unittest discover -s tests\stage6 -v
```

先执行合同和数据 Smoke：

```bat
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_0_contract_audit.py --config configs\stage6.yaml --debug
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_1_sequence_adapter_smoke.py --config configs\stage6.yaml --debug
```

随后按顺序 Debug。用户塔训练后先完成 Validation-only checkpoint 选择，再进入物品塔；I3 选择完成后才训练 E1：

推荐直接使用完整编排入口：

```bat
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 -u scripts\stage6\run_stage6_debug.py --config configs\stage6.yaml --device cuda --overwrite
```

只查看将执行的命令：

```bat
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\run_stage6_debug.py --config configs\stage6.yaml --device cuda --overwrite --dry-run
```

如果在 `i2_index` 失败，修复后可从该步骤继续：

```bat
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 -u scripts\stage6\run_stage6_debug.py --config configs\stage6.yaml --device cuda --overwrite --start-at i2_index
```

完整编排会把每一步状态和耗时写入 `artifacts/stage6_debug/manifests/debug_run_manifest.json`。Debug 只保留会被检索使用的 `best_loss` model-only checkpoint，不保存 Formal 才需要的 `final` optimizer checkpoint。

等价的逐步命令如下：

```bat
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_2_train_user_variants.py --config configs\stage6.yaml --debug
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_5_build_indexes.py --config configs\stage6.yaml --debug --variant U1
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_6_evaluate_variants.py --config configs\stage6.yaml --debug --variant U1
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_5_build_indexes.py --config configs\stage6.yaml --debug --variant U2
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_6_evaluate_variants.py --config configs\stage6.yaml --debug --variant U2
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_5_build_indexes.py --config configs\stage6.yaml --debug --variant U3
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_6_evaluate_variants.py --config configs\stage6.yaml --debug --variant U3
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_3_train_item_variants.py --config configs\stage6.yaml --debug
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_5_build_indexes.py --config configs\stage6.yaml --debug --variant I1
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_6_evaluate_variants.py --config configs\stage6.yaml --debug --variant I1
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_5_build_indexes.py --config configs\stage6.yaml --debug --variant I2
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_6_evaluate_variants.py --config configs\stage6.yaml --debug --variant I2
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_5_build_indexes.py --config configs\stage6.yaml --debug --variant I3
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_6_evaluate_variants.py --config configs\stage6.yaml --debug --variant I3
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_4_train_enhanced_two_tower.py --config configs\stage6.yaml --debug
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_5_build_indexes.py --config configs\stage6.yaml --debug --variant E1
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_6_evaluate_variants.py --config configs\stage6.yaml --debug --variant E1
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_7_compare_ablation.py --config configs\stage6.yaml --debug
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_8_channel_complementarity.py --config configs\stage6.yaml --debug
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_9_fuse_recall.py --config configs\stage6.yaml --debug
```

正式运行前必须人工审查 Debug artifacts。正式配置中的 `session_gap_seconds` 故意保持 `null`，Formal 只认选择审计，手工填写配置不能绕过选择。

Formal 用户塔严格按以下顺序运行：

```bat
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_0_contract_audit.py --config configs\stage6.yaml
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_1_sequence_adapter_smoke.py --config configs\stage6.yaml

REM 分别训练 U1 gap=600/1800/3600；每个 gap 比较 best_loss/final；只按 Validation Recall@100 冻结 gap 和 U1 checkpoint
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_1b_select_session_gap.py --config configs\stage6.yaml

REM 缺少 u1_checkpoint_selection.json 时，U2 会直接失败
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_2_train_user_variants.py --config configs\stage6.yaml --variant U2
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_5_build_indexes.py --config configs\stage6.yaml --variant U2
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_6_evaluate_variants.py --config configs\stage6.yaml --variant U2

REM 缺少 u2_checkpoint_selection.json 时，U3 会直接失败
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_2_train_user_variants.py --config configs\stage6.yaml --variant U3
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_5_build_indexes.py --config configs\stage6.yaml --variant U3
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_6_evaluate_variants.py --config configs\stage6.yaml --variant U3
```

`stage6_1b_select_session_gap.py` 写入：

- `audits/session_gap_selection.json`：三个 gap、各 checkpoint 的 Validation Recall@100、最终选择和 `test_used_for_selection=false`；
- `audits/session_definition.json`：冻结后的 session 定义；
- `manifests/u1_checkpoint_selection.json`：U2 唯一允许读取的 U1 checkpoint。

Formal 的 checkpoint 选择阶段只生成 Validation 指标，不读取 Test。等 E1 完成并产生 `e1_checkpoint_selection.json` 后，才允许显式增加 `--evaluate-test`，例如：

```bat
D:\Anaconda\envs\tencent_rec\python.exe -X utf8 scripts\stage6\stage6_6_evaluate_variants.py --config configs\stage6.yaml --variant E1 --evaluate-test --overwrite
```

Debug 为节省数 GB 的重复 checkpoint/index，继续只使用 `best_loss` 快速跑通，且不能作为正式选择结果。
