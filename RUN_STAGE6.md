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

正式运行前必须人工审查 Debug artifacts。由于现有 Stage 2 EDA 没有冻结 session gap，正式配置中的 `user_tower.short_session.session_gap_seconds` 故意保持 `null`；先在 Validation-only 候选 `[600, 1800, 3600]` 中完成选择并写回配置，否则正式脚本会主动终止。正式索引脚本会同时处理 `best_loss` 与 `final` checkpoint，评估脚本只用 Validation Recall@100 选择并写入 `<variant>_checkpoint_selection.json`，然后才对选中 checkpoint 运行 Test。Debug 为节省约数 GB 的重复索引，只验证 `best_loss` 单候选，结果不可用于正式选择。
