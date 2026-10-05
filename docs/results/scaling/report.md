# 扩容与冷启动实验汇总

生成时间（UTC）：2026-10-05T17:01:59.864463+00:00

仅列出已完成并通过逐项预测校验的运行。未完成或证据不一致的运行不进入统计；smoke 独立测试实体不混入主表。

所有 EM 表格显示正确数/事实数；本轮属于受控合成事实实验。CI 只描述固定模型下的事实抽样波动，不覆盖训练随机性。

主指标是真实逐 token 检索相对于同一 checkpoint 的 shuffled 与 empty 对照。强制正确 value 诊断（数据字段名 oracle）将同一条正确 value 注入全部 token，改变了真实稀疏读取所形成的训练分布；它不是数学上界，真实检索 EM 可以高于该诊断。单凭该诊断低分，不能断定主要瓶颈是读出无能，也不能确认逐 token 读取错误是否存在补偿效应；这些需要进一步消融。

## 数据规模、训练预算与记忆效果

LM steps 表示来源训练运行的更新预算；括号内为该运行新增更新。selected 是实际被开发集选中的 checkpoint 步数。

| Run / group | Variant | N | LM steps (new) | selected | cold train/deploy | pre real | immediate real | final real | 强制正确 value 诊断 | shuffled | empty |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| scaling_cold_20261005/coldtrained_emptyinit / G1 | stable | 4096 | 512 (0) | 512 | 128/0 | 0/128 (0.0%) | 32/128 (25.0%) | 31/128 (24.2%) | 12/128 (9.4%) | 0/128 (0.0%) | 0/128 (0.0%) |
| scaling_cold_20261005/stable4096cold128 / G1 | stable | 4096 | 512 (512) | 512 | 128/128 | 3/128 (2.3%) | 62/128 (48.4%) | 62/128 (48.4%) | 12/128 (9.4%) | 0/128 (0.0%) | 0/128 (0.0%) |
| scaling_extended_20261005/stable4096long / G1 | stable | 4096 | 1536 (1536) | 1536 | 0/0 | 7/128 (5.5%) | 127/128 (99.2%) | 127/128 (99.2%) | 87/128 (68.0%) | 2/128 (1.6%) | 0/128 (0.0%) |
| scaling_raw_20261005/raw128 / G1 | raw | 128 | 512 (512) | 512 | 0/0 | 0/128 (0.0%) | 1/128 (0.8%) | 0/128 (0.0%) | 118/128 (92.2%) | 0/128 (0.0%) | 0/128 (0.0%) |
| scaling_raw_20261005/raw4096 / G1 | raw | 4096 | 512 (512) | 384 | 0/0 | 0/128 (0.0%) | 6/128 (4.7%) | 0/128 (0.0%) | 127/128 (99.2%) | 0/128 (0.0%) | 0/128 (0.0%) |
| scaling_stable_20261005/stable1024 / G1 | stable | 1024 | 512 (512) | 512 | 0/0 | 2/128 (1.6%) | 48/128 (37.5%) | 51/128 (39.8%) | 15/128 (11.7%) | 0/128 (0.0%) | 0/128 (0.0%) |
| scaling_stable_20261005/stable128 / G1 | stable | 128 | 512 (512) | 512 | 0/0 | 0/128 (0.0%) | 13/128 (10.2%) | 17/128 (13.3%) | 8/128 (6.2%) | 0/128 (0.0%) | 0/128 (0.0%) |
| scaling_stable_20261005/stable4096 / G1 | stable | 4096 | 512 (512) | 512 | 0/0 | 3/128 (2.3%) | 55/128 (43.0%) | 56/128 (43.8%) | 15/128 (11.7%) | 0/128 (0.0%) | 0/128 (0.0%) |
| scaling_stable_20261005/standardtrained_coldinit / G1 | stable | 4096 | 512 (0) | 512 | 0/128 | 3/128 (2.3%) | 53/128 (41.4%) | 54/128 (42.2%) | 15/128 (11.7%) | 0/128 (0.0%) | 0/128 (0.0%) |

pre/immediate 没有测强制正确 value 诊断、shuffled、empty，不应把缺失读成零。

## 寻址、问法、控制与 value 分布

| Run | final R@1 / R@4 | decode residency / switches per example | paraphrase real / 强制正确 value 诊断 | control before / after | value effective rank | unique values / records |
| --- | --- | --- | --- | --- | ---: | --- |
| scaling_cold_20261005/coldtrained_emptyinit | 100.0% / 100.0% | 17.6% / 1.59 | 0/128 (0.0%) / 0/128 (0.0%) | 0/64 (0.0%) / 0/64 (0.0%) | 7.38 | 128 / 128 |
| scaling_cold_20261005/stable4096cold128 | 100.0% / 100.0% | 29.8% / 1.54 | 0/128 (0.0%) / 0/128 (0.0%) | 5/64 (7.8%) / 4/64 (6.2%) | 7.77 | 256 / 256 |
| scaling_extended_20261005/stable4096long | 100.0% / 100.0% | 94.4% / 0.45 | 0/128 (0.0%) / 0/128 (0.0%) | 0/64 (0.0%) / 3/64 (4.7%) | 9.79 | 128 / 128 |
| scaling_raw_20261005/raw128 | 40.6% / 72.7% | 4.4% / 1.70 | 0/128 (0.0%) / 0/128 (0.0%) | 0/64 (0.0%) / 0/64 (0.0%) | 6.91 | 128 / 128 |
| scaling_raw_20261005/raw4096 | 50.8% / 88.3% | 7.0% / 2.20 | 0/128 (0.0%) / 0/128 (0.0%) | 0/64 (0.0%) / 0/64 (0.0%) | 5.58 | 128 / 128 |
| scaling_stable_20261005/stable1024 | 100.0% / 100.0% | 24.9% / 1.62 | 0/128 (0.0%) / 0/128 (0.0%) | 0/64 (0.0%) / 1/64 (1.6%) | 8.07 | 128 / 128 |
| scaling_stable_20261005/stable128 | 50.8% / 78.9% | 11.7% / 1.70 | 0/128 (0.0%) / 0/128 (0.0%) | 0/64 (0.0%) / 1/64 (1.6%) | 10.06 | 128 / 128 |
| scaling_stable_20261005/stable4096 | 100.0% / 100.0% | 21.4% / 1.76 | 0/128 (0.0%) / 0/128 (0.0%) | 0/64 (0.0%) / 2/64 (3.1%) | 7.28 | 128 / 128 |
| scaling_stable_20261005/standardtrained_coldinit | 100.0% / 100.0% | 14.2% / 1.87 | 0/128 (0.0%) / 0/128 (0.0%) | 4/64 (6.2%) / 3/64 (4.7%) | 7.39 | 256 / 256 |

R@k 取实际生成前的最后 prompt token；decode residency 是正确记录出现在 decode top-4 中的比例，按实际 decode 查询数加权，不是 top-1 比例或 attention 权重。switches 为每例 top-1 变化次数，含 prefill 到首次 decode。缺少轨迹的旧运行显示 —。控制集是未观测随机事实，其 EM 不等于拒答能力。value 有效秩/唯一数描述数值多样性，不单独证明记忆可用。

## 按事实严格配对的差值

差值方向为 first − second，以百分点表示。ID 集合和标签必须完全一致；不取交集。

`deployment_cold128_minus_0_same_checkpoint` 只改变部署初始库，要求 checkpoint SHA 和来源训练配置完全一致，并分别保留 training cold=0/128 条件。不同 cold 训练产生的模型不会被当作同权重部署对照。`stable4096_LMschedule1536_minus_512` 比较 3 倍 LM schedule：oracle 与真实检索阶段同比放大、对齐保持 400 步。这是组合训练预算对照，不是等 FLOPs 比较，也不能声称总计算量恰为 3 倍。

| Contrast | first | second | paired N | Δ EM (pp) | 95% percentile CI (pp) |
| --- | --- | --- | ---: | ---: | --- |
| final_real_minus_empty | scaling_cold_20261005/coldtrained_emptyinit | scaling_cold_20261005/coldtrained_emptyinit | 128 | +24.22 | [+17.19, +32.03] |
| final_real_minus_shuffled | scaling_cold_20261005/coldtrained_emptyinit | scaling_cold_20261005/coldtrained_emptyinit | 128 | +24.22 | [+17.19, +32.03] |
| final_real_minus_empty | scaling_cold_20261005/stable4096cold128 | scaling_cold_20261005/stable4096cold128 | 128 | +48.44 | [+39.84, +57.03] |
| final_real_minus_shuffled | scaling_cold_20261005/stable4096cold128 | scaling_cold_20261005/stable4096cold128 | 128 | +48.44 | [+39.84, +57.03] |
| final_real_minus_empty | scaling_extended_20261005/stable4096long | scaling_extended_20261005/stable4096long | 128 | +99.22 | [+97.66, +100.00] |
| final_real_minus_shuffled | scaling_extended_20261005/stable4096long | scaling_extended_20261005/stable4096long | 128 | +97.66 | [+94.53, +100.00] |
| final_real_minus_empty | scaling_raw_20261005/raw128 | scaling_raw_20261005/raw128 | 128 | +0.00 | [+0.00, +0.00] |
| final_real_minus_shuffled | scaling_raw_20261005/raw128 | scaling_raw_20261005/raw128 | 128 | +0.00 | [+0.00, +0.00] |
| final_real_minus_empty | scaling_raw_20261005/raw4096 | scaling_raw_20261005/raw4096 | 128 | +0.00 | [+0.00, +0.00] |
| final_real_minus_shuffled | scaling_raw_20261005/raw4096 | scaling_raw_20261005/raw4096 | 128 | +0.00 | [+0.00, +0.00] |
| final_real_minus_empty | scaling_stable_20261005/stable1024 | scaling_stable_20261005/stable1024 | 128 | +39.84 | [+31.25, +48.44] |
| final_real_minus_shuffled | scaling_stable_20261005/stable1024 | scaling_stable_20261005/stable1024 | 128 | +39.84 | [+31.25, +48.44] |
| final_real_minus_empty | scaling_stable_20261005/stable128 | scaling_stable_20261005/stable128 | 128 | +13.28 | [+7.81, +19.53] |
| final_real_minus_shuffled | scaling_stable_20261005/stable128 | scaling_stable_20261005/stable128 | 128 | +13.28 | [+7.81, +19.53] |
| final_real_minus_empty | scaling_stable_20261005/stable4096 | scaling_stable_20261005/stable4096 | 128 | +43.75 | [+35.16, +52.34] |
| final_real_minus_shuffled | scaling_stable_20261005/stable4096 | scaling_stable_20261005/stable4096 | 128 | +43.75 | [+35.16, +52.34] |
| final_real_minus_empty | scaling_stable_20261005/standardtrained_coldinit | scaling_stable_20261005/standardtrained_coldinit | 128 | +42.19 | [+33.59, +50.78] |
| final_real_minus_shuffled | scaling_stable_20261005/standardtrained_coldinit | scaling_stable_20261005/standardtrained_coldinit | 128 | +42.19 | [+33.59, +50.78] |
| N4096_minus_N128_matched_updates | scaling_raw_20261005/raw4096 | scaling_raw_20261005/raw128 | 128 | +0.00 | [+0.00, +0.00] |
| N4096_minus_N128_matched_updates | scaling_stable_20261005/stable4096 | scaling_stable_20261005/stable128 | 128 | +30.47 | [+21.88, +39.06] |
| deployment_cold128_minus_0_same_checkpoint | scaling_cold_20261005/stable4096cold128 | scaling_cold_20261005/coldtrained_emptyinit | 128 | +24.22 | [+16.41, +32.03] |
| deployment_cold128_minus_0_same_checkpoint | scaling_stable_20261005/standardtrained_coldinit | scaling_stable_20261005/stable4096 | 128 | -1.56 | [-3.91, +0.00] |
| stable4096_LMschedule1536_minus_512 | scaling_extended_20261005/stable4096long | scaling_stable_20261005/stable4096 | 128 | +55.47 | [+46.09, +64.06] |

Bootstrap：10,000 次，随机种子 123。未进行多重比较校正；单训练种子的区间不能支持跨训练随机性的显著性结论。

## 可比性与排除记录

同组 G 编号表示 feature cache、运行模块源码和模型 revision 相同；跨组不做样本规模因果比较。

- G1: cache `e1f6c0165e01bbb6f9196fd3795bdab10eb2b8ba88d2be40aa8e4a1d7d935dff`；source `dc4ee157f238fb27aebbbc313d48ad68b3a118dad3ed9f6cfee24126603d11c5`；model `cdbee75f17c01a7cc42f958dc650907174af0554`。

- 未配对 scaling_cold_20261005/stable4096cold128 与 scaling_stable_20261005/stable128：Cross-run matching failed: training_cold_records。

- 未配对 scaling_extended_20261005/stable4096long 与 scaling_stable_20261005/stable128：Cross-run matching failed: training_lm_updates。

- 未配对 scaling_stable_20261005/standardtrained_coldinit 与 scaling_extended_20261005/stable4096long：Same-checkpoint deployment matching failed: checkpoint_sha256。

- 未配对 scaling_extended_20261005/stable4096long 与 scaling_cold_20261005/stable4096cold128：Training-budget matching failed: training_cold_records。

- 未纳入 scaling_smoke_20261005/stable32：Smoke run excluded (independent evaluation entities, seed 8042)。

公开 JSON 仅含聚合结果、证据 hash 与对照关系，不含逐例答案、预测、内部绝对路径、模型权重或连接配置。
