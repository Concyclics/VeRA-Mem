# 事实级训练：减少格式干扰的第一轮验证

现有证据支持**层输入表示与 Q/K/value 编码器对格式敏感**；尚不能将失败完全归因于 VDB 的相似度搜索，更不能证明表示中没有语义信息。本轮已经实现并运行三组针对性的训练探针：全视图训练改善了原问句读取新观测格式的寻址，固定 canonical teacher 改善了 value 与原向量坐标的兼容性，但新问法仍未解决。以下是缓存特征上的编码器实验，**不是生成准确率结果，也没有替换现有可用的同模板 checkpoint**。

完整汇总：[probe.json](results/factcentric/probe.json)。前序端到端实验：[generalization_results.md](generalization_results.md)；表示诊断：[address](generalization_address_diagnostic.md)、[value](generalization_value_diagnostic.md)。

## 如何理解“检索过度关注风格”

VDB 搜索的是编码器输出的 cosine 空间；若 query/key 不能保留跨格式稳定的事实身份，换数据库实现通常不能修复这个问题。前序诊断中，训练模板均值解释了 94.06% 的中心化 query 输入能量；这不是“94.06% 的检索决策来自风格”，也不是语义信息量测量。移除训练模板子空间后，部分新观测检索改善，新问法仍差，说明格式干扰是问题的一部分。

当前增强训练已经独立抽取 query 与 support 的格式，并非直接奖励“同风格即正例”。仍可能存在表示条件不佳、训练视图覆盖不足、有限更新预算以及 writer/reader 坐标漂移等因素。随机实体 ID→16 个单词的任务测试的是事实身份和关联保持，不能直接称为开放域语义理解。

训练需要同时约束两件事：同一事实的不同表述应能相互寻址；不同事实即使写法相同、甚至答案相同，也必须能区分。value 还必须能被既有 VeRA 分支读出；仅让两个可训练视图互相接近，不足以规定它们应保留什么信息。

## 本轮实际实现与运行

使用固定 Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554`，第 20 层 `mlp.down_proj` 的缓存输入，rank/key 维数均为 64。三个学生均从前序 canonical4096 的 1536 步 checkpoint 初始化 Wq/Wk/Wv、A/B/b；统一用训练集全部视图重估 query/support 中心，b 和随机矩阵冻结，只训练 Wq/Wk/Wv。

4096 条训练事实，每条 8 个问句视图、4 个观测视图。每组固定 1600 次 Adam 更新、batch 128、学习率 1e-4、梯度裁剪 1；每个 batch 包含 16 种答案各 8 条不同事实。答案标签只用于组成包含“同答案、不同实体”的负例，不输入 query 编码器。三组的事实曝光顺序完全一致，各 204,800 次；采用固定最后一步，不按开发集选择 checkpoint 或超参数。

| 方法 | 每条事实的编码视图 | 实际目标 |
| --- | --- | --- |
| sampled_pair | 2 个 query、2 个 support | 一对随机视图的双向实体 CE，另加 0.1×成对 Q/K/value 一致性 |
| all_view | 8 个 query、4 个 support | 每个 query 风格×support 风格分别进行双向实体 CE，32 个组合等权平均；相同一致性项 |
| anchored_all_view | 同 all_view | all_view 目标，加固定 canonical teacher 的 Q/K cosine 对齐及 RMS value MSE；三个 anchor 权重均为 1 |

每个风格组合内，所有候选 key 都有相同的观测格式；单靠偏好某种格式不能区分正确实体。所有组合均参与损失，不允许只匹配最容易的一个正视图。teacher 训练目标仅由**训练事实的原模板输入**生成并停止梯度，不使用开发或确认实体训练。

代码：[factcentric_losses.py](../src/vera_mem/factcentric_losses.py)、[probe_factcentric_training.py](../scripts/probe_factcentric_training.py)。损失库还提供将全部视图展开的均匀多正例 CE；**本轮实际运行的是按风格组合分 bank 的 `style_block_retrieval_loss`**，两者不是同一目标。后者没有额外要求不同格式 bank 的 logits 相互校准，未来还需混合格式 bank 评估。

三组事实量和更新数相同，但**计算量不相同**：sampled_pair 编码 query/support 各 409,600 个，全视图两组分别编码 1,638,400 / 819,200 个。只有 all_view 与 anchored_all_view 完全匹配视图数量。H100 上三组训练与诊断耗时分别约 13.62、15.40、16.68 秒；不含缓存构建、teacher 训练和共同准备成本，不能作为端到端运行成本。

## 开发集结果

开发集是未用于训练的 64 个实体。下表每个单元格均为 **R@1 命中数 / 64**；三种问法与三种观测格式交叉。九个条件复用相同实体，不是九组独立样本。旧开发格式已经在前序诊断中查看过，因此本轮属于探索性验证，不能当作新的确认集结果。

| query → support | sampled_pair | all_view | anchored_all_view |
| --- | ---: | ---: | ---: |
| 原问句 → 原观测 | 63/64 | 64/64 | 64/64 |
| 原问句 → 新观测 1 | 31/64 | **39/64** | 38/64 |
| 原问句 → 新观测 2 | 25/64 | **38/64** | 36/64 |
| 新问句 1 → 原观测 | 3/64 | 5/64 | 3/64 |
| 新问句 1 → 新观测 1 | 2/64 | 3/64 | 9/64 |
| 新问句 1 → 新观测 2 | 2/64 | 2/64 | 3/64 |
| 新问句 2 → 原观测 | 7/64 | 2/64 | 3/64 |
| 新问句 2 → 新观测 1 | 3/64 | 2/64 | 2/64 |
| 新问句 2 → 新观测 2 | 2/64 | 1/64 | 1/64 |

新问句 1/2 为保留的请求卡片/前置约束格式，具体文本以缓存协议为准。全视图组在“原问句→新观测”上从 48.44%/39.06% 提升到 60.94%/59.38%；这是局部寻址改善，单 seed、64 个实体不足以推断稳定效应。固定 teacher 没有全面改善寻址，新问句条件依然只有 1–9/64。

另用**训练集 teacher value 按答案平均得到的 16 个原型**，评估学生 value 与 teacher 坐标的兼容性：

| 观测格式 | sampled_pair | all_view | anchored_all_view |
| --- | ---: | ---: | ---: |
| 原观测 | 33/64 | 26/64 | **64/64** |
| 新观测 1 | 12/64 | 7/64 | 19/64 |
| 新观测 2 | 11/64 | 9/64 | 16/64 |

原格式 value 与同一事实 teacher value 的平均 cosine 分别为 0.456、0.411、0.989。这表明 anchor 能固定既有坐标。但前两组的 Wv **仅接受一致性损失，没有 LM 或答案监督**，第三组才有显式 teacher-value 目标。因此这不是公平的“答案语义学习能力”比较，不能据此宣称前两组丢失语义或第三组改善生成。检验学生自身保留的信息，应另外用各学生的训练 value 拟合自己的答案原型或线性诊断头。

本轮未运行 Qwen 前向训练、逐 token decode、真实 VeRA 残差读出、在线连续写入或答案 EM；也未选择或评估确认集特征。前序端到端负结果仍然成立。

## 下一阶段训练方案（尚未执行）

推荐保留本轮的事实级视图组合与实体负例，同时恢复真正约束 value 用途的 LM 损失，分开验证寻址与读出：

```text
L = L_LM(real retrieved values)
    + λ_addr L_style_block_entity
    + λ_anchor (L_Q_teacher + L_K_teacher + L_value_teacher)
    + λ_token L_answer_position_address
    + λ_replay L_LM(canonical replay)
```

1. **增加结构变化，而不仅是换词。** 训练覆盖字段顺序、实体位置、问答与记录格式、长度及无关上下文；同一事实的多个视图配同风格的其他事实负例。对真实多关系数据，以完整事实 ID（实体、关系、时间/版本）定义正例，不能仅按实体或答案合并。
2. **固定可靠的监督目标，同时训练真实读出。** 保留 canonical teacher 与一部分原模板 replay，联合 LM、寻址和 value 对齐，防止 writer/reader 一起漂移。teacher 只提供已验证的同模板坐标，无法凭空赋予未覆盖格式的语义。
3. **在能看到事实身份的位置监督寻址。** 最后一个 prompt token、答案预测位置可用正确记忆监督；不要强迫实体出现之前的所有 token 都检索该实体。实际推理仍由每个 token 的 VeRA 层输入生成 query，不引入外部实体解析器替代主路径。
4. **保持可判别的评估。** 固定预算比较 sampled/all-view/anchor，报告额外编码计算和缓存成本；新确认集同时隔离实体与结构风格，包含混合格式 bank、真实检索/正确地址 oracle/置换 value/零增量，以及更新后保持与多 seed。旧确认集已用于前轮分析，下一轮应另设未查看的确认集。

如果训练视图覆盖和联合 LM 监督后，新问法仍不能稳定寻址，再单独测试非线性编码器、不同层或输入聚合；不要把结构改动与训练损失同时混入一个实验，也不能只靠不断加大一致性权重来判断问题是否可解。

## 依据、复现与审计

- [Khosla et al., Supervised Contrastive Learning, NeurIPS 2020](https://arxiv.org/abs/2004.11362)：提供同身份多个正视图共同监督的依据。本实现以事实身份定义正例，并在实际探针中分风格 bank；没有把论文的分类结论直接套用到 VeRA。
- [Romero et al., FitNets: Hints for Thin Deep Nets, ICLR 2015](https://arxiv.org/abs/1412.6550)：固定 teacher 中间表示可作为学生训练目标。本轮同一基座的跨格式 Q/K/value anchor 是借鉴该思想的研究假设，不是论文已经验证的记忆泛化结论。
- [Bardes et al., VICReg, ICLR 2022](https://arxiv.org/abs/2105.04906)：仅视图一致性允许无信息解，需要保留区分信息的机制。本轮用了实体负例和固定 teacher，没有实现 VICReg，也不声称理论保证不会塌缩。

在已有缓存及 teacher 的工作区内复现（先检查可用设备，将 `GPU_UUID` 设为实际可用 GPU）：

```bash
CUDA_VISIBLE_DEVICES="$GPU_UUID" OMP_NUM_THREADS=4 \
PYTHONPATH="../env_deps:src" python scripts/probe_factcentric_training.py \
  --device cuda \
  --cache ../data/generalization/features_v1.pt \
  --teacher ../runs/generalization_control_20261005/canonical4096/best.pt \
  --output ../runs/factcentric_probe_reproduction/metrics.json
```

本次运行标识 `factcentric_probe_20261006`；实际开始时间为 **2026-10-05 18:30 UTC**，目录名不是时间来源。进程 complete/exit 0，三个 checkpoint 与源码哈希均经核验。公共 JSON 保留数据/模型/源码/权重哈希和训练历史；原始日志、执行源码快照、完整 module checkpoint 留在实验工作区并同步本地，不把模型权重放入代码仓。

缓存 SHA256：`00b94acb30450f31285c5268bb5709485a185c81327cf0dabcf02c1e1c66ab32`。三组事实曝光 SHA256 均为 `f2412f71484a9acadc085acac64d314b1e9e75e6d2740b0fb2a2c03f5b6483c7`。本地完整测试 140 项通过，包括新增 15 项损失、停止梯度、实体负例、采样和编码等价性测试；新增 15 项在 H100 的 Python 环境中以 CPU 执行也全部通过。
