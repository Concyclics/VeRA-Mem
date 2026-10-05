# VeRA-Mem 向量参数记忆契约与本轮实验计划

状态：**计划与实现约束，2026-10-05**。本页定义应当实现和验证的系统，不声称计划中的训练或对照已经完成。已执行配置、训练日志、checkpoint、逐样本预测和结果报告是完成状态的依据；如实现改变，须同步更新本页并保留旧配置。

## 用户要求与主方法边界

主方法必须保持以下闭环：

1. 每次推理时，从实际 VeRA 所在层的输入张量生成 query 向量。
2. 用该 query 在 VDB 中稀疏召回一组 value 向量。
3. 取回的向量进入 VeRA 的参数化分支，改变后续神经网络推理。
4. 新输入/已揭示观测经 writer 生成 key/value，写入 VDB，使后续推理能够利用新知识。

Query 不得改成外部文本 ID、已知正确证据编号或答案标签。主分支不把召回文本拼回 prompt，不用另一个答案分类器代替基座输出。Hash 只提供初始化、数值稳定性和训练方法的启发；hash 对照、文本 RAG、oracle 读出和固定 query 消融必须标明为对照，不能代替语义向量检索主方法。相关来源见 [hash 初始化核验](hash_initialization_review.md)。

## 张量与参数定义

基座选用 Qwen3-4B-Instruct-2507，固定模型 revision。当前插入点是某一层 `mlp.down_proj`；因此输入 `x_t` 是这个线性层的真实输入，不是最后一层 hidden state 或另一个模型的 embedding。具体层号、维度、rank 和所有训练设置写入运行配置。

令基座线性层为 `W ∈ R[d_out × d_in]`，其原输出为 `W x_t`。VeRA 固定随机投影为 `A ∈ R[r × d_in]`、`B ∈ R[d_out × r]`，共同输出缩放为 `b ∈ R[d_out]`。数据库保存：

```text
M_s = {(record_id_i, timestamp_i, key_i ∈ R[d_k], value_i ∈ R[r])}
```

主读取前向为：

```text
q_t       = L2Normalize(Wq · normalize_query(x_t))
scores_i  = q_tᵀ key_i                       # key 已 L2 归一化
I_t       = TopK(scores, k)
alpha_t   = softmax(scores[I_t] / temperature)
vbar_t    = Σ(i∈I_t) alpha_ti · value_i
delta_t   = diag(b) · B · diag(vbar_t) · A · x_t
y_t       = W · x_t + delta_t
```

条件于当前输入与读取结果，可以写成：

```text
W_effective(x_t, M_s) = W + diag(b) B diag(vbar_t) A
```

这解释了“向量作为参数化记忆”：value 充当输入相关的 rank 维参数缩放，驱动 VeRA 低秩分支。它并不意味着已经改写基座 `W`，也不意味着每个 VDB 条目都是直接用 SGD 优化的独立参数。由于 `vbar_t` 本身依赖 `x_t`，整个模块是输入相关的非线性映射，不能将整个系统描述为一个固定低秩矩阵。

空库时 `vbar_t=0`，对应零记忆增量。正常主方法在每个有效 prompt/decode token 上使用实际层输入计算 query；固定一次问题级检索只能作为明确标注的消融。一次回答使用同一数据库快照，但每个 token 可以召回不同条目。

当前 CPU reference VDB 使用 exact cosine scan：读取分数的代价为 `O(N d_k)`，随后只混合 top-k value。稀疏 value 访问不等于查询复杂度与库大小无关，也不等于 Engram 的确定性 `O(1)` hash 查表。规模实验必须报告真实检索/传输成本。

## Writer 与因果时序

新观测 `observation_s` 经过冻结基座的一次独立编码，获取相同插入层的输入特征 `h_s`。本轮采用当前约定的 support 表示和选定 token；该选取方式须与 query、teacher-forcing 标签位置区分。

```text
key_s   = L2Normalize(Wk · normalize_support(h_s))
value_s = ValueEncoder(h_s)
M_(s+1) = upsert(M_s, record_id_s, timestamp_s, key_s, value_s)
```

一个在线事件按以下顺序执行：

1. 用现有 `M_s` 回答当前问题并记录写入前预测；此时新事实的答案尚不可被 query、冷启动库或检索过滤器访问。
2. 输入流合法揭示完整观测，例如新事实、用户纠正或已完成的任务反馈。
3. 关闭记忆增量，对该观测提取冻结基座特征，再由已训练 writer 产生 key/value。Writer 此时可以编码新揭示答案，因为它已属于可观测输入。
4. 原子 upsert。已有 record ID 按时间更新，新 ID 追加；同一快照中不保留同一事实的重复正例来人为提高 top-k 权重。
5. 后续查询或明确标注的写入后重测使用 `M_(s+1)`。只读评测不得暗中写入、回放或修改数据库。

记录 ID 用于持久化、更新和评测审计；主 `search` 只接收 query，不得使用正确 ID 过滤候选。冲突更新的可用性取决于合法给出的 record identity；本轮不会把已知 ID 的 upsert 自动宣传成解决了开放文本的实体消歧或矛盾检测。

观测编码时禁用已有记忆，防止旧答案经当前模型反馈污染新 key/value。如果未来适配多个层、更新基座或在线更新 writer，必须重新检查旧 key/value 版本兼容性；不能混用不兼容编码器产生的数据库条目。

## 离线训练、在线写入与 TTT 的区别

| 对象/阶段 | 离线训练接口 | 本轮在线推理与写入 |
| --- | --- | --- |
| Qwen 基座 `W` | 冻结；但插入层之后保留传往模块的反向梯度 | 冻结 |
| 固定随机 `A/B` | 不参与优化 | 不参与优化 |
| `Wq/Wk/Wv/b` | 依课程训练，精确冻结状态记录在配置中 | 全部冻结 |
| 规范化中心 | 仅用训练 query/support 特征拟合 | 固定，不吸收 dev/test/未来观测统计 |
| 初始训练 bank | 每次由当前 writer 对合法训练 support 重算；梯度通过所选 value 和分数训练共享接口 | 从最终 writer 生成并保存的普通 key/value 快照 |
| 新在线记录 | 不适用 | 对合法新观测做前向编码和 upsert；无 optimizer step |

离线 bank 不是与 writer 无关、独立自由优化的 `nn.Parameter` 槽位。Hard top-k 索引本身不可微，但被选中的分数与 value 可以保留计算图；离线训练不能误用会 detach 的持久化 CPU 查询接口作为全部梯度路径。

本轮属于**离线学习读写接口、在线前向写入并修改条件参数记忆**。它具有持续增量存储，但不等同于 TTT 中推理期通过自监督梯度更新 fast weights，也不能称为在线微调了 Qwen。后续若实验 online SGD、delta-rule fast weights 或独立可学习槽位，应作为新的实验条件标明学习信号、状态、梯度和写入预算。

## Stable 变体的计划

本轮 stable 变体保留上述 VeRA 分支，用训练域中心与 RMS-normalized linear value 替代逐维 tanh writer：

```text
u(x)       = x / RMS(x)
z_q(x)     = RMSNormalize(u(x) - mean_train_query)
z_s(h)     = RMSNormalize(u(h) - mean_train_support)
q(x)       = L2Normalize(Wq z_q(x))
k(h)       = L2Normalize(Wk z_s(h))
v(h)       = RMSNormalize(Wv z_s(h))
```

所有分母包含运行配置中的 epsilon。value 的整体 RMS 受控，但单个坐标不强制落在 `[-1,1]`；因此旧方法的 `abs(value)>0.99` 饱和统计不能直接用于 stable。两者应共同报告每维方差、有效秩、样本间余弦、输出范数与梯度，并额外为 tanh 记录导数接近零的比例。

本轮 raw/stable 比较同时改变 query/support 中心化、value 非线性、优化器学习率、检索温度和训练课程，是一个**组合训练配方对照**。Raw-vs-stable 的差异不能单独归因于去除 tanh；若要获得单因素因果结论，需要后续逐项控制这些差异。同一 variant 内的样本规模对照则保持该配方一致。

## 本轮拟执行矩阵

先运行 `stable: train=32, updates=128` 的管线与学习诊断，验证模型、真实层 hook、损失、梯度、数据库读写与保存恢复。该运行地址预热 100 updates，随后 64 updates oracle、64 updates 真实 top-k，以覆盖 prediction-token addressing 和实际检索梯度路径；不保证能够拟合训练集，更不以拟合成功作为已知事实。在线评测使用独立 seed `8042` 生成的实体，实际评测 16 条 stream 与 16 条 control，不消耗主矩阵的 seed `7042` 测试流。它不参与主矩阵的优劣结论。

主比较计划如下，名称为设计标签，正式运行 ID 以保存配置为准：

| 条件 | 训练实体数 | 模块 | LM batch | LM updates | LM target exposures |
| --- | ---: | --- | ---: | ---: | ---: |
| raw-staged-small | 128 | tanh、value-only 中心化；全程 oracle reader | 8 | 512 | 4096 |
| raw-staged-large | 4096 | 同上 | 8 | 512 | 4096 |
| stable-small | 128 | 中心化/RMS value | 8 | 512 | 4096 |
| stable-medium | 1024 | 中心化/RMS value | 8 | 512 | 4096 |
| stable-large | 4096 | 中心化/RMS value | 8 | 512 | 4096 |
| stable-large-extended | 4096 | 同 stable-large；独立训练预算对照 | 8 | 1536 | 12288 |

主矩阵前五个条件的训练实体使用嵌套集合，LM target 顺序采用可复现的无放回遍历并按需循环：128 条重复 32 遍、1024 条重复 4 遍、4096 条一遍。这是相同 target exposures 下的数据多样性比较，尚不是相同总 FLOPs 或 wall time 比较。观测特征构建、alignment、bank 编码和验证成本须另计。

额外预先声明 `extended` profile：固定 4096 个训练实体，将 LM updates 从 512 增到 1536，即同一训练池三遍、12,288 次 target exposures；保持 400 次地址预热、128 条测试、空初始化 bank 和其余配方不变。Oracle/real 分别为 768/768 updates，仍各占一半。它用于回答**更多训练预算**是否有帮助，与样本数增加是不同问题；这是独立初始化的完整训练运行，不把较短运行的最好 checkpoint 当作额外挑选起点。3× 指 LM 更新预算，不代表包含固定预处理/地址预热的端到端成本恰好 3×。

主矩阵、extended 与冷启动对照使用同一组新测试实体 128 条与 control 64 条；control 是未知实体/隐藏随机答案控制，不自动代表拒答评测。开发集必须与训练/测试独立，checkpoint 和阈值只根据训练/开发选择。已观察过测试结果后调整方案，后续确认实验需使用新测试实体，避免反复把测试集当开发集。

主矩阵的实际配置课程如下；运行是否完成仍以日志为准：

1. 地址预热 400 updates，minibatch 128。按完整 batch 计算，地址配对曝光数为 **51,200**；与 LM 的 4,096 target exposures 分开报告。小样本可能每步看到全部 128 条，大样本平均重复 12.5 次，这正是不同数据覆盖条件，不能隐去。
2. Raw 两组延续 staged 基线：地址预热后 Q/K 的更新学习率为 0，512 次 LM updates 全部为 oracle reader；真实评测使用 `top_k=4`、`temperature=0.2`。其作用是检查原配方在等曝光数下扩大数据是否有帮助。
3. Stable 三组使用 256 次 oracle reader updates 建立 writer/VeRA 读出接口，然后 256 次真实稀疏读取 updates；`top_k=4`、`temperature=0.05`。真实阶段对问题与答案预测位置施加地址辅助损失，以更小的 Q/K 学习率共同优化。具体学习率以源码快照与配置为准。Oracle 课程、oracle 诊断成绩都不能代替真实检索主方法的成绩。

初始化 bank 的对照为 **0 与 128 条 train-only bootstrap records**。128 条固定背景来自训练观测，不含测试实体或未来答案。Real-read 离线 episode 持续纳入相同背景，并用当前 writer 重算其 key/value，使初始化分布与 reader 协同学习。Oracle 阶段仅作为课程上界；即使持有背景 bank，强制正确 value 时也不等于在真实背景干扰下检索。

每个 episode 应按 fact ID 对背景与当前支持去重。本轮冷启动试验固定训练池 4096 条，背景 128 条约覆盖 3.125%；不把 bank128 添加到 train128 的训练条件中与之混比。仍应记录每个 target 是否属于初始化 bank，并在需要时分层报告，避免重复背景带来的额外暴露被忽略。

采用训练背景 × 推理初始 bank 的 **2×2 交叉评估**：

| 训练时固定背景 | 推理初始 bank=0 | 推理初始 bank=128 |
| --- | --- | --- |
| 0 | `stable4096`，训练并评测 | `standardtrained_coldinit`，加载同一 `stable4096/best.pt`，仅评测 |
| 128 | `coldtrained_emptyinit`，加载同一 `stable4096cold128/best.pt`，仅评测 | `stable4096cold128`，训练并评测 |

每行共享完全相同 checkpoint，无新增训练步骤，以隔离推理初始化的作用；每列用于比较离线训练时是否接触固定背景的配方差异。两条训练运行的 train size、updates、seed、开发选择规则保持相同。仅评测运行必须同时记录训练 provenance 与评测用 `cold_bank_size`，不能覆盖 checkpoint 中原始训练条件，也不能在测试集上重新选择 checkpoint。

线上每个条件均逐条处理合法新观测并写入 key/value，记录写入前、写入后、延迟回忆、改写和 control；保留真实检索、oracle、value 置换、零增量对照。至少对最终配置进行多种子确认，主试单种子的结果只能用于初步选型。

## 完成与结果解释检查

| 要证明的事项 | 必需证据 |
| --- | --- |
| 实际层输入驱动检索 | hook 捕获输入与实际 query 一致；prompt/decode 均通过主前向；不使用正确 ID 路由 |
| value 驱动参数化分支 | 正确、置换和零 value 的成对预测；动态分支公式与张量形状检查 |
| 在线新事实进入记忆 | 读前/写后数据库内容 hash、record ID、时间戳和新 key/value；评测不变更库 |
| 无未来标签泄漏 | train/dev/test ID 与实体交集检查；初始化 bank 来源；事件顺序与特征提取输入审计 |
| 扩样有效 | 同 target exposures 的完整矩阵；独立开发选择；同测试集成对结果；多种子确认与成本账 |
| 稳定性改进 | 方差/秩/范数、梯度、正负 margin 和检索分布；tanh 专属饱和指标单列 |
| 寻址改善 | Recall@1/@4、正确 value 的 softmax 权重、生成期间驻留/切换；仅 top-4 命中不足 |
| 读出有效 | oracle 在未见实体优于置换/零 value，真实检索接近 oracle；改写不完全失效 |
| 冷启动有效 | train-only bank 的合法信息预算和额外字节；空库/暖库对照、旧事实保持与新事实学习 |
| 可复现与可交接 | 完整配置、模型/源码 revision、随机种子、训练日志、checkpoint、逐样本预测与备份验证 |

结果可支持“扩样有帮助”“stable 组合改善数值行为”“初始化 bank 改善/损害干扰”等相互独立结论。任何一项为负都应如实报告；训练 loss 下降或数据库成功写入本身，不足以得出有用的持续学习能力已成立。
