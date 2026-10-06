# QKV、关联记忆与稀疏 VDB：下一轮设计的文献依据

核对日期：2026-10-06。本文为文献与实验建议，**不是已冻结的实验协议，也不声称下述改动已取得效果**。按 Hugging Face papers 技能读取八篇论文的 `.md` 原文，并用 arXiv 与作者仓库核对机制。浏览器不支持部分 markdown 响应时，直接下载相同公开端点阅读；没有依赖第三方综述或 HF 自动生成摘要。

## 最值得先验证的判断

用户指出的“目前更像选择已有向量”触及了当前结果：模型能在已见 A/B 之间切换，但没有把新组合写入转化为正确输出。不过，普通 attention 也对已有 value 做加权组合，**“已有向量的组合”与“attention 式泛化”并不矛盾**。关键是 Q/K 学到可迁移地址，V 保留可解码内容，训练迫使系统使用当前 episode 的绑定，而不是共享参数中的固定实体—标签关系。[Attention §3.2](https://arxiv.org/html/1706.03762v7#S3.SS2)、[Fast Weight Programmers §6.1](https://arxiv.org/html/2102.11174v3#S6.SS1)提供这一区分的依据。

当前代码本来就有 Wq/Wk/Wv、基于实际 VeRA 层输入的 query、top-k 内 softmax 和 value 连续混合，不能说此前没有 QKV 或只有离散标签选择。实际瓶颈尚未被因果定位：[上一轮报告](reconstruction_results.md)中，三槽组 C/D 的 fact R@4 很高，但第三词槽经常没有进入读取；即使整段曾取回第三词槽，完整答案仍错误。地址、读取时刻、value 编码与乘性读出都仍是候选原因。

因此建议优先检验两项：**逐 episode 随机重绑定**，打断实体与旧标签的固定关系；**先检索事实组、再依当前 query 读取组内内容槽**，避免命中前两词就算命中整条事实。它们分别有随机关联召回实验和 LongMem 的直接先例，但迁移到 VDB→VeRA 仍是我们的实验假设。

## 八篇直接相关的一手工作

### 1. Attention Is All You Need（2017）

**论文证据。** §3.2.1 的输出是 `softmax(QKᵀ/√d) V`；§3.2.2 使用多个独立投影后拼接并输出投影；§3.2.3 的 cross-attention 让 query 来自 decoder，K/V 来自 encoder。单 head 的 softmax 混合本身受已取回 value 的凸组合约束，后续输出投影、残差与非线性才进一步变换。[原文 §3.2](https://arxiv.org/html/1706.03762v7#S3.SS2)

**设计含义与边界。** 查询与存储可以来自不同特征域；按生成前缀改变 query、保留位置/角色以及多 head，是有依据的读法。论文不保证更深 QKV 网络能解决我们的组合失败，也没有评估外部 CPU VDB、逐记录改写或低秩乘性 VeRA。加性 `W_O read(q)` 是 attention 路径的诊断，不能当作用户原方案已成功。

### 2. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention（2020）

**论文证据。** §3.2–3.4 将核化 attention 改写为累积关联状态 `S=Σφ(k)vᵀ`，另存归一化状态，query 从该状态读取；因果更新只依赖已观察前缀。固定特征维度下，推理状态大小不随历史长度增长。实验包括合成任务、图像生成与语音识别。[原文](https://arxiv.org/abs/2006.16236)

**设计含义与边界。** 参数化临时记忆不必是静态 embedding 选择，可由新内容前向构建 key–value 关联。不过固定矩阵状态把多条记录叠加，与可逐条替换、审计的 VDB 不同；有限维核也不能直接称与 softmax attention 等价。先把它作为无检索近似的关联读写正对照，不直接替换主线并宣称保留原有编辑语义。

### 3. Linear Transformers Are Secretly Fast Weight Programmers（2021）

**论文证据。** §3 连接线性 attention 与 fast-weight 外积写入；§4.2 用读出的旧值构造误差校正更新；§6.1 每序列随机采样 K/V，最后才给 query。§6.1.2 还允许同 key 重复写入，要求返回最新 value；合成任务中 value 为固定 one-hot，使用回归损失。[原文 §4.2](https://arxiv.org/html/2102.11174v3#S4.SS2)、[§6.1](https://arxiv.org/html/2102.11174v3#S6.SS1)、[作者代码](https://github.com/ischlag/fast-weight-transformers)

**设计含义与边界。** 这是训练随机绑定、检验重写最直接的依据：训练学习使用记忆的规则，测试绑定由新 episode 决定。它不证明 frozen LLM 会解码三词内容；one-hot 回归也不同于自然语言自由生成。文献常用缩写为 **FWP**；本文用该术语，不将“TFW”当成另一个已核实模型名。

### 4. Parallelizing Linear Transformers with the Delta Rule over Sequence Length（DeltaNet，2024）

**论文证据。** §2.2 的写入可写为 `S_t=S_{t-1}+β_t(v_t−S_{t-1}k_t)k_tᵀ`：先读当前 key 对应旧值，再写误差。§3 给出可并行训练算法；§4.1 包含 MQAR 等合成检验；语言模型实验训练 1.3B 参数、100B tokens，并比较含局部/全局 attention 的混合模型。[原文](https://arxiv.org/html/2406.06484v3)、[作者实现 FLA](https://github.com/fla-org/flash-linear-attention)

**设计含义与边界。** 对压缩关联状态，直接累加和按误差替换不同，适合做冲突 key 与旧值残留测试。当前 VDB 已能精确 upsert 单条向量，不需要 delta rule 才能物理删除旧值；本轮错误不等于数据库未更新。若引入 fast-weight 矩阵，应标为另一存储机制，记录碰撞/容量和写入顺序，不能借其大规模 LM 结果保证小 VeRA 成功。

### 5. Gated Delta Networks: Improving Mamba2 with Delta Rule（2024/2025）

**论文证据。** §3.1 在 delta 更新前结合衰减门；§3.2 分开讨论保留与过滤：衰减可能伤害简单 needle 长期保留，而在有大量真实干扰时帮助清理；§3.4 的 Q/K/V 使用线性投影、短卷积、SiLU，Q/K 另做 L2 归一化。§4 比较 1.3B 参数、100B FineWeb-Edu tokens。[原文 §3–4](https://arxiv.org/html/2412.06464v1)、[作者实现 FLA](https://github.com/fla-org/flash-linear-attention)

**设计含义与边界。** QKV 的上下文化和写入/遗忘门应分开消融；“加门一定改善记忆”不受论文支持。它处理序列上的有限 recurrent state，并非持久 VDB 的独立记录。稀疏外部记忆可借鉴归一化、内容相关 query 和更新冲突测试，不能把完整 GatedDeltaNet 的提升归于单独换一个 Wq。

### 6. Memorizing Transformers（2022）

**论文证据。** §3.1 在一层中以当前 query 做外部 kNN，取回 K/V 后重新计算 softmax attention，并以 head-wise gate 与局部 attention 融合；历史 K/V 不可微，逐文档独立。§3.2 讨论训练造成的缓存陈旧与 Q/K 归一化。§4.2 主要实验从头训练 500k 步、每步 2¹⁷ tokens，k=32；§4.5 也测试已有模型适配外部记忆。[原文](https://arxiv.org/abs/2203.08913)、[作者代码 Meliad](https://github.com/google-research/meliad)

**设计含义与边界。** 非可微检索与可学习 attention reader 可以配合，稀疏检索并不等于固定标签选择。但它把 value 融入 attention 激活，不是当 VeRA 参数。本文不能把其语言建模增益等同单条事实更新成功；它也不证明仅用我们的微型训练预算即可得到相同能力。

### 7. Augmenting Language Models with Long-Term Memory（LongMem，2023）

**论文证据。** §2 用冻结骨干编码 K/V，由可训练残差 SideNet 查询与融合，减少 encoder 更新导致的缓存陈旧。§2.3 先用 chunk mean key 选择块，再展开块内 token K/V 做 attention。§3.1 使用 407M 骨干、12 层 SideNet、26B tokens 适配，65k token bank、块长 4，每次取回 16 块共 64 token K/V。[原文 §2.3](https://arxiv.org/html/2306.07174#S2.SS3)、[§3.1](https://arxiv.org/html/2306.07174#S3.SS1)、[论文给出的代码入口](https://aka.ms/LongMem)

**设计含义与边界。** 这是“事实级索引 + 内容槽读取”的直接先例。冻结原文编码、训练 reader 也与缓存复用相关；若 Wk/Wv 仍在训练，投影后缓存仍会陈旧，必须重编码或标版本。其完整 SideNet 比单层 rank64 接口自由度大得多，不能直接复制性能预期；分组检索还改变返回向量数，需单列 bytes 和读出成本。

### 8. Zoology: Measuring and Improving Recall in Efficient Language Models（2023/2024）

**论文证据。** §3.2 将单查询关联召回扩成多 query、不同距离的 MQAR；§4 分析不同 mixer 的召回能力；§5 研究输入相关的稀疏交互。论文指出某些旧合成题容易通过，却不能解释真实语言中的召回差距。作者仓库提供小型合成架构测试。[原文](https://arxiv.org/html/2312.04927)、[作者代码](https://github.com/HazyResearch/zoology)

**设计含义与边界。** 应跨位置、跨 query、跨新绑定测试，不能只看一个固定模板或首预测命中。我们可采用多个相互独立的 episode，让同实体面对不同内容，从而削弱固定映射捷径；这项具体 VeRA 训练设计是我们的假设，不是论文已经验证。MQAR 的 token 关联也不能代替自然语言语义、三词有序重构或持续编辑局部性。

## 机制必须分清：主线与诊断

用列向量记号，令 `h_t` 为当前实际 VeRA 输入，`q_t=f_Q(h_t)`，VDB 中每条记录由观测文本构造 `(k_i,v_i)`。先检索集合 `I_t`，再读 `m_t=Σ_{i∈I_t} softmax(score(q_t,k_i)) v_i`。这是概念简写，归一化、缩放和 gate 必须以新实现/协议为准。

| 路径 | 读出的记忆如何使用 | 与用户目标的关系 | 可以回答什么 |
| --- | --- | --- | --- |
| 动态 VeRA 主线 | `Δh_t = diag(b) B diag(m_t) A h_t`（示意） | 保留 VDB 向量调制参数化增量 | 新内容能否通过同一乘性接口控制推理 |
| 分组 attention→VeRA | 先稀疏选事实组，再由当前 query 读其内容槽，得到 `m_t`，仍按上式调制 | 保留主线，只修改 QKV/读取 | 细粒度寻址与内容读取是否改善可写性 |
| 加性 attention 诊断 | `Δh_t = W_O m_t`，其他条件配平 | 改变注入机制，不是 VeRA 成功 | 如果成功而主线失败，提示需进一步隔离乘性接口限制 |
| fast-weight / delta state | 写 `S`，以 `S q_t` 读取，再选择注入方式 | 另一个存储/更新机制 | 关联学习与覆盖能力；不能继承 VDB 单条删除/恢复保证 |

加性与乘性对照必须报告输出幅值、可训练参数、投影初始化和优化预算，否则成功差异也可能来自尺度或容量。两者都失败时，不能直接说 VDB 无法泛化；两者都成功时，还要看是否依赖当前 value，而不是共享标签。

## 可以判别问题的最小实验建议

以下仅为建议，正式参数、数据种子和门槛需另行封存，不能根据新确认结果补规则。

1. **先训练新绑定，而不先扩大语料。** 小词表和小银行下，每个 episode 随机生成实体—三词内容绑定；同实体在训练中看到多种完整 payload。训练、开发、确认的完整组合与绑定分别封存；开发调架构、确认只验证。固定绑定对照与 episodic 对照按更新、目标曝光和序列长度配平。随机绑定的支持文本是合法写入材料，学生 prompt 仍只包含问题。
2. **分开改地址与内容读取。** 先固定一个主线 baseline，比对完整事实组检索后组内 attention；地址 key 可由支持文本中的实体/关系区间产生，value 来自内容区间。实体文字是合法观测，直接把数组 fact ID 塞给 query 或搜索是额外 oracle 权限。槽位置/角色可来自观测文本，不能从 gold 下一词或正确答案位置指定槽。多 head、非线性 QKV、更大 top-k 不应一次全改。
3. **保持银行大小，分开新组合和新实体。** 先同 16 条银行测试新绑定/新组合；再用同大小银行换新实体。增加 64/256 个干扰记录是另一个轴。这样避免上一轮 entity、payload 和银行大小同时变化。
4. **同检索、同 value 比较两个读出。** 主线保持乘性 VeRA；加性 attention 作为明确命名的诊断。记录每个生成 token 的 token ID、query、目标 group/slot 的检索及权重，把第三词 slot 是否在相关预测时刻被使用，与“整段曾命中”区分。不要将给定 gold 前缀的 NLL 或 R@k 当完整答案成功。
5. **验证当前绑定能覆盖旧绑定。** 固定参数后逐目标 A→新 B→A，加入 value shuffle、空库、同 key 错 value 和非目标邻居；报告新内容 EM、严格 update+restore、单边 real-control 差及正确的邻居保持。整条输出旧 A/B 比例只作事后描述，不能用来筛选确认样本。
6. **先验证 teacher 和缓存合同。** 教师拿原始支持文本、相同问题和相同生成预算；它只用于资格验收。episode 重绑定若直接交换旧的 contextual payload hidden，可能残留旧实体或前缀，不能称为重新编码新 note。要么重新冻结编码实际重绑定文本，要么明确命名为 cached-feature binding 诊断。只冻结骨干不等于投影后 K/V 永远不会陈旧。

最可取的结果不是“某个注意力分数变高”，而是在固定参数和封存新绑定下，新内容完整生成随 VDB 单条更新正确改变，回写恢复且旧事实保持；这一门槛尚待实际实验。上述论文支持尝试这条路线，不足以预言本方案必然成功，也不支持重新扩大语料后再用总 token 数替代机制验证。
