# VeRA-Mem：文献依据、研究问题与实验设计

核验日期：2026-10-05。基座：`Qwen/Qwen3-4B-Instruct-2507`。

本文给出研究设计与解释边界，不记录尚未发生的实验结果。具体实现、实际运行规模和结果以对应 run 的配置、逐题输出与结果报告为准。标为“后续”的分支不能因出现在本文中就视为已经实现或验证。

主架构是 **memory-conditioned VeRA extension**：VeRA 层的当前输入产生 query，在向量数据库（VDB）中稀疏检索 value；value 作为 VeRA 的 rank 维参数化向量参与当前 token 的计算。完整的新观测通过同一层的输入特征生成 key/value，提交到 VDB 后支持后续推理。其基础参数化来自 [VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454)，动态检索与在线数据库写入属于本项目待验证的扩展。

## 1. 要回答的研究问题

本项目研究：**冻结语言模型主体与 VeRA 的随机矩阵后，能否用层内逐 token 检索得到的向量动态参数化 VeRA，并通过输入生成的 key/value 持续写入外部记忆，在明确的存储、读写成本和能力保持约束下优于简单对照？**

先分开四个容易混淆的问题：

1. **写入能力**：给出正确的新事实后，模型是否能立即答对？
2. **保持能力**：继续写入其他事实后，早期事实是否还可以回答？
3. **寻址能力**：能否从正确的存储单元取回证据，而非只在已知正确位置时成功？
4. **读出能力**：已取回正确的潜在向量后，模型能否解码成答案，并适应新问法？

单次训练损失下降只直接支持第一项的一部分。同题重测改善不等于问法泛化；扩大适配器数量后的改善也不单独证明路由机制有效。主要贡献候选是动态 VeRA 参数向量的可写性、其与 token 输入的交互以及可复现的成本—保持率权衡。不能将“低秩适配 + 检索 + CPU 存储”的组合本身称为首次提出。LoRA bank 和加性 latent reader 仅作为对照或后续扩展，不替代上述主架构。

## 2. 最接近的文献与设计含义

以下仅概括与本项目直接相关的机制。论文中的特定硬件速度、训练规模和模型分数不作为本项目性能预测。

| 工作 | 已有机制 | 对本项目的约束或启发 |
| --- | --- | --- |
| **VeRA**，2023/2024 | 共享冻结随机矩阵 A/B，以可训练向量 b/d 对低秩分支进行参数化 | 本项目将固定的 rank 向量 d 改为 VDB 检索得到的动态 value 聚合，并学习输入到 query/key/value 的映射；这是原始 VeRA 的扩展，不是原论文已有的在线记忆功能。[论文](https://arxiv.org/abs/2310.11454) |
| **GRACE**，2022/2023 | 在隐藏空间维护离散 key–value 编辑码本，用局部检索修正输出，同时冻结主体权重 | “隐藏状态查询外部可写记忆”已有直接先例。必须评估局部性、同义改写与连续编辑，而不仅是记住训练题。[论文](https://arxiv.org/abs/2211.11031)、[作者代码](https://github.com/thartvigsen/grace) |
| **Larimar**，2024 | 增加经过训练的 episodic memory 控制器，支持快速读写及事实修改 | 一次写入不代表无需离线训练。可借鉴“冻结主体、单独训练读写接口”的分工，另计接口训练成本。[论文](https://arxiv.org/abs/2403.11901)、[作者代码](https://github.com/IBM/larimar) |
| **WISE**，2024 | 将原始参数记忆与编辑侧记忆分开，利用路由、知识分片和合并处理持续编辑 | 多 slot 隔离、保留原模型分支与路由门控都需要针对已有编辑系统定位。不能直接将参数分片解释为外部向量记忆。[论文](https://arxiv.org/abs/2405.14768) |
| **M+**，2025 | 基于 MemoryLLM，引入 CPU 长期潜在记忆与联合训练的检索器；读写使用不同 LoRA，取回记忆通过注意力参与生成 | 这是外部潜在记忆、LoRA 读写和 CPU 存储组合的重要近邻。本项目检索值用于调制 VeRA 参数向量，差别在读出接口与预算，不在“首次加入长期向量库”。[论文](https://arxiv.org/abs/2502.00592)、[作者代码](https://github.com/wangyu-ustc/MemoryLLM) |
| **Parametric RAG / PRAG**，2025 | 将文档知识参数化到 FFN 适配参数，在查询时使用检索到的参数知识 | 按文档或事实保存适配器、检索后激活或组合，不是空白方向。需比较写入成本及多适配器干扰。[论文](https://arxiv.org/abs/2501.15915)、[作者代码](https://github.com/oneal2000/PRAG) |
| **Doc-to-LoRA**，2026 | 元训练超网络，将新上下文经前向计算映射成 LoRA，以近似上下文蒸馏 | 后续“向量生成低秩参数”分支与其高度相关。应区分逐事实梯度写入、离线训练后前向写入，不能只比较在线时间而忽略元训练。[论文](https://arxiv.org/abs/2602.15902)、[作者代码](https://github.com/SakanaAI/Doc-to-LoRA) |
| **Understanding LoRA as Knowledge Memory: An Empirical Analysis**，2026 | 系统研究 LoRA 记忆的容量、知识内化及多模块组合；当前 arXiv 为 2026-07-29 的 v5 | 应作为容量、rank 与多模块设计的直接文献依据。需要做总容量匹配和 oracle/实际路由分解，而非只报告多 LoRA 的总分。[论文](https://arxiv.org/abs/2603.01097) |

### 2.1 需要区分的三类“记忆”

| 类别 | 更新了什么 | 与本项目的关系 |
| --- | --- | --- |
| 可追加外部记忆 | 数据库中的文本、向量或单元 | 追加一条记录属于在线非参数记忆；即使参与推理，也不自动成为 test-time training |
| 监督写入/在线适配 | 收到正确答案后写入输入编码向量；对照也可用梯度更新参数 | 前者不要求在线 SGD，但仍使用标签监督；适合受控关联任务与有标签编辑，应明确答案何时成为可用信息 |
| 序列内自监督快权重 | 用当前可观测序列定义的自监督目标更新隐藏状态模型 | 与 TTT、Titans 更接近，需要单独的目标函数、因果性和外循环训练 |

**TTT** 将循环状态本身设为模型，并用自监督学习步骤更新该状态；**Titans** 使用可学习的神经记忆及与关联损失相关的更新、遗忘机制。主架构在线冻结共享网络、追加输入编码向量，属于可写外部记忆。如果写入观测包含刚揭示的标准答案，应称为监督写入，不能因为没有在线 SGD 就称为无监督 TTT。[TTT](https://arxiv.org/abs/2407.04620)、[Titans](https://arxiv.org/abs/2501.00663)

**Engram** 使用基于 token n-gram 的确定性寻址与静态训练记忆表，其主存预取依赖地址能较早确定。隐藏状态完成后才知道查询向量的语义近邻检索具有不同依赖链，不能直接借用 Engram 的低开销结论。[Engram](https://arxiv.org/abs/2601.07372)

### 2.2 补充阅读顺序

1. 低秩参数化与参数效率：[LoRA](https://arxiv.org/abs/2106.09685)、[VeRA](https://arxiv.org/abs/2310.11454)。VeRA 的冻结随机矩阵仍占运行内存，仅报告可训练向量大小不足以代表系统成本。
2. 外部检索记忆：[kNN-LM](https://arxiv.org/abs/1911.00172)、[Memorizing Transformers](https://arxiv.org/abs/2203.08913)。检索后插值输出分布、检索注意力 KV 与注入 FFN 残差属于不同读出方式。
3. 记忆表示稳定性：[LongMem](https://arxiv.org/abs/2306.07174)、[MemoryLLM](https://arxiv.org/abs/2402.04624)。编码器变化后旧向量是否仍可检索，是独立于容量的问题。
4. 训练型稀疏记忆表：[Memory Layers at Scale](https://arxiv.org/abs/2412.09764)。预训练得到的大型表不能直接等同于运行时追加的 episodic memory。
5. 参数生成：[Text-to-LoRA](https://arxiv.org/abs/2506.06105)。任务描述到适配器与事实内容到记忆参数的训练对象不同，应分开比较。

## 3. 统一模型、数据与运行约定

基座固定为 `Qwen/Qwen3-4B-Instruct-2507`，记录实际 Hugging Face commit revision、权重与 tokenizer 来源、软件版本和精度。使用官方 `tokenizer.apply_chat_template`，不手写控制 token。该版本为 non-thinking 模型；评测不依赖思考内容提取。[模型卡](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

官方配置为 36 个 Transformer 层，hidden size 为 2560，MLP intermediate size 为 9728。单个 `mlp.down_proj` 因而是 `9728 → 2560`。单层标准 LoRA 的参数数目为 `r × (9728 + 2560) = 12288r`，不含额外 bias、router 和读写网络。层号统一使用从 0 开始的 Python 索引。[配置](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/main/config.json)

实验清单应锁定：

- `data_seed`、`initialization_seed`、写入顺序与实体/会话划分；正式运行使用至少三个随机种子。
- 主层号、rank、query/key 维数、top-k、温度、support 特征位置/池化、写入与更新规则、离线损失权重、学习率、最大序列长度与生成参数；对照另记录 slot 数与在线更新步数。
- 数据原文件 revision/SHA256、转换代码 commit、模型 revision、配置哈希与实际处理的样本 ID。
- 每个方法独立初始化参数、优化器和记忆；不能连续运行时隐式继承前一方法状态。

训练只对 assistant 答案及约定的终止 token 计算 loss；prompt 和 padding 使用 `-100`。必须在小样本上检查实际被监督的 token、shift 后标签与模板边界。生成采用固定 greedy decoding 与长度上限，同时保存原始文本和规范化答案。格式错误应单独计数，不能靠宽松字符串包含规则掩盖。

## 4. 主架构：输入驱动的 VDB–VeRA 读写

### 4.1 参数化与层内逐 token 读取

设所选线性层的输入为 `x_t ∈ R^din`，原始冻结权重为 `W ∈ R^(dout×din)`。对 Qwen 的 `mlp.down_proj`，`din=9728`、`dout=2560`；这里的 x 是 down projection 实际输入，不是直接用 2560 维 residual hidden state 代替。设置：

```text
A ∈ R^(r×din), B ∈ R^(dout×r)          冻结、随机初始化
b ∈ R^dout                             共享输出缩放向量
Wq, Wk ∈ R^(dk×din), Wv ∈ R^(r×din)   共享读写映射
M^(s) = {(k_i, v_i, metadata_i)}         第 s 版只读 VDB snapshot
```

原始 VeRA 使用 `delta = b ⊙ B[(A x) ⊙ d]`，其中 d 是共享适配参数。本项目用检索得到的 rank 向量 `vbar_t` 取代 d：

```text
q_t = l2_normalize(Wq · norm(x_t))
I_t = topk_i cosine(q_t, k_i),  i ∈ M^(s)
alpha_ti = softmax_i(cosine(q_t, k_i) / temperature), i ∈ I_t
vbar_t = Σ_{i∈I_t} alpha_ti · v_i

delta_t = b ⊙ B[(A x_t) ⊙ vbar_t]
y_t = W x_t + delta_t
```

`norm` 是配置中固定、带数值稳定项的特征归一化；cosine 的 L2 归一化与其区分。A/B 的随机种子、初始化分布和精度均需保存。空 VDB 定义 `vbar_t=0`，因此主分支输出严格退回冻结基座；不隐藏另一个会产生非零输出的静态适配器。若引入额外幅度系数，也应显式出现在公式、配置和日志中。

**query 必须在 VeRA 层 hook 内由每个 token 的实际输入生成。** Prefill 可以批量处理各 token，decode 对每个新增 token 刷新 q、top-k 与 value 聚合；不能先用问题级向量取一次 memory 然后把相同值广播到整个答案，并称其实现了主架构。该问题级缓存版本只能作为单独消融。

一次回答固定的是 `M^(s)`，不是 `I_t`：检索结果可以随 token 变化，但读取期间 memory 内容与共享权重不变。已有 token 在固定快照下生成的 KV 可以按自回归协议使用；提交新 memory 后的新请求应建立新上下文，不能跨 memory 版本无条件复用旧请求的 KV。

### 4.2 输入驱动的写入与在线持续记忆

收到完整 observation 后，用冻结主体在对应层取得 `support_x ∈ R^din`。首版采用预先规定的终止位置或 pooling，并保存其定义；support 特征提取禁用适配增量，使 key/value 来自稳定的原始输入表示。不同 pooling 是消融，不能看测试题后改变。

```text
观察完成后：
  k_new = l2_normalize(Wk · norm(support_x))
  v_new = tanh(Wv · norm(support_x))        # r 维 VeRA 参数向量
  pending = (k_new, v_new, event_id, timestamp, encoder_version)

原子提交：
  append_or_update(M^(s), pending) → M^(s+1)
```

在线阶段冻结基座、A/B、b、Wq/Wk/Wv；持续学习体现在输入生成的 memory 条目追加或更新，不通过每条测试事实的梯度优化共享网络。append 与 update 的语义必须固定：新增事件可追加；更正同一事件可按合法 event ID 更新；是否保留历史版本取决于时间查询协议。不能利用评测答案选择应更新的条目。

完整 observation 到来后才能 commit，新条目不能影响其自身被观察前的预测。合成/MedMCQA 监督轨道中 observation 可包含随后揭示的答案；真实会话轨道中只包含已经到来的正文和日期。答案作为过去观测的一部分写入 memory 是允许的，直接将未来标准答案放进当前 query 则是泄漏。

第一轮只使用一个 VeRA 插入层，保证同层输入的上游主体被冻结。若后续在多个层加入记忆，后层 x 可能已经受前层 memory 影响，需要单独处理 support/query 表示随 memory 快照变化的问题；“参数冻结”本身不足以保证这种多层特征稳定。

### 4.3 离线 episodic 训练：同时学会寻址与读出

在训练实体上构造 support/query episode，并将最终测试实体与训练/开发实体完全分开。support 只含 episode 中合法可见的观测；query 是该事实的另一问法。训练 b、Wq/Wk/Wv，冻结主体与随机 A/B：

```text
L = L_answer_LM + lambda_align · L_query_support_contrastive
                    + 可选的、预先指定的正则项

L_query_support_contrastive：
  拉近 question 前缀在目标层的 query 与匹配 support key，
  用同 episode 的其他实体/属性作为 negatives。

L_answer_LM：
  在层内实际检索和 VeRA 参数调制下预测答案，
  只对 assistant 答案及约定终止 token 计算因果语言模型损失。
```

对比对齐使用问答边界前的最后有效 query 或事先指定的问题位置，不能从完整答案后的 token 取 query。在 teacher-forced LM 训练中，预测下一个 token 时使用标准因果前缀；未来答案 token 不可见。测试生成只使用当前问题和模型已生成前缀。检索质量的主要诊断在答案生成前计算，避免把 teacher-forced 答案前缀的帮助混入无答案检索分数。

硬 top-k 的条目 ID 不可微，但被选中 values、检索分数和下游调制的梯度路径应保留。**top-k=1 时 softmax 恒为 1，LM loss 几乎不能通过该权重训练寻址；显式对比对齐尤为必要。** 离线训练时不能先将 trainable writer 输出 detach 后再期望 Wv 学会写入；在线提交时则应 detach 并保存稳定向量。

冻结主体权重不等于对注入点之后的整个前向使用 `no_grad`，否则共享参数无法经 LM loss 学习。初始化可以将 b 设为零以保持初始基座行为，但应保证 value 分支并非同时严格为零而造成全部 LM 梯度被阻断；保存各参数组梯度范数检查训练是否有效。联合训练与仅对齐、仅 LM 的对照分开报告。

### 4.4 当前实现：CPU 精确 VDB 与稀疏结果传输

pilot 的 key/value 和元数据实际保存在 CPU 的 `PersistentVectorDB` 中，在线读取不维护完整 GPU 索引镜像。每个 token 在 VeRA hook 中生成 query 后传到 CPU，执行精确 cosine top-k 和 softmax value 混合，再把所得 rank 向量传回 GPU 参与 VeRA 运算。Prefill 可批量传输查询，decode 为新增 token 重复查询。读取快照不与未提交写入交错。

这是小规模、同步的 CPU 精确检索参考实现，**不代表已经验证大规模 CPU offload 性能，也不能声称 CPU 检索或跨设备传输近乎免费**。每次精确扫描成本随条目数与 key 维数增长；需实测逐 token 同步开销。离线可微 episode 训练使用 GPU 上的临时 support 张量，不等于在线 VDB 镜像。后续可独立比较 ANN、GPU 索引镜像/cache 与预取，并分解检索、传输和调制成本。

### 4.5 必做机制对照与架构边界

对同一份已写入 memory 比较正确 **oracle 条目、实际逐 token 检索、空 memory、随机错误条目、打乱 value**。Oracle 只能选择已经合法写入的 support，属于诊断上界；不能将答案 token 直接作为检索值或新造证据。错误对照匹配条目数和维数。主方法仍使用实际 query 检索，oracle 不能替代它获得最终主分数。

如果 oracle 失败，优先检查 support 编码、参数向量容量和 VeRA 读出；如果 oracle 成功而真实检索失败，优先检查 Wq/Wk 对齐。若打乱 value 后表现不变，不能宣称模型使用了该记忆；若空 memory 与冻结基座不一致，先检查实现或隐藏的适配分支。

| 对照或扩展 | 与主架构的区别 | 用途 |
| --- | --- | --- |
| 原始静态 VeRA | d 为训练得到的固定向量，没有 VDB | 区分普通参数适配与动态外部记忆 |
| VeRA 参数 bank | 保存多个直接优化的 b/d，query 选择 slot | 检验参数隔离；不同于输入通过 Wv 生成新 value |
| LoRA bank | A/B 可训练，按 slot 做监督梯度更新 | 容量与干扰对照；不作为 VeRA-Mem 的替代实现 |
| 加性 latent reader | `delta=B m`，没有 `(A x)⊙vbar` 参数交互 | 检验乘性参数化是否有必要；属于单独基线 |
| 问题级缓存检索 | 一次检索后复用同一 value | 量化逐 token 寻址的代价和收益；必须明确偏离主设置 |
| 动态输出向量/超网络 | 同时生成 b 或更多权重 | 后续扩展，另计存储、训练量并对照 Doc-to-LoRA |
| 关联损失快权重状态 | 在线更新有限大小状态而非仅追加 VDB | 后续独立分支，不能与当前外部记忆混称 TTT |

## 5. 三阶段数据方案

### 阶段 I：受控新关联与因果单元实验

生成随机实体—属性—值映射，映射由独立置换得到，实体编号不得编码答案。离线 b/Wq/Wk/Wv 训练集、开发集与最终评测集采用实体不重叠、组合不重叠的划分。每个 episode 包含事实 support、另一问法 query 和其他实体的 negatives；最终测试实体仅经在线 input-derived key/value 写入进入系统，不能参与共享参数更新。写入文本与评测改写模板分开。

第一轮可用 32–64 个新事实、4 个写入块以及独立的未写入实体；正式规模依次扩大到 128、512、2048 个事实。小规模目的是暴露缺陷，不作为长期能力结论。至少覆盖：

- 原问题重测与独立问法；若是多选题，随机置换选项并同步答案标签。
- 同实体同属性更新为新值，分别询问“最新值”和过去时间点，避免把正确保留历史当作未更新。
- 相似实体、共享属性和无关干扰写入，用来区分语义混淆与容量不足。
- 未写入实体与缺失属性，检查拒答和错误串联。

答案值可选择自然单词及随机字符串两个难度层级。随机字符串多 token 的精确生成难度较高，应报告长度与 tokenizer 分布；不能只用任意代码生成失败推断全部记忆机制无效。需要长上下文压力时，可借鉴 RULER 的多 key、变量追踪与组合检索设置。[RULER](https://arxiv.org/abs/2404.06654)

### 阶段 II：MedMCQA 监督连续写入

MedMCQA 提供领域选择题和解释，可用于验证有标签知识写入和原能力干扰。它无法排除预训练暴露，因此与合成新关联互补。[论文](https://arxiv.org/abs/2203.14371)、[数据卡](https://huggingface.co/datasets/openlifescienceai/medmcqa)、[作者代码](https://github.com/medmcqa/medmcqa)

从固定 train 子集形成写入流，另用独立开发样本选超参数；validation 中保留一份未用于调参的控制集。首轮可用 64–128 条写入题及 32–64 条控制题，后续扩大。使用规范化 question+options 哈希去重，并保存来源 ID。加载时核验实际 schema 的 `cop` 到 A–D 的映射，不能盲信数据卡叙述。首跑可过滤为 single-choice，简化规范。

主读取输入仅含题干与选项。解释 `exp` 如果参与写入，所有方法都须获得同样解释，并标记为“QA+解释写入”独立条件。题目第二遍重测属于保持率；只有未见改写、选项置换或独立能力保留集才能支持相应的泛化/干扰论断。

### 阶段 III：真实多会话记忆

优先使用 LongMemEval 的 cleaned v1，锁定 revision 和文件 SHA256；官方仓库当前指向 cleaned 数据。先用开发子集验证管线，再冻结协议进行未调参的正式评测。`oracle` 与 S 的完整历史检索属于两个轨道，不合并分数；版本更新也不与既有版本混报。[官方仓库](https://github.com/xiaowu0162/LongMemEval)、[cleaned 数据](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)、[论文](https://arxiv.org/abs/2410.10813)

按真实时间写入历史正文；评测 `answer`、`answer_session_ids` 与证据标签只供 scorer 和检索诊断使用。不得用评测 QA 反向构造写入教材。每题重置 memory；多个问题共享同一用户历史时，必须显式采用 user-level 分组与一致的历史快照。先比较文本 RAG、oracle 文本与潜在 memory，再检查信息抽取、跨会话、时间、更新和拒答的分项表现。

LoCoMo 作为后续外部验证，按完整 conversation 分组划分与统计，避免同一历史下相关问题横跨开发/测试。若仅使用文本部分，需在结果中注明。[作者仓库](https://github.com/snap-research/locomo)、[论文](https://arxiv.org/abs/2402.17753)

公开仓库保存下载与转换代码、版本、样本清单及汇总结果；第三方数据是否分发遵守其许可，私有会话不进入公开测试集。

## 6. 最小公平对照与成本核算

| 对照 | 单层 down_proj 配置 | 共享/总训练参数 | 每次读取的可变信息 | 回答的问题 |
| --- | --- | ---: | ---: | --- |
| Frozen | 无更新、无记忆 | 0 | 0 | 原有能力与格式基线 |
| Text RAG | 固定非训练检索器、top-k、token 预算 | 0 | 检索文本 | 简单外部记忆是否已经足够 |
| Text oracle | 合法历史中的正确证据 | 0 | oracle 文本 | 当前问题在证据充分时是否可回答 |
| 静态 VeRA | 1 个 b/d，冻结随机 A/B | `dout+r` | 固定 b/d | 参数化本身的基线 |
| VeRA-Mem | 共享 b/Wq/Wk/Wv + input-derived VDB | `dout+din×(2dk+r)`，无 bias 时 | 每 token 的 top-k、权重与 rank 向量 | 主架构 |
| VeRA-Mem oracle | 同一 checkpoint、同一 VDB | 与主架构相同 | 正确 support 的 value | 读出诊断上界 |
| LoRA-1 r4 | 1 slot × r4 | 49,152 | 49,152 个激活 LoRA 参数 | 低预算单模块 |
| LoRA-4 r1 | 4 slots × r1 | 49,152 | 12,288 个激活 LoRA 参数 | 总参数相同的分片比较 |
| LoRA-1 r16 | 1 slot × r16 | 196,608 | 196,608 个激活 LoRA 参数 | 高容量单模块 |
| LoRA-4 r4 | 4 slots × r4 | 196,608 | 49,152 个激活 LoRA 参数 | 与 r4 单模块激活量相同、总容量增加的分片比较 |

主架构加入 Wq/Wk/Wv 后不能沿用原始 VeRA 的 `dout+r` 参数数目。例如 `dk=r=64`、`din=9728`、`dout=2560` 时，共享训练参数为 **1,870,336**，另有 **786,432** 个冻结 A/B 元素，以及随记忆数增长的 key/value。若使用 bias、可训练 norm 或其他投影，继续追加计数。在线冻结这些共享参数不意味着离线训练成本为零。

表中的激活参数只是计算规模代理，不等于实际 FLOPs 或墙钟延迟；逐 token Wq、检索、写入时 Wk/Wv、索引和文本处理均额外计量。相同总参数与相同激活计算是两个公平性维度，通常不能用一个对照同时满足。VeRA-Mem 的 query 随 token 变化，不能像静态 VeRA 一样声称将分支一次合并入 W 后没有额外推理开销。

先在同一 r、dk 和共享 checkpoint 下比较 VeRA-Mem 的真实、oracle、empty 与 shuffle 读取，避免容量混淆。LoRA bank 对照可比较 hash、冻结特征路由与 oracle 路由。不同方法可在同一开发集上接受相同次数的超参数尝试；强行使用相同学习率不保证公平。记录离线训练 token/步数、在线写入信息、更新方式与总写入秒数，另外画质量—时间曲线。

存储成本至少分为：共享训练参数、冻结随机矩阵、每事实 key/value、原文与元数据、CPU 权威 VDB、GPU 查询/结果暂存、索引、optimizer state、checkpoint、峰值 CPU/GPU 内存。主架构 value 维数等于 r；每元素 `bytes_per_element` 字节时，仅原始向量为 `N × (dk+r) × bytes_per_element`。索引、元数据和传输暂存应实测追加，不能把此公式当总内存。Faiss 的 `IndexFlatIP` 使用 float32 向量存储，即使原始落盘文件为半精度，也应按实际索引 dtype 计量。[Faiss 索引说明](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)

## 7. 因果协议：先读、后写、再只读评测

```text
离线：在独立训练实体的 episodes 上训练 b/Wq/Wk/Wv
      用独立开发实体选配置；冻结共享参数并保存 checkpoint
在线初始化：加载冻结基座与共享 checkpoint；测试 memory 为空

对每个时间块 t：
  1. pre_update：固定 t 之前的 memory snapshot；层内逐 token 检索并预测
  2. write：完整观测/标签揭示后，提取 support_x 并生成 key/value
  3. commit：仅此时 append/update 权威 VDB；记录 VDB 版本
  4. immediate：在新 snapshot 上只读评测本块与既有块，构成保持率矩阵 R
  5. controls：评测未写入、无关与新问法样本；禁止优化器更新

结束：保存参数与 memory；重新加载；全程只读执行 delayed evaluation
      校验评测前后参数/memory 校验和一致
```

query 来自 VeRA hook 当前 token 的因果输入；问答边界前只含问题，生成期间仅追加已生成 token。完整观测之后取得的 support_x 可以编码已合法揭示的答案，但它只走 write 路径。不可把 support_x 或“完整问答最后 token”当作预测前的 query；也不可用一次问题向量替换逐 token 层内查询。

离线共享网络训练与在线测试不能混淆：b/Wq/Wk/Wv 可以在训练实体上学习“如何写、寻址和读”，但测试实体内容只能经规定 observation 的 key/value 提交进入。用测试答案更新共享参数后再报告未见实体泛化是不成立的。只读评测应同时检查共享权重与 CPU VDB 版本；仅没有调用 optimizer 不能排除隐式记忆写入。

## 8. 指标、统计与诊断

### 8.1 质量指标

- **合成任务**：规范化完整 value 的 exact match，原题与改写分开；多选版本另报候选 log-probability accuracy 与自由生成 invalid rate。子串命中仅作辅助诊断。
- **MedMCQA**：主指标为严格 A–D accuracy；补充候选打分准确率、选项置换测试与独立能力控制集变化。生成准确率和候选排序准确率不是同一指标。
- **会话任务**：按官方任务分类报告，并区分本地 EM/F1、人工核验与官方 judge。未运行官方 judge 不写“官方得分”；如果使用外部付费模型，需单独记录模型版本、成本及授权。
- **检索**：答案生成前边界 token 的 evidence recall@k、MRR；补充生成过程逐 token 的命中与切换轨迹，oracle 与真实检索质量差距。多证据题同时报告至少一个命中与全部必需证据命中。不得把使用 teacher-forced 答案前缀得到的命中率冒充无答案检索性能。

对所有监督答案 token 累加负对数似然：

```text
token_weighted_NLL = Σ_examples Σ_answer_tokens (-log p(token)) / Σ_examples n_answer_tokens
corpus_PPL = exp(token_weighted_NLL)
```

不要把逐题 PPL 的算术平均当作 corpus PPL；长尾值会改变其解释。主损失是否包含终止 token 要固定并注明。RAG 与闭卷方法的答案指标可以比较，但不能将两者不同 prompt 长度的全序列 loss 当同一指标。

### 8.2 保持与遗忘

令 `R[t,b]` 表示写完块 t 后，在块 b 上的只读准确率。报告下三角矩阵及最终行；对于已完成写入的块：

```text
immediate[b] = R[b,b]
final[b] = R[T,b]
forgetting[b] = max_{u=b..T} R[u,b] - R[T,b]
```

同时报告立即到最终的直接差值，因为“历史最好值”可能受采样噪声影响。按距上次写入的事件数/时间分桶画 recall；冲突更新使用指定时间语义评分。记录未触及控制集的前后差值，避免将提升全部建立在原能力下降上。

### 8.3 运行与统计

主架构记录 VDB 条目被检索频率、top-k 权重熵、逐 token 切换率、同一事实不同问法的检索一致性，以及输出增量与原投影输出的范数比。LoRA/VeRA bank 对照另记录每 slot 负载和归一化熵；空 slot 使 max/min 无定义时单独标记。

系统统计包括峰值显存、CPU 权威 VDB 与 GPU 运算/传输暂存各自占用、完整读请求的 p50/p95 延迟、每 token query/检索/调制分项、实际发生的传输时间、生成 tokens/s、每事件写入时间和存储字节。GPU 计时在边界同步；warm-up 与正式样本分开；冷启动、模型加载和稳定推理分别报告。当前同步 CPU 精确检索的延迟只代表该规模，不能外推大型数据库或 ANN 实现。

单 seed 的小样本只用于探索与软件验证。正式实验至少三个 seed，方法共享数据划分及事件顺序。给出各 seed 值与均值；对方法差值使用配对 bootstrap，合成任务按 fact、会话任务按 conversation/user 聚类。跨 seed 的推断同时保留 seed 维度，不能将多个 seed 对同一道题当作互相独立的大样本。主比较固定后再检验，探索性消融与多重比较另行注明。

## 9. 后续确认性实验的预注册草案

以下阈值是**正式多 seed 实验开始前需要冻结的计划**，不是对正在进行或已看过结果的初步尝试所作的事后预注册。若开发集暴露难度不合适，可以修订并记录理由；正式测试后不得据结果更换主指标或阈值。

| 假设 | 预先指定的判断 | 失败后优先处理 |
| --- | --- | --- |
| H0：任务与评测有效 | 合成 oracle 文本 EM 达到 90%；冻结基座无记忆明显低于 oracle；只读评测校验和不变 | 若 oracle 都失败，先修正数据、模板、解码或指标，不解释 memory 优劣 |
| H1：VeRA 参数向量确实传递信息 | 在未见实体上，正确 oracle value 比空记忆/打乱 value 至少高 10 个百分点，配对差值 95% CI 下界高于 0 | 检查 Wv、b 梯度、support 编码、rank 和参数调制；不先扩大 VDB |
| H2：实际寻址可用 | 正式主设置中，答案前边界 token 的真实 recall@k ≥ 90%，且最终 EM 距同一 checkpoint 的 oracle 不超过 10 个百分点 | 改进 Wq/Wk 对比对齐、粒度与温度；检查生成时查询变化 |
| H3：冻结共享参数后能持续写入 | 测试实体仅经 VDB 写入后明显优于写入前；扩大到预定最长流时，首块立即到最终的准确率下降不超过 10 个百分点，未触及控制集下降不超过 2 个百分点 | 分解检索竞争、冲突更新、value 压缩与读出问题；不声称追加存储自动解决遗忘 |
| H4：效率有实际意义 | 固定 GPU 内存或固定完整读延迟预算下，至少一个方法位于质量—成本 Pareto 前沿；计入共享投影、逐 token CPU retrieval 与往返传输 | 若只节省 checkpoint 而读成本更高，改写为存储取舍，不声称整体高效 |

90% 和 10 个百分点等是项目的推进门槛，不是文献公认标准。不同阶段必须采用与其难度匹配、在开发阶段冻结的门槛；不能要求复杂真实会话任务达到合成 oracle 的阈值。

## 10. 分阶段执行清单与学生交接

**第一轮：确认主架构可运行与定位瓶颈。** 固定一个 seed 和小规模合成流，先运行 Frozen、Text oracle 与 Text RAG 检查任务；在独立训练实体上训练共享 b/Wq/Wk/Wv，然后对未见实体执行 input-derived VDB 写入。读取同时比较层内逐 token 真实检索、oracle、empty 与 value shuffle。保存负结果，报告训练实体拟合和未见实体读出是否出现差距；不能用 LoRA bank 结果代替主架构结果。

**第二轮：冻结三个 seed 的核心对照。** 使用三个预定 seed，在相同事实流下比较主方法与静态 VeRA、VeRA/LoRA bank、文本 RAG 及加性 reader。按总训练参数、每事实存储和读写时间分别匹配或画预算曲线，输出逐题结果、遗忘矩阵、配对区间与成本表。层号、rank、对齐损失权重与学习率只在独立开发集选择。

**第三轮：迁移和机制消融。** 引入 MedMCQA 和 LongMemEval 开发轨道。优先消融注入层、r/dk、support pooling、top-k、对比对齐/LM 损失、tanh value 与尺度、动态与静态 rank 向量、逐 token 与问题级缓存查询。大型 VDB、CPU ANN 和 GPU 索引镜像是独立系统扩展。每轮只改变一个主要因素；只有 oracle 读出和实际寻址均通过后，才扩大长期容量。

每个可交接 run 至少包含：配置及哈希、环境清单、代码 commit、数据 ID 清单、逐题预测/标签/检索信息、训练日志、优化器步数、memory 和 adapter checkpoint、聚合指标、成本与复现实验命令。原始模型权重和环境缓存可用 revision 重建；训练产物、实验数据与日志需要独立备份并校验。公开报告只呈现已验证结果，内部设备信息与私有材料不进入仓库。

最终报告应回答具体问题，例如“训练事实的 VeRA value 可以读出，但新事实生成的参数向量尚不能泛化”，或“oracle 可用而逐 token 寻址不稳定”。即使结果为负，也比只报告训练 loss 降低更能指导下一轮实验。
