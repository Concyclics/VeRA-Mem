# Hash 记忆、初始化与 VeRA-Mem 冷启动设计核验

核验日期：2026-10-05。本文将原论文/公开代码事实与本项目的实验建议分开；本页本身不报告新的 GPU 结果。新试验的配置、日志和结果以相应运行目录为准。

## 结论

扩大训练数据值得做，但更关键的是让 writer、可读出的 value 表示、VeRA 读出和寻址在同一任务上形成稳定接口。随机初始化的非空数据库只有容量，没有事实知识；可用的冷启动应来自合法训练观测编码，或者来自已训练的参数表与配套 reader。Engram/Qwen 的经验支持协同训练、归一化及门控，不证明把随机 value 填进数据库即可解决读出问题。

本项目继续保持用户要求的主链路：**VeRA 层输入产生语义 query → 稀疏查询向量数据库 → 取回 value 调制 VeRA → 已完成的新观测编码为 key/value 写入数据库**。Token hash 仅适合作为独立寻址对照或辅助候选召回；把主方法改成查 token ID 会改变研究问题。

## 官方模型名称与证据边界

用户提到的 **Qwen3.8-Flash 确有官方对应**。Qwen 官方模型卡说明它是基于开源 **Qwen3.8-Flash-Next** 的服务版本。本文核验的是该开源架构、技术报告和公开框架实现，不把未披露的服务端实现视作已知。主实验基座仍按用户指定使用 Qwen3-4B-Instruct-2507。[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、[Qwen 官方仓库](https://github.com/QwenLM/Qwen3.8-Flash-Next)

## Engram：初始化什么，训练什么

Engram 的地址由输入 token 的局部 n-gram 经确定性多头 hash 生成；没有学习“query 到地址”的近邻映射。取回表向量后，才用 hidden state 与投影 key 的归一化内积产生 sigmoid gate，控制线性 value 的注入。其表、投影与骨干在预训练中协同学习。主要比较训练 262B tokens；小规模结构消融也使用 100B tokens。论文附录记录表 embedding 使用 Adam、学习率为基准的 5 倍、weight decay 为 0，短卷积零初始化；未明确披露生产表向量初始化的标准差，也未证明从冻结 Qwen 基座迁移即可零训练使用。[Engram 原论文 §§2、4、6 与附录 A](https://arxiv.org/html/2601.07372v1)

官方公开的是说明数据流的 **demo**，不能混同完整训练代码。固定版本 `fb7f84a21f91223715394a33a1dc24bbfb7f788e` 的证据如下：

| 项目 | 公开代码观察 | 应如何解释 |
| --- | --- | --- |
| Hash | 随 seed/layer 固定的整数乘法、XOR 与质数取模 | hash 参数不通过语义损失学习；避免了当前可学习寻址的共同漂移，但不提供语义改写泛化 |
| 表初始化 | `MultiHeadEmbedding` 直接构造 `nn.Embedding`，未覆盖初始化 | 该 demo 使用 PyTorch 默认分布，不能推断生产训练也如此 |
| Key/value | 可学习线性层；Q/K 使用 RMSNorm | 门控与 value 表示分离；没有对 value 本身施加 `tanh` |
| Gate | 归一化点积除以 `sqrt(d)`，随后 signed-square-root，再 sigmoid | demo 比论文式 (4) 多了 score 变换；复现必须记录使用哪一种 |
| 卷积 | demo 直接创建卷积，没有显式零初始化 | 论文附录列出零初始化，进一步说明 demo 不是完整训练配方 |

[固定版本代码：hash](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py#L188)、[表与读出](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py#L305)、[卷积](https://github.com/deepseek-ai/Engram/blob/fb7f84a21f91223715394a33a1dc24bbfb7f788e/engram_demo_v1.py#L123)

PyTorch 文档规定普通 `nn.Embedding` 的默认权重为 `N(0,1)`。因此这只是上述 demo 的构造结果，不是适合当前 VeRA-Mem 的推荐尺度。大量随机槽位未经训练也不会带来语义知识。[PyTorch Embedding](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Embedding.html)

## Qwen3.8-Flash-Next：能确认的初始化路径

技术报告采用多头 n-gram hash 与上下文门控；该部分消融使用每个激活参数 300 个训练 token。扩大表降低训练 loss，但下游指标并不单调改善。表使用无 weight decay 的 Adam。另一个 **QSA attention indexer** 采用先蒸馏预热、再联合训练；该流程可以启发我们的训练课程，但不能描述成 n-gram 表的初始化算法。[Qwen 技术报告 §§2.1.2、2.3、3.1](https://arxiv.org/html/2608.30320v1)

Qwen 官方仓库主要发布说明和报告，实际公开建模路径位于 Transformers 的 `qwen4_exp`。本次读取固定 commit `f5ab85619d989359ef47b5efed8a91a15045627b`；官方模型配置固定 revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`。

| 项目 | 已核验路径 | 结论 |
| --- | --- | --- |
| Hash 表 | `Qwen4ExpTextNGramEmbedding` 构造 `nn.Embedding` | 固定整数 hash 检索学习过的行，不是 ANN/VDB 语义近邻 |
| 表/线性层初始化 | 模型 `_init_weights` 调用 `PreTrainedModel._init_weights`；父类初始化 embedding/linear 为零均值高斯；官方 config `initializer_range=0.02` | **公开从配置新建模型** 的表初始化为 `N(0,0.02²)`；加载 pretrained checkpoint 会使用已有权重；不能据此断言内部生产训练所有阶段的初始化完全相同 |
| 读出 | value 线性投影，Q/K 规范化后 gate | 与 Engram demo 一样具有 signed-square-root score；没有用 `tanh(value)` 作为存储表示 |
| 卷积/归一化 | PLE 卷积显式零初始化；RMSNorm 权重为零，前向使用 `1 + weight` | 归一化有效增益初始为 1，不能误读成把整条记忆分支置零 |

[表构造](https://github.com/huggingface/transformers/blob/f5ab85619d989359ef47b5efed8a91a15045627b/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1072)、[门控读出](https://github.com/huggingface/transformers/blob/f5ab85619d989359ef47b5efed8a91a15045627b/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1175)、[模型初始化覆盖](https://github.com/huggingface/transformers/blob/f5ab85619d989359ef47b5efed8a91a15045627b/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1325)、[父类初始化](https://github.com/huggingface/transformers/blob/f5ab85619d989359ef47b5efed8a91a15045627b/src/transformers/modeling_utils.py#L2297)、[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)

## 迁移到 VeRA-Mem 的可检验方案

以下均是本项目提出的设计和假设，尚不能作为上述论文的实测结论。

### 1. 把 value 数值稳定性与记忆相关性门控分开

旧式 `tanh(Wv h)` 同时承担幅度限制和内容编码。一旦大多数维度接近 ±1，梯度变小且不同观测可能映射到相同符号向量。建议先比较：

- 旧 `tanh`，作为保留的失败基线。
- 训练集统计中心化/规范化后的线性 value；必要时在向量级限制 RMS，避免逐维截断。
- 同上加独立 memory gate，控制注入强度，gate 输入包含 query 与取回 key 的一致性。

不要同时将输出增益、value encoder 和 gate 全部精确置零。需要至少一条可学习梯度路径。小的非零输出尺度与严格零增量初始化应单独消融；设置上限与监控梯度比直接照搬 `std=0.02` 更有依据。值向量可以有可控范数，但必须记录方差、平均余弦、有效秩及每维梯度，避免“饱和率为零”掩盖所有向量相同。

VeRA 的固定随机 A/B 与共同输出缩放仍可能限制任意 value 的可读性。先保留架构测试 rank 64/256 与 reader 训练样本规模；在相同条件下比较 oracle value 与随机置换。如果 oracle 仍不学习，应测量 value 中是否保留答案信息，以及 `B diag(value) A x` 是否有足够可控方向。可学习低秩 B 或附加 value 残差读出只能作为明确标注的表达能力对照，不能悄然替换用户要求的 VeRA 主方法。

### 2. 先建立读写接口，再让寻址参与联合训练

建议三阶段：

1. **地址预热**：在训练实体上使用问题/观测配对，训练或初始化 Q/K；开发实体、模板独立。记录 Recall@1/@k 与正负样本 margin。
2. **reader 预热**：暂时固定地址编码器，用正确 value 训练 writer/VeRA。用新实体的 oracle 测试确认不是直接记住训练问题。
3. **稀疏协同训练**：恢复真实 top-k，保留地址辅助损失；Q/K 学习率小于 writer/reader。可加入对预热检索分布的蒸馏或参数锚定。先比较冻结 Q/K 与微调 Q/K，再决定哪种用于持续写入。

如果改变 key encoder，就需要重编码/版本化旧 key；不能让库中旧表示与当前 query 默默失配。离线每轮可从训练 support 重新构建 bank；在线阶段先固定 writer/Q/K，只增量写入，保证因果与可解释性。若以后引入 online SGD，须作为额外条件统计写入时间、遗忘和所用监督。

当前只对齐最后一个问题 token，却在所有推理 token 上查询，存在训练/推理位置差异。除首 token Recall@k，还应记录生成期间正确条目驻留率、候选切换、正负 margin 与读取熵；高 Recall@4 并不代表 softmax 给正确 value 足够权重。top-1、top-4、温度和空读 gate 均需在开发集选择。问题级固定检索可以定位此问题，但它只作消融，不替换逐 token query 的主实现。

### 3. 冷启动至少分四个条件

| 条件 | 初始库内容 | 能回答的研究问题 |
| --- | --- | --- |
| Empty | 无条目 | 空库开始的在线写入基线 |
| Random | 与冷启动同数量、同字节数的随机向量 | 非空本身是否有效；也检查噪声敏感性 |
| Train-prefill | 仅训练 support 编码所得 key/value，或在训练集上拟合的表行 | 已学习分布、长期知识和 gate 是否帮助新流；不能包含测试未来事实 |
| Observed-prefill | 流开始前已经合法给出的背景观测 | 已有知识下继续学习；必须把背景输入计入系统资源和信息预算 |

对“新随机实体 → 新随机答案”任务，train-prefill 本身不携带新测试事实；不应期待首次见到实体就能预测隐藏随机答案。其效用主要看合法写入后的读出、抗干扰与稳定性。把全部测试答案先放入数据库，可以作为重测/读取实验，但不能称为写入前泛化或公平冷启动。

更强的训练表初始化可以采用 `v_i = encoder(support_i) + learned_residual_i`，只优化训练表行与 reader，再将行残差蒸馏回 writer；上线新事实仍由 writer 生成 value。必须同时测试未见实体、未见答案组合，防止变成只能读取被直接优化过的旧表。表行的参数量、优化器状态和离线训练成本要计入比较。

### 4. 样本量实验要分开“数据增加”与“训练算力增加”

建议使用嵌套的 128/512/2048/8192 个训练事实，保持开发/在线测试实体不重合；至少对最终候选使用三个训练种子。先用相同优化步数比较样本多样性，再用相同 epoch 比较允许额外算力时的表现，报告总 token、更新次数和 GPU 时间。初步运行可先完成 512/2048 两档，再按开发曲线决定是否继续；这不是保证某个样本数能跨过读出门槛。

训练每个事实应有多种 query/support 模板、无关填充、冲突更新以及多 token 答案；答案随机且不从实体 ID 推导。先在易于诊断的有限答案词表上建立读出机制，再做答案组合、改写与真实会话迁移。所有模板变化需要跨方法共享，不能把新方法的较容易任务与旧方法较困难任务直接比较。

完成判断至少需要：真实检索优于空读和置换值；oracle 与真实检索差距缩小；新实体/改写有效；旧事实保持与冲突更新不退化；容量、延迟、范数/秩及梯度诊断完整。训练 NLL 下降、库非空、value 互不完全相同都不足以单独证明参数化记忆学习成功。

## 对学生应保留的边界

- Hash table 的随机初始化、从 pretrained checkpoint 加载表、从合法观测构建 VDB 是三种不同的冷启动。
- Engram/Qwen 的初始化和优化配方来自不同架构与训练规模；其参数值是参考点，不是本项目有效性的证据。
- 这些静态可训练表主要通过离线梯度学习，不自动等同于推理期追加观测的持续学习系统。
- 本文提出的非饱和 value、训练 bank 暖启和课程式协同训练，必须由本项目新试验验证；负结果也应保留并用于区分容量、寻址、读出与数据规模问题。
