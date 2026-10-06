# Context 蒸馏与可检索参数记忆：最近相关工作

核对日期：2026-10-06。本文补充研究定位，不修改已固定的[五臂尝试协议](context_distillation_protocol.md)。方法描述来自下列原论文；“与本方案的区别”和最后的验证建议是我们的分析，不是原作者替本项目给出的结论。

已有文献分别覆盖了有 context 的 teacher、on-policy 分布监督、hidden-state 对齐、一次前向生成 adapter，以及可检索的参数记忆库。因此，本项目不能将这些单个概念或未经查重的组合直接宣称为首创。

## 六篇最相关工作

### 1. OPCD：直接支持 context-aware on-policy 蒸馏

[Ye et al., *On-Policy Context Distillation for Language Models*, 2026](https://arxiv.org/html/2602.12275)，§3、§3.1。

- **Teacher 上下文：** teacher 获得额外 context；student 不获得该原文。二者在相同的 student 生成前缀上比较下一 token 分布，可以使用同一基座初始化的冻结 teacher。
- **On-policy：** 是。student 先采样，再以 `KL(student || teacher)` 训练；论文实现还使用 student top-k token 的 KL 近似。
- **Hidden 监督：** 主方法未加入 hidden-state 匹配。
- **写入与读取：** 通过训练将 context 诱导的行为写进 student 参数，推理时直接使用更新后的模型。
- **与本方案的区别：** 我们训练可复用的 writer/reader，在线新事实仅生成并写入 VDB 向量；每个 token 再根据层输入召回 VeRA value。OPCD 没有验证这条稀疏读写链路。当前实验使用完整词表 KL，也不是论文实现的逐项复现。

### 2. GKD：用于拆开 rollout 来源与蒸馏目标

[Agarwal et al., *On-policy Distillation of Language Models: Learning from Self-Generated Mistakes*, ICLR 2024](https://arxiv.org/html/2306.13649)，§3、§3.1。

- **Teacher 上下文：** 框架不要求 teacher 拥有 student 缺失的事实 context；核心是同一序列前缀上的 teacher 分布监督。
- **On-policy：** 可选。GKD 显式控制 student 轨迹与固定数据的混合比例，且不反传离散采样过程。
- **Hidden 监督：** 主目标比较 token 分布，不是 hidden-state 回归。
- **写入与读取：** 更新 student 参数；不设计独立记忆条目与检索接口。
- **与本方案的区别：** 它提供训练对照的依据，而非参数记忆架构。原文将轨迹来源与 forward KL、reverse KL、JSD 分开选择，所以 **on-policy 不等于 reverse KL**。我们的 off/on 两组保持相同 reverse KL，才能检验采样前缀的额外作用。

### 3. SADA：最接近“有 context 的状态指导动态 adapter”

[Gao et al., *SADA: Bridging In-Context Learning and Fine-Tuning via State-Aligned Distillation Adapters*, ACL 2026](https://aclanthology.org/2026.acl-long.1046/)，[原论文 PDF](https://aclanthology.org/2026.acl-long.1046.pdf)，§3.2、§3.3。

- **Teacher 上下文：** teacher 使用完整历史，student 使用截断 context 与动态参数更新。
- **On-policy：** 正文训练处理预训练序列或任务目标，未描述以 student rollout 为核心的数据收集机制。
- **Hidden 监督：** 有。Stage I 使用逐层 hidden MSE 与 forward KL；Stage II 使用 hidden MSE 与目标 token 的 LM loss。
- **写入与读取：** 被移出历史的 attention 输出经分块压缩和递归状态更新形成记忆；Meta-LoRA 使用静态 A 与记忆生成的动态 B 注入冻结模型。
- **与本方案的区别：** SADA 已覆盖动态 adapter 的状态与行为蒸馏；其核心记忆是演化状态，未提出我们这种独立 VDB 条目、层输入驱动的逐 token 稀疏 value 召回。我们首轮仅对齐最终 RMSNorm hidden，也不同于其逐层 MSE。

### 4. Doc-to-LoRA：最接近一次前向写入新文档

[Charakorn et al., *Doc-to-LoRA: Learning to Instantly Internalize Contexts*, 2026](https://arxiv.org/html/2602.15902)，§3、§5、Appendix B.1。

- **Teacher 上下文：** 冻结原模型读取文档与问题，为无原文的 adapter student 提供分布目标。
- **On-policy：** 主训练使用预生成的 teacher 回答与保存的 logits；目标为 forward KL，不是 student 在线 rollout。
- **Hidden 监督：** 冻结模型的层激活是 hypernetwork 的输入；这不等同于 hidden-state 蒸馏损失，主目标没有逐层 hidden 匹配。
- **写入与读取：** Perceiver hypernetwork 从 context 层激活生成各层 MLP `down_proj` 的 LoRA A/B；新文档一次前向生成 adapter，随后无须重复输入原文。长文档通过分块生成与 rank 拼接组合。
- **与本方案的区别：** “新 context 无需单独梯度训练”已有直接先例。我们希望存储共享固定 A/B 下的小型 VeRA 向量，并在推理内部动态选择记忆；D2L 主方法生成文档级 adapter，没有这类逐 token VDB 读取。

### 5. Latent Memory Management：参数记忆库与检索已有先例

[Zheng et al., *Context Distillation as Latent Memory Management*, 2026](https://arxiv.org/html/2605.28889)，§3、Appendix C、D。

- **Teacher 上下文：** 每个文档的有 context 基座监督对应的 LoRA student。
- **On-policy：** 文档写入使用合成 QA 与独立 adapter 蒸馏；另报告 reverse KL、top-k logits、EMA 等目标变体。不能仅凭这些目标名称认定所有变体采用相同的 student rollout 协议，原文也未像我们的五臂设计那样独立控制该因素。
- **Hidden 监督：** hidden state 用作候选 adapter 路由特征；不是 teacher/student hidden 对齐损失。
- **写入与读取：** 每个文档单独梯度训练 LoRA。外部文本 embedding 余弦检索 top-k 候选，再结合首 token hidden 与 entropy 选择一个 adapter，并门控是否启用。
- **与本方案的区别：** 不能声称首次检索参数记忆。我们的待验证区别是通用 writer 无梯度写新条目、内部层输入逐 token 查询，以及多个小型 VeRA value 的稀疏混合，而非外部检索后选择一个完整文档 adapter。

### 6. Cartridges：context 蒸馏到可复用的紧凑记忆

[Eyuboglu et al., *Cartridges: Lightweight and general-purpose long context representations via self-study*, 2025](https://arxiv.org/html/2506.06266)，§4.1、§4.2、§5.4、§6。

- **Teacher 上下文：** teacher 读取语料或子语料；student 使用可训练的短 KV cache。
- **On-policy：** 主方法使用合成 self-study conversations 作蒸馏数据，不是当前 student 的在线 rollout。
- **Hidden 监督：** 核心是 forward KL 的下一 token 分布匹配，未加入本方案的 hidden 对齐。
- **写入与读取：** 每个语料单独优化一个 KV cache；推理时加载，多个独立 cartridge 可以拼接使用。训练成本通过后续多次查询摊销。
- **与本方案的区别：** Cartridge 是 attention KV prefix，仍需每语料训练。它不提供通用 writer 一次写新事实，也不使用 VeRA 参数向量的内部稀疏检索。

## 本项目应该如何定位

可以将当前研究问题表述为：

> 能否通过 context-conditioned 行为与状态蒸馏，学习一个通用的写入和读取接口，使新观测被一次编码为共享固定低秩基下的 VeRA 参数向量，并由模型内部层输入逐 token 稀疏检索，在不重复输入观测原文、不更新在线共享权重的条件下使用新事实？

这是一项**组合与效率假设**，不是已经证实的创新性或性能结论。SADA 使 hidden 蒸馏的先例更直接；D2L 覆盖 amortized context-to-adapter；Latent Memory Management 覆盖可检索参数记忆；OPCD 覆盖有额外 context 的 on-policy teacher。当前五臂实验只检验蒸馏训练是否改善已有 VeRA–VDB 接口，不足以证明优于这些完整方法。

## 后续必须补的区分性证据

1. **写入成本与存储量：** 新事实是否确实只需前向写入；每条记忆的 bytes、写入时延、索引成本，以及共享 writer/基矩阵成本均应报告。与逐文档训练 LoRA 的差异不能只用训练结束后的推理时延描述。
2. **读取粒度：** 对比外部 embedding 选一个 adapter、问题级固定召回，以及当前层输入逐 token 动态召回，检验内部寻址是否产生额外收益。
3. **容量与预算：** 单层 rank-64 VeRA 与多层生成 LoRA 的容量不同。应明确参数、记忆、前向计算和训练数据预算，不能把容量差异归因于 OPD 或 hidden loss。
4. **记忆的因果作用：** 新事实写入、更新、删除、多事实组合和干扰记录；比较 real、shuffled、empty 以及强制正确 value。仅 hidden 更相似或训练 loss 更低不能证明记忆有效。
5. **泛化与干扰：** 新实体、新结构表达、真实多 token 值、无关查询及多 seed 验证。当前短答案合成开发集结果只能支持机制层面的有限判断。

最近相关工作提供了试验依据，也说明目标并非仅替换一个蒸馏 loss。若该方向有效，关键证据应是紧凑可写记忆在真实稀疏读取下的行为收益、成本和更新能力。
