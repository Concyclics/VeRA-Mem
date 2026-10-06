# 整块 value 读取与动态低秩记忆：文献依据

本轮在[已有 QKV 文献映射](qkv_literature.md)基础上，只核对三个最直接的一手来源。结论是：**整块召回有合理先例，记忆构造动态矩阵也有严格理论联系；但仅把共享线性投影移到混合之前，不会增加表达能力，也没有论文保证这能解决我们受控新组合任务的失败。** 本轮的具体可证伪设计见 [block_protocol.md](block_protocol.md)。

## 先区分三个不同操作

单query、单head的标准attention仍输出一个向量：`m=Σ_s α_s v_s`。对共享线性B以及固定当前输入`z=Ax`，有：

```
B (Σ_s α_s v_s) = Σ_s α_s B v_s
B [(Σ_s α_s v_s) ⊙ z] = Σ_s α_s B (v_s ⊙ z)
```

在同一权重、没有中间非线性或slot专属映射时，后移求和/前移B只是等价计算顺序。召回整块后照样运行上述式子，不足以解释为新容量。所有reader最终都需产生向量供下一层使用；应检验**输入相关算子的形式**，不能把“最终是向量”本身作为失败根因。

本轮使用每槽生成的外积`w(v_s)u(v_s)ᵀ`作用于z。这允许非对角坐标混合，区别于原`diag(m)z`。对照保留共同`diag(m)`，比较`w(meanV)u(meanV)ᵀ`与`mean_s[w(v_s)u(v_s)ᵀ]`；两者参数完全相同。由于u含归一化与bias，这个对照同时涉及因子生成与池化的顺序、外积项rank，不能将结果唯一归于rank或矩阵存储形式。

## 1. Fast Weight Programmers：动态矩阵可以由内容外积构造

**Schlag, Irie & Schmidhuber，Linear Transformers Are Secretly Fast Weight Programmers，ICML 2021。** [原文 §3.1–3.2](https://arxiv.org/html/2102.11174v3#S3)，[§4.1–4.2](https://arxiv.org/html/2102.11174v3#S4)，[作者代码](https://github.com/ischlag/fast-weight-transformers)。

论文把去softmax的attention写成`V(Kᵀq)=(VKᵀ)q`；其中`W=Σ v kᵀ`是由当前内容生成的fast-weight矩阵，而普通softmax形式仍是value的加权和。其后分析有限维关联记忆的key干扰，并引入基于旧读出误差的delta写入。合成实验使用随机key/value关联及同key更新，但这不是冻结LLM生成新三词组合的直接证据。

对本实验的启示是：可以保留每个事实块，把内容转换成小型动态算子再作用于层输入；同时必须承认它仍有有限rank、干扰和writer/readout匹配问题。我们不把整库累积成一个全局fast-weight状态，而是保持VDB每条事实可独立写改，再稀疏选择一块。因此它支持数学设计，不直接证明我们的写入局部性或泛化。

## 2. DeltaNet：有效的写入规则比“有矩阵”更具体

**Yang et al.，Parallelizing Linear Transformers with the Delta Rule over Sequence Length，2024。** [原文 §2.2](https://arxiv.org/html/2406.06484v3#S2.SS2)，[方法 §3](https://arxiv.org/html/2406.06484v3#S3)，[官方实现](https://github.com/fla-org/flash-linear-attention)。

DeltaNet的核心更新可写为`S_t=S_(t−1)+β_t(v_t−S_(t−1)k_t)k_tᵀ`：用已有矩阵读出同key的旧值，再写入误差。论文还解决该递推的并行训练，包含关联召回任务与大规模语言建模；其结果不能外推为“把一个已训练模型的VeRA向量改成矩阵即可泛化”。

我们的VDB已经精确替换目标记录，无须为单条upsert再引入delta规则；本轮也**不是DeltaNet复现**。若以后把大量事实压缩进同一个固定矩阵，才需要进一步比较误差校正写入、遗忘、碰撞和顺序敏感性。本轮先比较低秩读出，避免同时更换写入规则和物理存储权限。

## 3. LongMem：块召回保留token级结构，但最终仍做attention

**Wang et al.，Augmenting Language Models with Long-Term Memory，2023。** [原文 §2.3](https://arxiv.org/html/2306.07174#S2.SS3)，[训练设置 §3.1](https://arxiv.org/html/2306.07174#S3.SS1)，[论文提供的代码入口](https://aka.ms/LongMem)。

LongMem先对chunk keys作均值检索，召回完整chunk K/V，再展开为token级K/V做softmax attention；其memory输出最终仍是每query一个加权向量，并与local attention门控融合。配置为冻结407M骨干和可训练12层SideNet，训练26B tokens，每token召回16个四token块，合计64对K/V。

它直接支持“索引粒度和返回内容粒度可以不同”：稀疏召回完整块可避免检索时丢弃邻近信息。但它没有把整块V作为VeRA的动态低秩参数矩阵；其训练规模和SideNet容量也远大于我们的单层受控试验。我们借用完整块读取的结构直觉，不把它的成功视为本实验的预期答案。

## 本轮可以声称什么

若block_outer在相同regime、seed、数据、目标曝光、共享参数数量、K/V字节和对角路径下，稳定优于pooled_outer，并且真实内容更新优于shuffle/empty，才支持“逐槽构造算子比先池化更适合当前任务”。其成本必须按实际算子计算量报告；相同参数与存储并不等于相同FLOPs。

若仅对diagonal改善，额外参数与更复杂因子本身仍是解释；若新组合仍失败，不能推出所有fast weights或attention都无效。高事实召回而答案错误也不能单独锁定value编码，因为后续decode可能漂移，错误前缀与路由变化可能互为因果。三词已定位span、有限词表、单词替换组合和有限优化预算仍限制外推；本轮没有自动自由长文事实发现、跨域持续学习或首创性结论。
