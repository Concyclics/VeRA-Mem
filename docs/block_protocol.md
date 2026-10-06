# 稀疏整块记忆与动态低秩算子：实验协议

**V1 预注册协议（正式训练前封存），2026-10-06。** 本文先规定公式、数据、预算、指标和继续研究门槛，再由主 agent 在正式训练前记录源码、数据、计划与本文 SHA。正式结果尚未产生；封存后不按 C/D 分数更换公式、模板、训练步数或门槛。文献与等价性说明见 [block_literature.md](block_literature.md)，此前失败证据见 [qkv_results.md](qkv_results.md)。

本轮问题是：**稀疏召回完整的三槽 value 矩阵后，让各槽分别生成动态低秩算子，是否比先池化成一个向量再生成算子更能读取未训练的完整内容组合？** 单纯把共享线性投影移到求和之前数学等价，不作为新模型。本轮保留原对角 VeRA 路径，并在相同实际分组检索上比较两种额外算子。

## 1. 固定矩阵与公式

| 训练绑定 | diagonal | pooled_outer | block_outer |
| --- | --- | --- | --- |
| static：每实体固定 A/B | static_diagonal | static_pooled_outer | static_block_outer |
| rebind：每 epoch 重新绑定 | rebind_diagonal | rebind_pooled_outer | rebind_block_outer |

六臂各用三个种子 `91042 / 91043 / 91044`，共 **18 个模型**。同 seed 的共同张量逐位同初始化；两个 outer 臂的新张量亦逐位相同。模型固定为 Qwen3-4B-Instruct-2507，revision `cdbee75f17c01a7cc42f958dc650907174af0554`，第 20 层 down projection 的输入/输出维数为 9,728/2,560；latent rank 与 key dimension 均为 64。骨干与 A 冻结，B 可学习；基础记忆库关闭，三个 contextual word slots、共享 Wq/Wk/Wv 和零初始化 slot_position 沿用已验证接口。

每个实际生成或 gold 预测位置，从该位置的真实层输入 x 构造 query。每个事实的三个 L2-normalized keys 与 query 做 cosine；fact 分数为 `logsumexp(slot_cosine / 0.05)`，实际 CPU VDB 取 top-1 fact，返回其完整 `V ∈ R^(3×64)`。**全部六臂均在选中块内使用固定 1/3 权重**，不再使用上轮的组内 softmax。因此本轮 diagonal 也必须重新训练，不能将上轮 grouped checkpoint 当成配平基线。

设 `z=A x`，`m=(v_0+v_1+v_2)/3`。定义共享可学习函数：

```
u(v) = (P_in v + c) / max(||P_in v + c||₂, 1e-6)
w(v) = P_out v

M_diagonal(V) = diag(m)
M_pooled(V)   = diag(m) + w(m) u(m)ᵀ
M_block(V)    = diag(m) + (1/3) Σ_s w(v_s) u(v_s)ᵀ

Δh = b ⊙ B [M(V) z]
```

`P_in/P_out` 都是 64×64，共同 identity 初始化。c 是 64 维可学习 bias；使用独立 CPU 随机流 `seed+104729` 采样标准正态后除以自身 RMS，初始 RMS 为 1，不消耗共同参数的全局初始化随机流。c 非零是声明的架构组成，不能在结果后取消或改变。b 零初始化；没有额外可学习 gate、手工 outer 倍率或事后幅度重标定。

两个 outer 臂各增加 `2×64²+64=8,256` 个参数，分别总计 2,042,624；diagonal 为 2,034,368。核心同参数对照为 **同 regime/seed 的 block_outer 对 pooled_outer**；对 diagonal 的变化可能同时来自额外参数和算子结构，必须另列。pool 外积项 rank≤1，block 外积项 rank≤3；共同对角项使总 M 仍可能 rank64，不能把整个模型称为 rank1/rank3，也不能把外积 rank 的增加等同于有效信息容量提高。

每事实保存的纯 FP32 K/V 仍为 `3×(64+64)×4=1,536 bytes`，16 条为 24 KiB；不在 VDB 保存一个额外的 64×64 矩阵。P_in/P_out/c 属于共享参数，读取时构造因子并以两次矩阵向量乘法应用算子。block 与 pooled 的参数、K/V 字节相同，实际算子计算量不同，不能称等 FLOPs。CPU exact search 扫描全库 keys，稀疏的是选中 value 块，不宣称 ANN 或次线性检索。

top-1 fact 的离散选择不可微；块内 uniform 也不给生成 CE 提供 slot 分数梯度。所有臂都有相同 dense fact-group 地址 CE，向 query/keys 提供地址监督。地址仍来自内容相关的 contextual keys，不宣称已经完全分离地址与内容。即使完整块被取回，各槽通过 `u(v_s)ᵀz` 产生的贡献仍可能小、带负号或互相抵消；uniform 检索权重不代表三个槽的有效输出贡献相同。

## 2. 新数据、真实特征与训练曝光

数据种子固定为 `221042`，数据协议 `block-outer-binding-data-v1`；完整实体和 payload 排除 reconstruction `101042` 与 QKV `121042` 的全部相关 A/B/C/D/SWAP 数据。只读取历史数据规则，不按历史模型分数选新题。词汇与模板继续复用，是受控机制实验，不称未见词汇或全研究未见格式。

训练为 64 个实体、128 个独特三词 payload，各位置 16 个词分别出现 8 次。static 令实体 e 固定绑定 payload `2e/2e+1`；rebind 每 epoch 对 128 个 payload 重新排列后分配 A/B。每 epoch 8 步，每步 8 个目标，覆盖全部 64 个实体一次、128 个目标 payload 一次。相同 seed 六臂的目标、背景与银行顺序完全一致；regime 改变实际绑定，不改变目标 payload 总曝光。背景驻留单独统计，不假定逐 payload 背景访问也完全一致。

每步银行由 8 个目标和 8 个其他实体组成，物理顺序随机。对每个目标，A 银行为当 epoch 的全部 A 绑定，B 只替换本样本自己的目标记录，其他 15 条不变；八个目标各自的 A/B 合批为 16 序列，不能共用一个同时改八条的 B 银行，也不能跨样本检索。正确 fact 索引仅用于地址 loss，不输入 query。

必须对全部 `64×128=8,192` 个真实 entity+payload 文本冻结重编码，不能交换来自旧实体的 contextual hidden。writer 精确包装为 `Remember this information: ` 加支持文本；在记忆关闭时提取已定位的三个词 span 特征。这个 writer 使用支持文本中已标注/可定位的三词范围，**不是自由长文自动发现事实边界的系统**。训练投影后的 keys/values 随参数重算。共同统计只使用 canonical static A/B 的 128 条支持记录与 64 条 query，不用完整笛卡尔积、C/D、dev/confirm 或 heldout 模板拟合。

known/dev/confirm 各 16 条，所有训练样本和评估 bank 均为 16 facts。known 使用前 16 个训练实体；dev 和 confirm 各用不同新实体。三个 packet 按行共享 A/B/C/D payload，以隔离实体变化。A/B 是训练出现的完整 payload，但在 rebind 下，某个固定 entity–payload 配对未必作为目标被监督过，必须统计实际绑定曝光，不能将所有 known A/B 简写成“监督已见绑定”。

C/D 是彼此不同、完整组合未参与训练的新 payload；第 i 行只替换 A 的 `i%3` 个词，另两词保持，位置分布 6/5/5。组件词均已训练。CC/HC/CH/HH 的第一字母指 support、第二字母指 query；heldout 只相对本轮训练，是历史使用过的模板。D/confirm 特征可提前冻结编码，但其生成/评分不参与梯度、统计、早停或模型选择。

## 3. 固定训练与成本

每模型固定 **2,048 更新、batch8 targets、每步16个 A/B 序列**；仅终点 checkpoint 进入主要评价，不用 C 或 D 挑中间 checkpoint。每模型 256 epochs、16,384 targets、32,768 序列、2,048 次合批 backbone forward；每实体 256 次目标曝光，每 payload 256 次监督序列曝光。18 模型合计 **36,864 更新、294,912 targets、589,824 序列和36,864次训练 backbone forward**。

loss 仅为 `完整答案含 EOS 的 CE + 0.2×dense fact-group address CE`。两项都先对每序列的有效预测位置取平均，再对16序列等权，A/B权重相等。无 KD、hidden、OPD、额外重构、正交、熵或负载均衡 loss。teacher forcing 指真实答案前缀，不是教师模型蒸馏。

Adam 的 LR：Wq/Wk/slot_position `1e-4`，Wv、B、P_in/P_out/c `3e-4`，b `0.005`；betas `(0.9,0.999)`、epsilon `1e-8`、weight decay0，各参数组 clip norm1。两个 outer 臂的附加参数作为相同独立 optimizer group。没有额外 warmup、续训、补足到同训练 loss 或结果后调参。

保留每步训练 CE/地址 loss、各组梯度、映射和 target/background 曝光、token与padding、实际输入位置、backbone调用及耗时。第1步与每128步可在捕获的实际 gold-prefix 位置做无梯度读取诊断，记录残差/输入尺度，不增加 LM forward 或训练目标。预算相同不保证优化程度相同；若 rebind 未收敛，必须与结构结论并列，不能把失败宣称为容量不可能。各臂真实 token、计算量和并行耗时据实报告，过程秒数不冒充独占 GPU 延迟。

## 4. 教师资格、真实在线改写与固定评估

教师是冻结原模型，获得当前目标的单条原始支持文本和相同问题，不访问记忆，不额外追加 gold 标注，不参与训练。正式训练前 canonical64 A/B 预检要求128/128。开发/确认覆盖全部对应非 SWAP 条件；保留全样本分母与 teacher 合格子集，不能删除失败题或依据确认结果改变 prompt。

学生共享参数全冻结，prompt只有问题，使用真实 CPU VDB。每 phase 先生成16个初始 A；随后每个目标和世界独立从干净 A bank 出发，执行 `A→X→A`。每次写入仅改变目标三槽，所有非目标 keys/values/timestamps严格不变。回写后目标 K/V回到A，时间戳可递增。每次生成目标X、固定邻居`(i+1)%16`的A、恢复后的目标A；邻居before使用同phase初始A。

非SWAP世界另运行 value-shuffle 和 empty：shuffle保持全部keys，以固定无不动点fact cycle搬移完整三个values，保留组内顺序；empty确实关闭所有记忆路径，输出增量严格为0。SWAP只把 donor `i^1` 的A内容写给目标，donor不改，不是同时互换两条记录。解码固定 greedy、最多32新tokens，保存文本/tokenIDs/预算截断。

| packet / 阶段 | 相位与干预 | 每模型生成 | 独立干预 | 恢复 | 教师一次共享生成 |
| --- | --- | ---: | ---: | ---: | ---: |
| known / development | CC：B/C/SWAP | 224 | 48 | 48 | A/B/C 48 |
| dev / development | CC：B/C | 176 | 32 | 32 | A/B/C 48 |
| known / confirmation | CC：D | 96 | 16 | 16 | A/D 32 |
| confirm / confirmation | CC/HC/CH/HH 各B/D | 704 | 128 | 128 | 各A/B/D共192 |
| 每模型 |  | **1,200** | **224** | **224** | — |

18模型共 **21,600次学生生成、4,032次独立干预、4,032次恢复，共8,064次real group写入**。教师预检128+开发96+确认224，共448次；正式总生成22,048次。18train+72student eval+5teacher+1prepare共96个正式child；smoke另列。初始填库与控制银行物化不混入online logical writes。全部预定架构和seed终点评估完成，不由开发结果筛除失败臂；本轮没有事后择种子或赢家替换。

## 5. 指标与继续研究门槛

完整答案规范化 EM 是主要内容指标，原始文本另存。known C/D 的 `update_restore` 要求新内容生成正确且回写A后正确；同时报告初始A、新内容、恢复A三时点均正确。新增实体 confirm D 的CC及HC/CH/HH固定完整报告，不能只选最好format或只报R@read。A/B pair要求初始A和更新B均正确；locality_joint要求邻居before/after均正确，错→错文本不变不通过。

主结构效应是同regime/seed的 `block_outer − pooled_outer`；diagonal是额外参照。C仅作已规定开发描述，D用于最终确认；所有架构与种子都报告，不能根据C追加预算再以D声称同预算。空库对于不同答案的确定性paired EM存在逻辑零，controls使用**同一世界单边EM差**。

每个regime的block_outer独立检查以下固定整数标准，**三个seed各自全部满足**才构成继续投入该块读出结构的证据：

| 条件 | 固定门槛 |
| --- | --- |
| known CC C update_restore相对同seed pooled_outer | 至少增加4/16 |
| known CC D update_restore相对同seed pooled_outer | 至少增加4/16 |
| block_outer known CC A/B pair | 至少12/16 |
| block_outer known CC C单边real−shuffle与real−empty | 两项各至少4/16 |
| block_outer known CC D单边real−shuffle与real−empty | 两项各至少4/16 |
| 对应教师资格 | 相关条件教师全部正确；不合格则标结论受限 |

这不是统计显著性保证，不以某seed正确1–2条宣称解决泛化；也不是扩大语料的自动授权。本轮结束先完整报告是否达到标准、跨实体/格式结果与局部性，再决定下一轮。没有预设新实体数值门槛，不能在看分数后添加最有利的门槛；新增实体D差或locality差，即使上表通过，也必须明确限制“已解决泛化”的表述。

辅助指标固定为逐词正确、改动词正确且另两词保留、完整答案包含率、输出旧A/B/任意训练payload比例、词数、budget_hit、NLL、首预测与decode事实召回。严格三词诊断与“存在该词位置”的诊断分开；NLL是否含EOS及归约据实现说明，不与含EOS训练CE直接跨口径比较。三seed共用数据，16行/多phase/多模型结果不是独立语料样本。

## 6. 审计与解释边界

保存每个生成token的实际query、选组indices/scores/weights与银行hash，按真正首次预测及decode对齐；selected values由保存的编码银行、写入事件和indices逐位重建，不在每条JSON trace重复存储。CPU审计若拥有完整query，可以重放全库group选择；只有已选分数则只能核权重/索引，不能声称证明全库top-1。uniform三槽均被读时，slot mass只是取回权重，不能作为有效内容贡献的因果解释。保存银行初值与写回事件，核参数前后hash、控制置换和非目标不变。

两outer臂的比较同时改变**池化前后生成非线性归一化因子的顺序**及外积项最大rank，不能只凭结果归因于“保留矩阵”或某一个rank数值。它们都是动态参数算子，最终仍产生一个latent向量；向量输出本身不是信息不足的证据。共享B后移的等价改写不作为新表达能力。即使本轮失败，也只约束这个单层、三槽、线性因子、rank64、固定预算的小任务，不否定多头、深层reader或所有attention memory。

封存前校验：历史排除/真实span编码、8192文本与train-only统计、同seed共同初值及两个outer初值、逐epoch曝光、单样本银行隔离、tensor/CPU公式等价、共同对角项、空库零增量、outer有效梯度、同token路由、教师预检和真实模型smoke；随后记录源码/数据/计划/本文SHA。实质bug必须版本化修复并保留原产物，不能把无效运行算作负面实验结论。
