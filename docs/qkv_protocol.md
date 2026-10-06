# 随机绑定与分组 QKV 读取：固定实验协议

**版本 V1，正式训练前封存。** 2026-10-06 教师预检 128/128、四组真实模型 smoke 与 1,085 项 CPU 测试通过；主 agent 在注册文件登记源码、数据与本文 SHA。此前没有本轮正式效果；封存后不按确认结果换题、换模板、调整预算或移动门槛。文献依据见 [QKV 文献映射](qkv_literature.md)，此前证据见[小数据重构结果](reconstruction_results.md)。

本轮检验：**训练每个实体绑定多种内容，并改变稀疏记忆的读取结构，能否让当前层输入构造的 query 读取新写入的内容，而不是只重现训练完整标签？** 主线仍为真实 VDB→value→乘性 VeRA。加性 attention 风格读出单列为诊断，其成功不能充当 VeRA 成功，也不能证明所有更强 reader 均有效。

## 一、固定矩阵与共同接口

| 臂 | 训练绑定 | 读取 | 增量形式 | 可作主线候选 |
| --- | --- | --- | --- | --- |
| static_flat | 实体固定 A/B | flat top-4 slots | 乘性 VeRA | 是 |
| static_grouped | 实体固定 A/B | top-1 fact，再读该组三槽 | 乘性 VeRA | 是 |
| rebind_flat | 每 epoch 随机重绑定 | flat top-4 slots | 乘性 VeRA | 是 |
| rebind_grouped | 每 epoch 随机重绑定 | top-1 fact，再读该组三槽 | 乘性 VeRA | 是 |
| rebind_grouped_additive | 同 rebind_grouped | 同 grouped | 加性向量读出 | **否，仅诊断** |

五臂均固定 Qwen3-4B-Instruct-2507 revision `cdbee75f17c01a7cc42f958dc650907174af0554`，第 20 层 down projection 输入 9,728、输出 2,560，rank/key dimension 均 64，temperature `0.05`。骨干与 A 固定，B 可学习；每事实三个真实 contextual word-span 向量，Wq/Wk/Wv 为共享线性投影，不加多头、不扩大 rank、不用基础库、不新增 value MLP。每臂还具有相同的可学习 `slot_position[3,64]`，零初始化，添加到 key 的线性投影之后、最终 L2 归一化之前；不改变 value，query 不接收正确 slot 标签。

同种子所有初始 A/B/Wq/Wk/Wv/b 与 slot_position 张量逐位相同；仅 regime/read_mode/readout 不同。共同统计由训练 canonical static A/B 的 128 条支持记录和 64 条 query 拟合，不用 dev/confirm、C/D、heldout 模板或整个 entity×payload 笛卡尔积拟合。五臂统计相同。word slot 的顺序来自实际观测文本，不能由 gold 下一词、目标数组索引或人工正确 slot 指定检索。

### 明确读取公式与梯度

对每个实际预测位置，`q_t=encode_query(h_t)`，其中 `h_t` 是当前真实 VeRA 层输入。标准化后的三个 slot keys 为 `k_{i,j}`，cosine 为 `s_{t,i,j}=q_t·k_{i,j}`，`j∈{0,1,2}`。

- **flat：**在整个 16-fact×3-slot 银行中取 cosine 最大的四个 slots，在这四个分数除以 temperature 后做 softmax，并混合对应 values。允许跨 fact 混合。
- **grouped：**对每个 fact 计算 `g_{t,i}=logsumexp_j(s_{t,i,j}/0.05)`，取 `i*=argmax_i g_{t,i}`；再只在该 fact 的三个 slot 上按相同 temperature 做 softmax。返回三个 value 的混合，不根据 gold token 选择词槽，也不把所有 fact 的每个槽均值折叠为一条值。

grouped 的 top-1 fact 选择不可微；生成 CE 不会通过这个离散选择向未选 fact 提供选择梯度，CE 只训练被选三槽的分数和值。所有臂另用相同的 **dense fact-group 地址 CE**，给所有银行 keys/query 提供地址监督；必须明确区分该训练辅助目标与实际稀疏读取。group score 是三槽 logsumexp，不是独立学习的实体地址 encoder，因此本轮不能宣称已经实现地址与内容完全分离。

记忆混合值记为 `m_t`。乘性主线使用 `Δh_t=b⊙B[(A h_t)⊙m_t]`；加性诊断使用 `Δh_t=b⊙B m_t`。加性臂保留相同初始化的 A buffer，但不使用 A 的乘性路径；全部其他 trainable 张量和监督相同。在第 1 步及每 128 步，用实际捕获的 gold-prefix 预测位置（含 EOS）于参数更新前进行无梯度读取重放，记录两路径输入、混合值及残差 RMS；不增加 backbone forward 或训练 loss。这是训练位置诊断，不能代替自由生成的读取指标，也不能把效果差异直接归于单一机制而忽略幅度/优化差别。

五臂均有 2,034,368 个可训练参数（含 192 个 slot-position 参数），每事实纯 FP32 K/V 为 `3×(64+64)×4=1,536 bytes`，16 条银行为 24 KiB。flat 混合 4 个 value，grouped 混合 3 个 value；虽然存储与参数配平，实际读取值数量不同，不能称同 FLOPs。CPU exact search 扫描所有 keys，稀疏的是 value 选择/混合，不宣称 ANN 或次线性搜索复杂度。

## 二、数据、随机绑定与特征合同

数据协议为 `qkv-binding-episodes-v1`，新数据种子 `121042`。训练有 64 个实体、128 个独特三词 payload。实体和完整 payload 排除上轮 reconstruction 数据种子 `101042` 的全部 train/known/dev/confirm A/B/C/D；词汇表和历史模板可以复用，不宣称从未见过的词汇或研究模板。每个位置的 16 个词在 128 个训练 payload 中各出现 8 次，不能靠位置词频推断新组合。

static 中实体 `e` 固定绑定 payload `2e`、`2e+1`。rebind 在每个 epoch 对全部 128 个 payload 作确定性随机排列，再把排列的 `2e/2e+1` 分配为该实体的 A/B。epoch 包含 8 步；目标顺序是 64 个实体的一个随机排列，每步取 8 个，因此一个 epoch 中每实体恰为目标一次，128 个 payload 各作为 A/B 监督目标出现一次。

每步银行由当步 8 个目标实体和从其他 56 实体中采样的 8 个背景实体组成，再随机打乱 16 条的物理顺序。相同种子五臂的目标、背景实体和银行顺序完全一致，使用不依赖 regime 的随机流。映射变化会改变背景的实际内容，**目标 payload 曝光相同不保证每个 payload 的背景驻留次数相同**；应记录两者，不将整库内容访问数当成独立监督样本数。

对每个目标，各建一个 A 银行和一个 B 银行：A 银行全部记录使用当前 epoch 的 A 绑定，B 只替换该样本自己的目标记录为当前 B，其他 15 条保持 A。不得把整批八个目标同时换 B，也不得跨样本检索。A 八条、B 八条按固定顺序合并成 16 序列的一次 backbone forward，各序列保有独立银行。目标事实索引仅供训练地址 loss，不输入 query 或 CPU 搜索。

**训练支持缓存必须对真实的 64×128=8,192 个 entity+payload 组合文本重新冻结编码。** 特征提取输入沿用准确 writer 包装 `Remember this information: ` 加实际支持文本，记忆路径关闭；以 tokenizer 对齐的三个 word spans 分别池化当前层输入。不得把旧实体的 contextual hidden 直接换给新实体而声称重编码。Wk/Wv 和 slot_position 训练期间按当前参数重编码所需银行，避免投影后缓存陈旧。原始冻结特征可缓存，但不跨越所声明的文本合同。

### 固定评估划分

三个评估 bank 均为 16 条；训练每个样本的 bank 也为 16 条：

| 数据包 | 实体 | A/B 内容 | C/D 内容 | 用途 |
| --- | --- | --- | --- | --- |
| known16 | 训练前 16 个实体 | 其 canonical static 的训练 payload | 新完整组合 | 识别已见绑定/新内容写入 |
| dev16 | 16 个新实体 | 与 known 对应行相同 | 与 known 对应行相同 | 开发阶段实体迁移/候选选择 |
| confirm16 | 另 16 个新实体 | 与 known 对应行相同 | 与 known 对应行相同 | 最终实体迁移验证 |

known/dev/confirm 按行刻意共享内容，不能宣称评估答案在三个划分间互不重叠。新实体 A/B 的完整内容已训练，是用来在相同 bank 大小、相同 payload 下隔离实体变化。C/D 完整组合不在本轮训练或上轮数据中，且彼此不同；行 `i` 仅改变 A 的第 `i%3` 个词，其余两个词保持不变，16 行改动位置数为 6/5/5。C 与 D 均是已见词组成的新完整组合，不能称未见词汇泛化；D 也沿用同一受控单词替换规则，并非独立任务分布。

canonical support/query 在各 split 相同，heldout support/query 只相对于**本轮训练**未见；它们是本研究历史已经使用的模板，不当作全研究封存的新表达。CC/HC/CH/HH 的第一字母指 support，第二字母指 query。全部冻结特征（含 D/confirm）允许评分前预计算，但 D 和 confirm 的生成/评分在候选封存后才执行；不进入梯度、统计拟合或候选选择。

## 三、训练预算与监督

三个优化种子为 `81042 / 81043 / 81044`。五臂各三个种子，共 **15 个模型**，全部从共同对应初值独立训练，不能将某臂额外训练后与较少步数对照。

每模型固定 2,048 更新，batch 8 个目标、A/B 合计 16 序列，唯一主要 checkpoint 为终点。每模型 256 epochs、16,384 目标曝光、32,768 个学生序列、2,048 次合批 backbone forward；每个实体 256 次目标曝光，每个训练 payload 256 次监督序列曝光。15 模型合计 30,720 更新、245,760 目标曝光、491,520 序列、30,720 次训练 backbone forward。token 数、padding、实际输入位置和时间据实记录，不能用序列数冒充等 token/FLOPs。

损失只有 `完整答案含 EOS 的 CE + 0.2×fact_group_address_CE`。两项都先对每个序列有效 gold 预测位置（含 EOS）取平均，再对 16 序列等权；A/B 各八条，世界权重相同。地址目标是目标三个槽概率之和，不把每个答案 token 指派给某个词槽。无 KD/FKL、hidden、OPD、style/AP、replay、load-balancing 或其他新 loss。teacher forcing 指 gold 前缀，不是教师模型蒸馏。

Adam：Wq/Wk/slot_position `1e-4`，Wv `3e-4`，B `3e-4`，b `0.005`；betas `(0.9,0.999)`、epsilon `1e-8`、weight decay `0`，各参数组 clip norm 1。slot_position 与 Wq/Wk 同组，其余组沿用共同规则。b 零初始化使全部路径初始增量为零；所有臂不得借此添加未声明暖启动。只记录中途训练诊断，不用 C/D 选择早停或最好 checkpoint。

上轮每训练实体 512 次目标曝光，本轮是 256 次，且训练实体/内容量、slot-position 与训练日程变化；不能把跨轮差异视作仅 QKV 改动的因果效果。主要效应只在本轮配平矩阵内比较。若已见重构仍未稳定，仍完成预定小矩阵，如实记录，不追加预算追赶门槛。

## 四、教师资格与在线评估

教师是冻结原模型，获得当前目标的单条原始支持文本和相同问题；不访问 VDB、不追加 gold 标签或答案定位标注，不参与任何训练 loss。首先对 canonical static 64 个训练实体的 A/B 运行预检，要求各 64/64、pair 64/64。失败则停止正式训练，修复后另版封存，不删掉教师失败样本。

开发/确认教师覆盖相同的非 SWAP 世界及相位，保留原文、预测、token 统计和无记忆访问证明。教师不合格条件仍保留学生全样本分母，并列 teacher-paired eligible 子集；不能筛题制造高分。最终确认教师不足不能自动转化为 VeRA 泛化失败或事后修改 prompt。

学生共享参数全冻结，只把实际观测支持的 writer 输出写入真实 CPU VDB，学生 prompt 始终仅有问题。每 phase 先生成 16 条 A，之后每个 target、每个世界从独立干净 A 银行出发，执行 A→X→A；所有其他 key/value/timestamp 逐位不变，目标三槽原子替换，恢复后的 key/value 回到 A，但时间戳可递增。

每次干预生成：目标新 X、固定邻居 `(i+1)%16` 的 A、恢复后的目标 A。邻居 before 使用同 ID/phase 的基线 A。除 SWAP 外，另生成 value-shuffle 和 empty 的目标 X。SWAP 固定将另一实体 `i^1` 的 A 内容写入目标，donor 不变；它不是真正交换两个记录，也不是未见组合。所有对照保持其声明的 keys/模板/解码参数。

shuffle 以整个三槽 fact 为单位，保留组内次序与全部 keys，使用固定无不动点 cycle；控制随机种子与 permutation 记入数据/评估 manifest，不能按预测选择。empty 移除全部记忆，基础库确实关闭。每次调用均记录，跨世界若未来实施缓存必须明确另版成本，不能伪造多个独立生成；本轮预算按下表逐条生成计算。

### 固定评估矩阵与调用数

每次 generation 最大 32 新 tokens，固定解码设置；学生与教师相同。

| packet / 阶段 | 相位与干预 | 每模型学生生成 | 独立干预 | 恢复写入 | 教师生成（全模型共享一次） |
| --- | --- | ---: | ---: | ---: | ---: |
| known / development | CC：B、C、SWAP | 224 | 48 | 48 | A/B/C：48 |
| dev / development | CC：B、C | 176 | 32 | 32 | A/B/C：48 |
| known / confirmation | CC：D | 96 | 16 | 16 | A/D：32 |
| confirm / confirmation | CC/HC/CH/HH，各 B、D | 704 | 128 | 128 | 四相位 A/B/D：192 |
| 每模型学生合计 |  | **1,200** | **224** | **224** | — |

公式：每 phase 基线 A 为 16 次；每非 SWAP 世界为 `16×(updated+neighbor+restored+shuffle+empty)=80` 次；SWAP 为 48 次。全部 15 模型 **18,000 次学生生成、3,360 次独立干预、3,360 次恢复，共 6,720 次 real group 写入**。教师预检 128 次，加开发 96 与确认 224，共 **448 次教师生成**；总正式生成 18,448 次。初始化填库和控制银行物化不计在线 logical writes，成本单列；smoke 另列。

## 五、主要指标、候选封存和停止条件

主要指标是规范化完整答案 EM；normalization 沿现有实现（NFKC、大小写、标点、空白），另保存原始文本。已见 A/B pair 要求起始 A 与写入 B 都正确。C/D `update_restore` 要求写入新内容和回写 A 两个时点均正确；起始 A+新内容+回写 A 的三时点联合率另外报告，不替换门槛。`locality_joint` 要求邻居 before/after 均正确，文本相同但两次都错不通过。

对相同问题的不同世界，空库确定性答案无法同时匹配两个不同 payload，所以 empty 的严格 paired EM 必为零，不把这一逻辑零当作记忆成功证据。使用同世界单边 real−shuffle、real−empty 的 EM 差。

### 开发门槛（四个 VeRA 主线臂，三个种子逐个满足）

| 检查 | 固定整数门槛 |
| --- | --- |
| known CC 已见 A/B pair | ≥12/16 |
| known CC C update+restore | ≥12/16 |
| known CC C 单边 real−shuffle、real−empty | 两项各 ≥0.50 |
| known CC C 邻居联合正确 | ≥15/16（对应 ≥90%） |
| dev 新实体 CC C 单边完整 EM | ≥8/16 |

门槛是本轮用于决定是否扩大实验的工程标准，不是统计显著性保证，也不声称达到部署质量。五臂全部完成同预算，即使早期 baseline 不通过，也不据此取消预定对照。

在满足全部门槛的 VeRA 臂中，按以下顺序确定唯一架构：最大化三个种子中最小的 known C update+restore；再最大化最小的 dev 新实体 C 单边 EM；再少可训练参数、少每事实向量字节；仍相同按 `static_flat → static_grouped → rebind_flat → rebind_grouped`。只看开发终点，不择种子；additive 永不参与排序。没有合格臂则记录“无候选”。

全部 15 个训练和 30 个开发学生评估完成、教师资格已保存后，封存候选与证据 SHA；任何确认 child 的启动在此之后。所有模型的 D/confirm 可完整报告，但不得事后选另一个确认表现较好者。

### 最终扩样门槛（仅开发选定架构，三个种子逐个满足）

- known CC D update+restore ≥12/16。
- known CC D 单边 real−shuffle、real−empty 各 ≥0.50。
- known CC D 邻居联合正确 ≥15/16。
- 新实体 confirm CC D 单边完整 EM ≥8/16。

新实体 D 的 update+restore、controls、locality 与四相位 A/B pair 均完整报告，但不能临时加成或去掉主门槛。HC/CH/HH 用于独立表达变化描述，不用确认新格式挑模型。无候选、任何所选种子未达标或对应教师不合格时，不自动扩大至 256/1024 条；教师不合格时标记结论受限，不用学生零分判断能力上限。additive 达标也不能触发主线扩样。

## 六、路由、内容和成本诊断

每个生成 token 保存 token ID、原始文本对应关系、真实 query 路由、selected fact、三个词槽的权重总量、分数和银行 hash。严格区分 prefill 最后位置的首次预测与后续 decode；不能把问题首部的 trace 当首个答案 token。按 tokenizer 解码与累计字节/词边界，把 slot 访问对齐到**模型实际生成**的词位置，生成与 gold 分词不同时保留不确定性，不能用 gold 强行指派正确 slot。

grouped 返回的 index 顺序固定为 slot0/1/2，不按分数排序。因此 R@1 必须由最高分/权重的真实槽或明确的 selected fact 定义，不能读取列表第一项冒充最高排名。主要路由报告为目标 fact 是否被实际读取、grouped selected-fact 准确率、各 slot mass、相关词预测时刻的命中，以及 decode 驻留；flat 实际读四槽、grouped 实际读三槽，不能误写成相同 R@4 权限。

完整答案包含率、词数、budget_hit、改动词位置正确率、输出旧 A/B/任意训练 payload 比例是预先指定的诊断，不用于筛选样本。NLL 作为 teacher-forced 辅助指标，评价是否包含 EOS、归约方式必须说明；不与包含 EOS 的训练 CE 混作同一口径。首 token 或 fact 命中不代表内容成功。

训练记录每 entity/payload 的目标曝光、背景驻留、逐步 mapping/目标/背景/银行顺序及 digest；记录参数初值、训练末状态、实际梯度、每组 LR/clip、token/输入位置、backbone 调用和 elapsed。评估保存银行初值、逐目标写入/恢复、控制 permutation、参数前后 hash、K/V 快照与 CPU 路由。多进程累计耗时不能宣称独占 GPU 延迟；实际 values 读取和 key 全扫描成本分开。

## 七、结论边界与封存前审查

主矩阵可以区分本预算下固定绑定与 rebind、flat 与 grouped 的效果及其交互，不能将 grouped 的效果称为实现了完全分离的地址/内容编码。它仍是单 head、三槽、单层 rank64；无改善不否定所有 QKV 或 attention 记忆。加性诊断改变注入形式，既不属于 VeRA 主结论，也不支持直接宣称乘性是唯一失败原因。

三个种子共用同一微型数据、词表和模板，变化的是初始化和训练日程；不是三个独立语料。C/D 是按行单词替换、内容在 known/new 间有意配对，不能泛称任意组合、任意语义或跨域持续学习。组内 attention 即使取回所有三个槽，也可能赋予新词极小权重，反之曾命中该槽也不保证有效读出。

封存前必须核对：全部历史排除与配对数据；8,192 个真实训练文本编码的 writer prefix/span/关闭记忆合同；共同 train-only 统计；同 seed 五臂初始化相同；每 epoch 目标实体及 payload 精确配平；单样本 A/B 只改自身；CPU/tensor 读取及空库等价；group选择梯度和 dense 地址辅助可用；真实模型预检128教师/合批smoke；生成 token 与 route 对齐；45个训练+开发条件选择屏障；本文、源码、计划与数据 SHA。任何实质修复须重新版本化并保留原产物，不把实现失效记成模型失败。
