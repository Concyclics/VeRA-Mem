# 随机绑定与分组 QKV：可写向量记忆实验

**状态：全部正式训练、开发、确认与教师结果已完成并本地严格复算。** 十五个训练、六十个学生评估和五个教师作业全部齐全；模型条件75/75，无pending或重复，教师448/448完整正确。开发候选在任何确认启动前封存为“无合格候选”，冻结协议、模型和选择规则未改变。

**结果：本轮固定预算下，随机重绑定和分组QKV未建立完整新组合的在线重构能力。** 所有十五个模型在known/dev新组合C、known新组合D和新实体confirm四相位D上的完整EM均为0/16。加性诊断同样失败，所以不能把结果单独归咎于VeRA乘性路径。随机重绑定对新实体上的已训练payload A/B和部分被编辑词有收益，但种子差异明显，也没有保住完整三词组合。开发候选于 `2026-10-06T13:50:07.472005+00:00` 封存为 `selected_arm=null`。最终确认没有改变这一结论，按固定规则不扩大至256/1024条。局部词改善、较低NLL或单次命中目标组均未代替完整可写性。

证据入口：[固定协议](qkv_protocol.md)、[文献映射](qkv_literature.md)、[完整汇总](results/qkv/summary.json)、[关键数字](results/qkv/key_facts.json)、[封存选择](results/qkv/selection.json)、[注册记录](results/qkv/registration.json)。[完整诊断](results/qkv/answer_diagnostics.json)、[教学案例](results/qkv/examples.json)、[训练末段指标](results/qkv/training_convergence.json)与[历史计划](results/qkv/plans.json)一并保存。

## 本轮具体改变了什么

[上一轮小数据重构](reconstruction_results.md)能拟合已见 A/B，却未建立新组合的在线写入能力。当前系统早已有 Wq/Wk/Wv、真实层输入构造的 query、稀疏候选中的 softmax 与连续 value 混合，因此本轮不能写成“首次加入 QKV”或“从离散查表变为 attention”。本轮检验两个更具体的因素：训练时打断固定实体—内容绑定，及先选择完整事实组再读取组内内容。

冻结 Qwen3-4B-Instruct-2507，revision `cdbee75f17c01a7cc42f958dc650907174af0554`。第 20 层 down projection 的实际输入构造 query；rank/key dimension 均为 64，每条事实存三个真实 contextual word-span 向量。基础向量库禁用，评估使用 CPU VDB，学生 prompt 只有问题。三槽均由共享线性 Wk/Wv 编码；可学习 `slot_position[3,64]` 在 key 线性投影之后、L2 归一化之前相加，零初始化。它不是人工指定正确下一词槽的 oracle。

| 臂 | 实体—内容训练关系 | 实际稀疏读取 | 残差注入 | 是否参与 VeRA 候选选择 |
| --- | --- | --- | --- | --- |
| static_flat | 每实体固定 A/B | 全库 top-4 slots | 乘性 VeRA | 是 |
| static_grouped | 每实体固定 A/B | top-1 fact，再混合该组三槽 | 乘性 VeRA | 是 |
| rebind_flat | 每 epoch 随机重绑定 | 全库 top-4 slots | 乘性 VeRA | 是 |
| rebind_grouped | 每 epoch 随机重绑定 | top-1 fact，再混合该组三槽 | 乘性 VeRA | 是 |
| rebind_grouped_additive | 同 rebind_grouped | 同 grouped | 加性向量读出 | **否，仅诊断** |

flat 按所有 slots 的 cosine 取四个；grouped 先按每个 fact 的三槽 `logsumexp(score/0.05)` 选一个 fact，再对该组三槽 softmax。每个生成 token 都重新由实际层输入查询。grouped 的 fact 选择是离散操作，生成 CE 不向未选 fact 传播选择梯度；共同的 dense fact-group 地址 CE 提供额外地址监督。本轮没有实现独立的 entity-only key，也不能把 logsumexp 组分数称为已经完全分离地址与内容。

主线残差为 `b ⊙ B[(Ax) ⊙ m]`；加性诊断为 `b ⊙ Bm`。全部模型 B 可学习，A 固定；加性臂保留相同 A buffer 但不使用其乘性路径。同种子五臂共有完全相同的初值和 train-only 统计，可训练参数均为 **2,034,368**。纯 FP32 K/V 每事实 **1,536 bytes**，16 条银行 **24 KiB**，不含 ID、时间戳、索引与对象开销。flat 读四个 values，grouped 读三个，参数/存储相等不代表相同 FLOPs。CPU exact search 扫描所有 keys，稀疏的是取回和混合，不能宣称 ANN 或次线性查找。

## 训练日程与可识别范围

数据种子 `121042`，优化种子 `81042 / 81043 / 81044`；五臂各三种子，固定 2,048 更新终点，不选最佳中途 checkpoint。每模型每步八个目标，A/B 合批为十六个独立银行序列，一次 backbone forward。损失只有完整答案含 EOS 的 CE 加 `0.2 × fact-group 地址 CE`；两项先按每序列有效预测位置平均，再对序列等权。教师不参与 KD、hidden state 蒸馏或 OPD。

训练有 64 个实体、128 个独特三词 payload。每 epoch 八步覆盖每个实体一次、每个 payload 一次 A/B 监督；256 epochs 后每实体有 256 次目标曝光、每 payload 有 256 次监督序列曝光。static 固定绑定，rebind 每 epoch 对 128 个 payload 作确定性排列再分配 A/B。相同种子所有臂共享目标、背景实体和银行物理顺序，单样本 B 银行只替换该样本自己的目标，其余十五条仍是 A；不是一次把八个目标全换 B。

每个样本的银行为十六条，含当步八个目标和另外八个随机背景实体。目标 payload 总曝光相同不等于背景内容驻留相同；两者分别记录。非 padding token 总量按日志逐序列复算，跨 regime 的 padding 可能不同，不强行声称完全相同计算量。

支持特征对真实 **64 × 128 = 8,192** 个 entity+payload 文本逐一冻结编码，沿用 `Remember this information: ` 包装，并按 tokenizer 对齐的三个实际 word spans 池化。这个 8,192 是缓存笛卡尔积大小，不是 8,192 条独立训练事实或每个组合均被监督的证明。准备过程通过 `payload_features(supports, answers)` 定位已观察support中的三词payload span，再分成三个词槽；内容来自合法观测，但依赖结构化三词字段及已标注/可定位span，并非自由长文中自动发现任意事实边界的writer。学生推理问题没有原文或答案，不代表自然Wikipedia文档自动写入已被实现。共享特征统计仅从 canonical static A/B 的 128 条记录和 64 条 query 拟合，不用整个笛卡尔积、dev/confirm 或 C/D；训练时 Wk/Wv 使用当前参数重编码，不能把陈旧投影后的 K/V 冒充可训练 writer。

[训练末128步描述](results/qkv/training_convergence.json)显示，static六模型平均CE为0.00232–0.01412，rebind九模型（含additive）为0.31948–0.57729。相同更新/总payload曝光没有配平任务难度、每个绑定的监督密度或最终拟合程度；本轮没有增加rebind预算至同等loss的对照，因此只能判断固定配方/预算的效果，不能证明容量不可能或训练到充分优化后的能力上限。末128步只作描述，未用于checkpoint或模型选择。

上轮每实体有 512 次目标曝光，本轮是 256；实体/内容量、数据、slot-position 与日程也变化，不能用跨轮分数直接估计单一 QKV 改动效果。主要比较限于本轮四主臂的 2×2 矩阵；additive 只与对应 rebind_grouped 解释注入形式差别，不纳入主矩阵主效应或替代成功门槛。

## 同银行大小的数据与在线干预

known、dev、confirm 均各十六条，训练样本银行也均为十六条。known 使用训练前十六个实体；dev/confirm 分别使用不同的新实体。三种实体组按行共享对应 A/B payload，故新实体 A/B 主要隔离实体变化，不把它误称为新内容组合测试。但rebind的known A/B是训练见过的实体和完整payload的固定配对测试，不保证每个特定绑定均被目标监督。[实际绑定曝光审计](results/qkv/binding_exposures.json)确认，32种known canonical A/B绑定在三个种子的目标监督次数范围为1–10、1–8、0–7；种子81044有3种从未作为该目标绑定接受监督（entity index3/A、10/B、12/B）。这不等于未在背景、冻结编码或初始化统计出现。同seed三种rebind臂共享此日程，需避免把它们的known A/B弱表现全部归为遗忘或容量不足。

C/D 是由已见词构成的未训练完整组合，同样在 known/dev/confirm 间按行共享。行 `i` 仅替换 A 的第 `i%3` 个词，三个改动位置为 6/5/5，C/D 替换词不同；它们不是未见词汇或任意语义泛化。训练实体与完整 payload 按协议排除上一轮数据；模板和词汇沿研究历史复用，heldout 只指本轮训练未见的表达，不是整项研究首次接触的模板。

C 用于开发选择，D 与 confirm 生成/评分在候选封存后进行。全部冻结特征（包括 D）允许事先计算；封存的是其梯度、拟合统计、评分和选择用途，不能说 D 从未被计算。相位 `CC / HC / CH / HH` 的第一字母指 support，第二字母指 query。

每次从干净 A 银行独立执行目标 `A→X→A`，所有非目标记录保持不变。先生成本 phase 的十六条 A；每次更新后生成目标 X、固定邻居 `(i+1)%16` 的 A 和恢复后的目标 A。除 SWAP 外，还在同一世界分别运行 value-shuffle 和 empty。shuffle 以整条三槽 fact 为单位置换 values、保持 keys；SWAP 是把 donor `i^1` 的 A 内容写入 target，donor 本身不变。

A/B pair 要求初始 A 和更新 B 同时正确。C/D update+restore 要求新内容正确且回写 A 正确，起始 A+新内容+回写 A 三时点联合率另外报告。`locality_joint` 要求邻居更新前后都正确，同样答错或只是文本不变均不足。空库面对同问题不同答案的 paired EM 必然为零，因此采用同世界单边 real−shuffle/empty 差作为记忆依赖检查。

## 开发结果与不可更改的选择

以下为真实开发结果。三元组按 `81042 / 81043 / 81044` 排列，每个分母均为 16，不把三个种子或多次世界干预当作独立新事实。

| 臂 | known A/B pair | known C update+restore | known C locality joint | 新实体 dev C 单边 EM |
| --- | --- | --- | --- | --- |
| static_flat | 16 / 16 / 15 | 0 / 0 / 0 | 16 / 16 / 14 | 0 / 0 / 0 |
| static_grouped | 16 / 16 / 16 | 0 / 0 / 0 | 16 / 16 / 16 | 0 / 0 / 0 |
| rebind_flat | 5 / 1 / 0 | 0 / 0 / 0 | 8 / 2 / 2 | 0 / 0 / 0 |
| rebind_grouped | 6 / 1 / 0 | 0 / 0 / 0 | 7 / 3 / 3 | 0 / 0 / 0 |
| rebind_grouped_additive，诊断 | 4 / 10 / 4 | 0 / 0 / 0 | 10 / 14 / 11 | 0 / 0 / 0 |

所有模型、两个开发实体组的C单边real/shuffle/empty均为0/16，两个real-control差均为零；C的pair、update+restore、三时点联合率也均为零。已见static A/B可重构与新内容可写性没有同时成立，不能以其高训练绑定分数替代新组合门槛。

| 臂 | known B邻居联合正确（/16） | 新实体dev A/B pair（/16） | known SWAP单边EM（/16） |
| --- | --- | --- | --- |
| static_flat | 14 / 15 / 14 | 0 / 0 / 0 | 1 / 2 / 1 |
| static_grouped | 15 / 16 / 15 | 0 / 0 / 0 | 1 / 2 / 0 |
| rebind_flat | 8 / 3 / 2 | 4 / 0 / 0 | 10 / 2 / 1 |
| rebind_grouped | 8 / 3 / 2 | 2 / 2 / 0 | 9 / 3 / 0 |
| rebind_grouped_additive | 10 / 14 / 11 | 5 / 2 / 3 | 9 / 13 / 11 |

static两个臂的新实体dev A/B pair各种子均为零；rebind_flat为4/0/0、rebind_grouped为2/2/0，加性诊断为5/2/3。重绑定改善了部分实体迁移和SWAP的已训练内容重分配，但乘性臂known固定canonical A/B pair也明显下降，不能只挑最好种子称为稳定泛化。这里新实体A/B完整payload都来自训练，是新关联而非新三词组合。known B的邻居联合正确也低于目标单边正确，局部性需独立报告。

四个 VeRA 臂要在每个种子都满足：known A/B pair ≥12/16、known C update+restore ≥12/16、C real 对两个控制各领先 ≥0.50、C 邻居联合正确 ≥15/16、新实体 dev C 单边 EM ≥8/16。通过后先按最差种子的 known C update+restore 排，再按最差种子的新实体 C 单边 EM，之后按参数、字节与固定臂序打平。additive 永远排除。没有合格臂则 `selected_arm=null`，不能选一个相对最高但未过门槛者。

开发选择在 `2026-10-06T13:50:07.472005+00:00` 封存；45/45模型条件及46条artifact evidence通过检查，后一数字含一条128/128教师预检。所有四个VeRA候选的三个种子均未通过完整门槛，无候选。selection SHA-256为 `c6979915d7edc2aaaffda0f609ef45bf830590a752350ac16003069e7e587628`。开发教师另保存并审计，不混入四臂的效果排序。确认开始后只读取原封存选择，不重新 `--select`。

## 教师、完整答案与真实路由

训练教师预检已由真实原始预测核对为 canonical A/B 各 64/64，共 **128/128**，其不可变 receipt 已进入 registration。教师获得目标单条原始支持文本和同一问题，显式绕过 VDB；学生只获得问题及整个事实银行，两者权限不同。教师只判断文本任务可行，不参与本轮训练损失。

全部448次教师生成均完整正确，开发和确认teacher-paired eligible子集与完整分母一致。学生C/D失败不能沿用原始Wiki教师不合格的解释；教师仍有直接原文权限，结果并不意味着三词信息必须能由本接口轻易压缩和解码。

| 教师条件 | 设计生成数 | 实际完整正确数 | A/X 配对资格 |
| --- | ---: | --- | --- |
| train 预检 CC A/B | 128 | 128（已审计） | 64/64（已审计） |
| known 开发 CC A/B/C | 48 | 48 | A/B、A/C均16/16 |
| dev 开发 CC A/B/C | 48 | 48 | A/B、A/C均16/16 |
| known 确认 CC A/D | 32 | 32 | A/D为16/16 |
| confirm 确认四相位 A/B/D | 192 | 192 | 各相位A/B、A/D均16/16 |

开发C在known和dev共480条预测记录中，完整EM和完整答案包含均为零，478条恰好三词、没有32-token预算命中。480是相同小数据在十五模型、两个实体组上的重复记录数，不是480条独立内容；两个实体组共享C payload。这些记录排除了“零完整EM仅由多余格式或普遍截断造成”的解释。

逐词诊断采用**输出恰好三个规范化词**的严格口径，其他长度逐词也计错；与summary中只要求对应位置存在的辅助计数区别开。known C结果如下，均为每种子/16：

| 臂 | 被编辑词正确 | 另外两个未改词同时正确 | 完整三词正确 |
| --- | --- | --- | --- |
| static_flat | 3 / 3 / 2 | 8 / 7 / 5 | 0 / 0 / 0 |
| static_grouped | 2 / 2 / 2 | 5 / 6 / 5 | 0 / 0 / 0 |
| rebind_flat | 8 / 5 / 6 | 4 / 1 / 1 | 0 / 0 / 0 |
| rebind_grouped | 4 / 7 / 6 | 1 / 2 / 1 | 0 / 0 / 0 |
| rebind_grouped_additive，诊断 | 10 / 8 / 11 | 5 / 5 / 4 | 0 / 0 / 0 |

部分新词生成确实改善，但与未改两词的保持没有在同一条回答中同时成立，不能概括为“完全没有读取新内容”。known C中，输出恰为某条128个训练payload的比例随模型为4/16至13/16；并非所有输出都局限于这些标签，也并非全部仍输出本实体旧A。该表是开发描述，不用于追加规则或调参。

[教学案例](results/qkv/examples.json)固定使用种子81042、known/CC开发条件，属于事后目的性说明，不代替全量指标。例如把 `mirror crystal badger` 改为 `mirror delta badger`：static_grouped输出 `mirror glass frame`，rebind_grouped与additive均输出 `mirror delta cougar`。三者首步与decode都实际读取目标组；重绑定两臂改对第二词，但末词仍错。另一个第三词替换例子中，static_grouped与additive整个轨迹都读取目标组，却仍输出旧A；rebind_grouped则读取错组。这说明不同样本可以同时存在路由问题与读出问题，不能用某一个例子锁定统一根因。路由质量仍不等于attention的因果贡献。

三个槽来自上下文化特征，改动前面的词还可能改变后续词槽的 hidden；`edited_word_index` 对应槽并非唯一可能携带新信息的槽。没有读取这个槽不能单独证明完全没有获得新内容，读取它也不能证明内容已被正确解码。

本轮 raw 按每个真实生成 token 保存 ID、首次预测或 decode 标记、slot indices/scores/weights、目标 fact mass 和三个 slot masses。首预测取最后 prompt 位置，非问题首位置。**R@read** 是实际取回的四个 flat slots 或三个 grouped slots中是否含目标 fact；**R@1** 按最大实际权重槽定义，grouped 返回顺序是 slot0/1/2，不能用列表首项冒充最高排名，也不能统称相同的 R@4。

known C首预测R@read：static_flat为15/15/15，static_grouped为14/13/12，rebind_flat为16/16/16，rebind_grouped为8/16/16，加性诊断为14/15/16，分母均16。grouped选中目标组时会实际混合其三个槽，但两个rebind_grouped种子即使首次目标组16/16命中，完整答案仍为零。说明首步地址命中不足以保证组合重构，不能直接证明所有后续寻址已经正确或纯reader是唯一根因。自由生成的错误前缀会改变下一步query，因此decode路由偏离既可能促成错误，也可能是先前错误的结果；仅靠相关trace不能确定因果方向，更不能锁定value encoder。

known C的gold-prefix NLL，rebind_grouped为3.199–3.257，加性诊断为2.029–2.630；后者较低且被编辑词更常正确，仍没有完整答案。这支持保留局部能力改善与整条失败的双重描述，不能据此宣布加性方案成功。命中目标fact不等于给改动词槽足够权重；整个生成曾访问某槽也不保证在所需预测时刻使用。known D的首预测R@read在rebind_flat仍为16/16/16、rebind_grouped为7/16/16、additive为14/16/16，完整D仍全部为零。纯 JSON 汇总器只核 token ID 与 trace 的逐位置一致及源改动词槽权重，不重新执行 tokenizer。若最终报告声称“第几个实际生成词”的对齐，必须引用另行经过 tokenizer 累计解码的诊断，不能用 gold 强行指派。

训练在第1步及每128步捕获含 gold 前缀、含 EOS 的预测位置，记录 input/mixed-value/residual RMS、目标 fact mass 和实际读取宽度；该诊断使用更新前参数、不新增 backbone forward或损失。它不是自由生成时的分布，也不证明幅度差异就是因果根源。加性/乘性效果比较应连同这些尺度报告。评价答案 NLL 不包含 EOS，训练 CE 包含 EOS且按序列等权，不能直接跨口径比较或用 NLL 代替可写性。

## 最终确认与是否扩样

![QKV各模型固定终点与预先固定门槛](results/qkv/qkv_endpoints.png)

全部确认已完成。下表三元组按三个固定种子排列，每项分母16；图中每个点是同一数据上的一个种子，没有把种子当独立样本绘制置信区间。

| 臂 | known D update+restore（/16） | known D locality joint（/16） | 新实体 confirm CC D 单边 EM（/16） | 新实体 CC B pair（/16） |
| --- | --- | --- | --- | --- |
| static_flat | 0 / 0 / 0 | 16 / 16 / 15 | 0 / 0 / 0 | 1 / 1 / 0 |
| static_grouped | 0 / 0 / 0 | 16 / 16 / 16 | 0 / 0 / 0 | 1 / 0 / 0 |
| rebind_flat | 0 / 0 / 0 | 8 / 2 / 2 | 0 / 0 / 0 | 2 / 0 / 1 |
| rebind_grouped | 0 / 0 / 0 | 8 / 3 / 2 | 0 / 0 / 0 | 3 / 1 / 0 |
| rebind_grouped_additive，诊断 | 0 / 0 / 0 | 10 / 14 / 12 | 0 / 0 / 0 | 3 / 4 / 3 |

所有新实体confirm相位的D单边real/shuffle/empty、D update+restore均为0/16。新实体B pair在HC、HH所有模型为零；CH只有rebind_flat种子81042为1/16，其余为零。CC的A/B迁移也很有限，表中additive为3/4/3，未建立稳定跨实体或跨表达的可写记忆。D邻居联合正确与controls完整保留在汇总中，不以目标答案为零省略。

独立从raw重读known D的240次重复预测：完整EM和答案包含均0，238条恰好三词且无预算命中；新实体confirm D四相位共960条重复预测同样EM/包含均0，868条恰三词、6条命中32-token上限。格式和截断不能单独解释全部失败。这些是共享payload跨模型/实体/格式的重复记录，不能当作1200条独立内容或计算独立样本置信区间。

只有开发选定的 VeRA 架构有资格触发扩样：三个种子分别 known D update+restore ≥12/16，D real 对 shuffle、empty 各领先 ≥0.50，D 邻居联合正确 ≥15/16，且新实体 confirm CC D 单边 EM ≥8/16；对应教师还需全资格通过。其他架构的确认结果即使更好也不能代替预选者。无候选时 `final_gate.per_seed.complete=false` 的含义是无可判定的预选架构，并不自动表示缺少全矩阵确认；须同时报告 `all_formal_results_complete`。

最终 `eligible_to_expand=false`，不扩大语料。开发已封存 `selected_arm=null`，因此`final_gate.per_seed.complete=false`表示没有可评判的预选架构，**不表示作业缺失**；`all_formal_results_complete=true`，75个模型条件和5个教师作业实际全部完成。这些是预先固定的工程门槛，不是统计显著性或部署质量保证。

## 实测成本与审计证据

正式十五个训练已经全部完成并严格复算：30,720更新、245,760次目标曝光、491,520个学生序列、30,720次训练backbone forward、2,549,760个含EOS的gold tokens、37,524,480个实际输入位置和38,682,400个含padding位置。训练进程时间累计2,745.208秒，不是独占GPU延迟。

实际完整学生评估与固定预算一致：每模型四个eval条件共1,200次自由生成、224次独立干预、224次恢复；全矩阵18,000次学生生成、3,360次干预加3,360次恢复，共6,720次real group写入。教师另有448次生成，正式合计18,448次。prepare另列，smoke另列；初始化填库和shuffle/empty控制物化不伪装成目标在线写入。

| 实测成本项 | 实际值与证据 |
| --- | --- |
| 正式完整训练/学生评估/教师条件 | 15 / 60 / 5（另1个prepare） |
| 训练更新、目标、序列、backbone calls | 30,720；245,760；491,520；30,720 |
| gold含EOS tokens、实际输入位置、含padding位置 | 2,549,760；37,524,480；38,682,400 |
| 学生生成次数/tokens/答案评分tokens | 18,000 / 90,306 / 77,175 |
| 干预、恢复、real group写入 | 3,360 / 3,360 / 6,720 |
| 教师生成次数/tokens | 448 / 2,359 |
| 训练进程时间、学生生成计时、教师生成计时累计 | 2,745.208 / 3,757.202 / 69.867秒；非独占GPU延迟 |
| smoke/excluded计算 | 显式smoke和排除产物单列，不进入上表 |

严格汇总器只读 raw、summary、manifest、训练日志与文件SHA，复算规范化 EM、严格联合指标、teacher资格、token与trace对齐、weight/score softmax、slotmass、实际读取宽度、所有干预/回写与调用数。它独立重建随机episode、逐步mapping/目标/背景/顺序、曝光计数和日志中的序列长度，验证2,048步终点checkpoint lineage及15×4学生网格；不反序列化权重，也不重新运行模型或tokenizer。

正式 train/eval 必须逐文件匹配登记的44个runtime源码SHA、协议SHA、注册计划及其中显式参数、数据cache SHA；各不可变snapshot也分别核对。prepare和teacher preflight在后续纯日志字段完善之前运行，保留各自snapshot，分别绑定登记的feature result与preflight receipt，不错误要求它们含后来新增的诊断代码。远端与本地存在约315秒钟差，因果顺序按封存、source/data SHA和调度证据核对，不直接减两个机器的墙钟字符串。

[独立张量审计](results/qkv/independent_audit.json)已通过：`complete=true`、`checks_passed=true`、正式矩阵完整；CPU运行79.514秒，未初始化CUDA。覆盖81个正式作业（15train、60eval、5teacher、1prepare），另列8个smoke作业的52次生成、8次干预和8次恢复。已有冻结特征的writer重编码最大绝对误差（含smoke）为key 5.22×10⁻⁷、value 3.37×10⁻⁶。银行单条写入、恢复、control置换及哈希经CPU回放；已记录的selected indices/scores、softmax权重、slot mass和token位置按日志逐一复算。训练初值、统计、终点与Adam状态亦有独立检查。四个确认suite的选择快照都等于原封存SHA，46条开发/预检证据的288个文件重新哈希不变。

未保存实际完整query向量，故此审计不能离线重做并证明全库top-k或获胜group的最优性；选择路径由冻结运行源码与日志绑定，已选分数间自洽不等于完整查询重放。需明确区分已有冻结特征的writer重编码、保存银行的CPU写改回放和训练初末参数检查，与真正从原文重跑Qwen骨干或自由生成。审计没有重新运行语言模型预测；初末状态和日志也不能单独证明每个中间时刻绝无非授权梯度，来源哈希不是可信执行认证。

## 文献如何解释这些结果

[Attention](https://arxiv.org/html/1706.03762v7#S3.SS2)本来就对已有 values 加权混合，因此“混合已有向量”与attention式泛化不矛盾。应检验网络是否根据当前query和当前写入内容作正确计算，而非只看是否有softmax、更多QKV参数或更高检索分数。

[Fast Weight Programmers §6.1](https://arxiv.org/html/2102.11174v3#S6.SS1)直接支持随机绑定和覆盖测试的动机，但其随机K/V合成实验使用固定one-hot values与回归损失；本轮是冻结指令LLM上的三词自由生成和外部数据库编辑，不是该论文的复现。[LongMem §2.3](https://arxiv.org/html/2306.07174#S2.SS3)先用chunk mean key检索，再展开块内token K/V attention，支持分组内容读取的思路；本轮用slot分数logsumexp选fact，reader仍是单层rank64接口，不能写成同一架构。LongMem报告的26B-token适配、12层SideNet也不能用来预测本微型预算的效果。

[Memorizing Transformers](https://arxiv.org/abs/2203.08913)说明非可微外部检索与可学习reader可以共同工作，其values进入attention激活，与VDB向量调制VeRA参数不同。[DeltaNet](https://arxiv.org/html/2406.06484v3)和[Gated DeltaNet](https://arxiv.org/html/2412.06464v1)提供压缩关联状态的误差校正/门控写入，但当前VDB已经物理upsert单条记录；答案继续输出旧内容不能未经诊断就归因数据库没有覆盖。完整文献与训练规模见[文献映射](qkv_literature.md)。

开发证据表明，改变训练绑定分布对已训练内容的新关联和部分新词有影响，但既未稳定恢复canonical A/B，也未完成新组合。分组读取没有把完整C从零变为成功；additive在部分词与NLL上较好，同样完整C为零。因而当前失败不能只归于VeRA乘性项，也不能归纳为所有QKV/attention/外部记忆不可行。更深reader、多头、独立地址编码、不同监督或足量内容重构训练均未由本轮覆盖，不能据本结果否定；新增这些条件也不能冒充已完成实验。

## 给学生的复算与后续约束

先读本报告的门槛、数据权限和局限，再核验三份入口：`key_facts.json`提供逐种子整数计数与来源SHA，`summary.json`提供全部条件和成本，`selection.json`提供只用开发证据的不可更改选择。任何新报告都应把训练拟合、在线新内容写入、实体迁移和表达迁移分开呈现。

完整本地备份后，从仓库根目录复算，不需要GPU、SSH或模型：

```bash
python3 scripts/summarize_qkv.py \
  --runs-root ../runs/xtrah100 \
  --registration ../plans/qkv_registration_20261006.json \
  --selection ../plans/qkv_selection_20261006.json \
  --output ../plans/qkv_summary_final_20261006.json
```

独立张量回放需使用安装PyTorch的项目Python环境，本次为`/home/chenhan/miniconda3/envs/agent/bin/python`；无需加载Qwen骨干或GPU：

```bash
/home/chenhan/miniconda3/envs/agent/bin/python scripts/audit_qkv.py \
  --runs-root ../runs/xtrah100 \
  --registration ../plans/qkv_registration_20261006.json \
  --selection ../plans/qkv_selection_20261006.json \
  --require-final \
  --output ../plans/qkv_independent_final_audit_20261006.json
```

`--select --selection-output ...`只用于历史开发封存：需15train+30dev齐全、教师结果已保存、无任何已启动确认child。确认之后不重新调用该模式。测试入口为`python -m pytest tests/test_summarize_qkv.py -q`。本轮记录的完整pytest为1,163项通过（114.73秒），见[测试记录](results/qkv/tests.json)；独立审计器另外14项自测通过，不把后来新增测试的collection数量冒充又一次全量通过。各历史启动命令、plan、source snapshot与revision按registration和suite保存。公共summary SHA-256为 `d1876f4720cbfd0f906e1d874a75f094d10b0008c3ad4fb9c58775606e5a6ba7`；key_facts同时记录此SHA、选择和诊断来源，完整本地备份可复算。

若继续做微型机制诊断，可先固定正确fact、让模型自行读取组内内容，明确命名为oracle-correct-fact额外权限对照；它用于隔离自由寻址与内容读出，任何成功都不计主线VDB自主检索成功。之后可分别检验地址/内容分头、内容重构辅助监督与逐生成前缀的地址稳定性，避免一次同时修改并失去可解释性。这些是尚未实施的建议，不是本轮结果；本轮没有测试MLP、多头或真正独立的地址/内容编码。

本轮不通过扩样门槛；继续工作应先另立微型诊断协议，不能用追加大语料替代完整重构验证。以后即使通过，仍需另行封存更大银行、更多干扰、自然语料和更广表达，不能从十六条受控编辑直接外推长期持续学习或效率优势。新增oracle地址、逐词标签、teacher KD或额外预算都应明确是新条件。
