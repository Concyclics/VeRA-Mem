# 整块动态低秩记忆：完整实验结果

**整块动态算子带来了有限、可观察的改善，但仍未解决新内容组合与跨表达泛化。** 本轮18模型、72学生评估、5教师任务和1次特征准备共96个正式任务全部完成，教师448/448正确；全部模型固定终点、全部条件完整报告，没有结果后择臂或择种子。

随机绑定的 `block_outer` 在新组合C上更新并恢复正确 `3/5/2` 条，同参数 `pooled_outer` 为 `0/0/0`；独立确认D的更新正确为 `6/7/3`，更新并恢复为 `6/6/3`，pooled相应为 `1/0/0`。方向上，逐槽构造算子比先池化在这两组新组合上更好；但相对原对角reader，C并非每个seed都改善，新实体与新问题表达仍弱，预注册的三seed一致门槛未通过。**不据此扩大Wikipedia语料。**

本文所有斜杠三元组依种子91042、91043、91044排列，每项分母16，绝不是单个比率。三个seed共用微型数据，不能把重复事实当作独立语料样本。结论以[完整汇总](results/block/summary.json)、[紧凑指标](results/block/key_facts.json)、[逐词诊断](results/block/answer_diagnostics.json)为依据；实验条件见[封存协议](block_protocol.md)、[登记文件](results/block/registration.json)和[全部计划](results/block/plans.json)。

## 1. 本轮真正改变了什么

固定骨干为Qwen3-4B-Instruct-2507，原模型全部参数冻结，在第20层down projection使用rank64接口。实际层输入生成query，真实CPU VDB按QK组分数选top-1事实，返回完整的三个64维value。三种reader均采用块内uniform权重，保留相同的对角路径；CPU读取将选中块的三行value传入reader，再构造相应算子。

记 `z=Ax`、`m=mean_s(v_s)`、`u(v)=L2Normalize(P_in v+c)`、`w(v)=P_out v`：

| reader | 作用于z的latent算子M |
| --- | --- |
| diagonal | `diag(m)` |
| pooled_outer | `diag(m)+w(m)u(m)ᵀ` |
| block_outer | `diag(m)+mean_s[w(v_s)u(v_s)ᵀ]` |

最终增量都是 `Δh=b⊙B[Mz]`。新增外积使latent坐标可非对角混合；不是把共享B搬到加权求和之前。后者在没有中间非线性时满足严格线性等价，不能增加容量。具体文献映射见[block_literature.md](block_literature.md)，固定公式、初始化和归约见[block_protocol.md](block_protocol.md)。

两个outer臂的附加参数均为8,256，总可训练参数2,042,624；diagonal为2,034,368。每事实的纯FP32 K/V均为1,536 bytes，16事实为24 KiB，没有额外存储64×64矩阵；共享投影在读取时生成因子。block附加项rank≤3，pooled附加项rank≤1，**完整的对角加外积算子仍可能rank64**，三个reader最终均经过共同宽度64的B。相同参数与记忆字节不等于相同FLOPs。

这一reader在选中事实后，对V的行置换不变。它不是有显式槽顺序的逐词解码器，也不是多头原生Transformer KV attention；contextual word-span向量可携带原文前缀和顺序信息，不能由行置换不变再推断完全没有顺序信息。本轮没有测试多头、非共享slot映射、多个原生注意力层或更宽B。

## 2. 数据、预算和封存边界

新数据种子221042，排除reconstruction/QKV历史208实体和608个完整答案。训练为64实体、128个三词payload；每个训练支持均按真实entity+payload文本重编码，共8,192个冻结特征组合。writer用已观察支持文本中已标注/可定位的三词span；这仍是结构化任务，尚未实现从任意Wikipedia文档自动发现事实边界。

static对每实体固定A/B；rebind每epoch重新排列绑定。每模型2,048更新，batch8目标、合计16个A/B序列，每样本有独立16事实银行，B仅更新自身目标。每8步覆盖所有64实体及128payload；训练只有含EOS的完整答案CE和0.2 dense地址CE。6臂×3seed，共18模型，全部评估固定最终checkpoint；没有根据C筛选模型或早停。

known/dev/confirm银行均为16条，后两者实体新；三个packet按行共享payload以隔离实体变化。C/D各只改A的一个词，组件词均已训练、完整组合未训练。heldout模板仅相对本轮训练未见，历史研究已经使用。D/confirm特征可提前编码，其评分不用于梯度、统计或模型选择。

known A/B应称“训练出现的实体与完整payload的固定配对测试”。在rebind中，这些特定绑定的目标监督平均次数分别为4.0625、4.40625、4.21875；seed91042有两条A绑定从未作为目标监督，另外两seed无此零项。static每条固定绑定为256次。零目标曝光不代表未做过特征编码、训练统计或背景驻留，不能笼统叫“完全未见”。

教师只用于资格检查：原始骨干获得当前目标单条context和相同问题，不访问VDB、不追加gold标签，不参与蒸馏。最终教师448/448正确：训练A/B预检128、known/dev开发各48、known D确认32、新实体四相位A/B/D确认192。每个新实体phase的A/B/D各16/16，所以有教师对应的非SWAP学生条件，其teacher合格子集与全样本一致，未按教师成功筛题。

## 3. 开发终点：部分组合信号，不是稳定胜出

所有数字均为三seed正确数，分母16。C的`update_restore`要求写入C后正确且回写A后正确；本表中的C updated与update_restore逐项相同，不把二者视为天然等价。

| 臂 | known A/B pair | known C updated | known C update_restore | dev新实体C updated |
| --- | --- | --- | --- | --- |
| static_diagonal | 16/16/16 | 0/2/0 | 0/2/0 | 0/0/0 |
| static_pooled_outer | 16/16/16 | 0/1/0 | 0/1/0 | 0/0/0 |
| static_block_outer | 15/16/16 | 0/0/0 | 0/0/0 | 0/1/1 |
| rebind_diagonal | 15/16/14 | 2/0/4 | 2/0/4 | 0/1/1 |
| rebind_pooled_outer | 15/16/13 | 0/0/0 | 0/0/0 | 0/0/0 |
| rebind_block_outer | 15/15/14 | **3/5/2** | **3/5/2** | **1/5/0** |

known C的shuffle/empty全部为0。rebind block的邻居before/after联合正确为15/15/15；其余static为16/16/16，rebind diagonal和pooled均15/16/14。这个联合指标比“输出文本未变”严格，不把两次都错算作局部性成功。

严格恰好三个规范化词的诊断如下。“保留两词”要求另两个词同时正确，但不要求改动词正确；两列的成功样本未必相同，不能相加。

| 臂 | known C改动词正确 | known C另两词同时正确 | dev C改动词正确 | dev C另两词同时正确 |
| --- | --- | --- | --- | --- |
| static_diagonal | 3/5/3 | 6/5/3 | 2/5/4 | 0/0/0 |
| static_pooled_outer | 2/2/1 | 3/6/6 | 2/2/4 | 1/0/0 |
| static_block_outer | 1/4/5 | 6/6/7 | 2/5/4 | 0/1/1 |
| rebind_diagonal | 8/8/8 | 7/6/9 | 4/5/5 | 2/2/5 |
| rebind_pooled_outer | 4/4/6 | 4/5/5 | 3/4/4 | 1/3/5 |
| rebind_block_outer | 9/12/7 | 8/8/8 | 6/8/4 | 4/8/5 |

逐词改善说明不能再概括成“完全没有读取新内容”，但只有完整答案正确才能说明三词组合在该例成功。格式、长度、完整答案包含率、旧A/B回退和任意训练payload输出比例在[answer_diagnostics.json](results/block/answer_diagnostics.json)逐条件保留，不以首词或单槽命中代替EM。

## 4. D确认：局部收益延续，完整门槛未通过

D是训练与开发评分均未使用的新完整组合。各模型都从独立A银行开始写入D并回写A。**updated正确和update_restore联合正确须分开：** rebind block的seed91043有7条D答对，但其中只有6条回写A也正确。

| 臂 | known D updated | known D update_restore | known D shuffle | known D empty | known D locality_joint |
| --- | --- | --- | --- | --- | --- |
| static_diagonal | 0/2/1 | 0/2/1 | 0/0/0 | 0/0/0 | 16/16/16 |
| static_pooled_outer | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 | 16/16/16 |
| static_block_outer | 2/3/3 | 2/3/3 | 0/0/0 | 0/0/0 | 16/16/16 |
| rebind_diagonal | 3/3/2 | 3/3/2 | 0/0/0 | 0/0/0 | 15/16/14 |
| rebind_pooled_outer | 1/0/0 | 1/0/0 | 0/0/0 | 0/0/0 | 15/16/14 |
| rebind_block_outer | **6/7/3** | **6/6/3** | 0/0/0 | 0/0/0 | 15/15/15 |

同参数对照中，rebind block相对pooled的D update_restore提升为5/6/3条；相对diagonal为3/3/1条。已知实体的新D内容有改善，但静态block的开发C仍为0/0/0、D才达到2/3/3，不能据确认D改称“所有新组合均稳定改善”。

新实体确认D的单边完整答案正确数如下。CC/HC/CH/HH依次指canonical/canonical、heldout support/canonical query、canonical support/heldout query和两者均heldout。

| 臂 | confirm CC D | HC D | CH D | HH D |
| --- | --- | --- | --- | --- |
| static_diagonal | 0/1/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| static_pooled_outer | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| static_block_outer | 0/2/0 | 0/2/1 | 0/0/0 | 0/0/0 |
| rebind_diagonal | 1/1/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| rebind_pooled_outer | 1/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| rebind_block_outer | **3/1/1** | **1/0/0** | 0/0/0 | 0/0/0 |

新实体CC的update_restore为：static diagonal 0/0/0、pooled 0/0/0、block 0/1/0；rebind diagonal 1/1/0、pooled 1/0/0、block 3/1/1。所有D的shuffle/empty在四相位均为0。CC邻居联合正确，static三个reader分别为2/3/2、2/4/1、2/4/2，rebind为10/9/10、10/10/10、10/10/10；新实体初始重构本来就弱，不能把这种联合低分全部归因于在线更新破坏。

已训练完整payload的新实体CC A/B pair在static三reader都是1/0/1，在rebind三reader都是8/7/8。随机绑定帮助迁移实体—已训练完整内容的对应关系，但新组合、特别是heldout问题表达仍明显困难。所有phase的A/B、更新/恢复、locality和controls均保留在完整summary，没有只挑最好格式。

| 臂 | known D改动词正确 | known D另两词同时正确 |
| --- | --- | --- |
| static_diagonal | 5/3/4 | 6/8/6 |
| static_pooled_outer | 3/4/3 | 6/3/6 |
| static_block_outer | 8/7/4 | 6/7/10 |
| rebind_diagonal | 7/10/8 | 8/8/7 |
| rebind_pooled_outer | 7/7/5 | 4/6/6 |
| rebind_block_outer | 11/11/10 | 9/10/7 |

低完整EM不能仅解释为输出格式：known D的288次real updated生成中287次恰为三词，答案包含正确数与EM同为36；新实体四phase D的1,152次中1,110次恰为三词，答案包含数与EM同为15。两组都没有32-token截断。这里是跨模型的重复测量计数，不是独立样本推断。

预设继续研究门槛要求**三个seed全部**满足：known C和D update_restore相对同seed pooled各增加至少4/16；known A/B pair至少12/16；C/D real单边对shuffle及empty各领先至少4/16；教师合格。static的三个seed都未过；rebind只有91043单seed全过，91042未过C项、91044未过C/D项。因此两个regime的完整门槛都为false。它是事先声明的证据标准，不是统计显著性或“绝对无研究价值”的证明；本轮没有自动扩大语料，也不把最佳seed代表架构成功。

## 5. 机制证据：路径有效，但不能把收益唯一归于rank

独立训练审计确认18模型中，同seed六臂的共同初值逐位相同，同seed四个outer臂新增参数初值也逐位相同；12个outer模型全部8,256个新增坐标在训练后发生变化。新增参数各有2,047步非零梯度：第1步b=0使其梯度为0，是预定初始化行为，不是reader未接入。见[training_audit.json](results/block/training_audit.json)与[operator_diagnostics.json](results/block/operator_diagnostics.json)。

同seed、同regime三个reader训练终点的Wq/Wk/slot_position逐位相同；六组全部成立。第一步known C fact召回也完全一致：static为12/12/13，rebind为15/16/15。因此reader之间的差异不能由寻址参数学成不同值解释。不同生成答案会改变后续前缀及query，仍可能造成decode路由分歧；不能由参数相同宣称所有自由生成轨迹相同。该事后核对见[qk_matched.json](results/block/qk_matched.json)，不参与选择。

rebind block额外项在更新2048**之前**的gold-prefix诊断位置，平均有序奇异谱的`mean(σ₂)/mean(σ₁)`为0.126118/0.126156/0.126593，第三奇异值相应比为0.045665/0.046692/0.047933；该均值谱的熵有效rank约1.6515/1.6555/1.6613。它没有在这个汇总意义上完全变成一个外积，但第二、三方向较弱。

这些是先逐位置做SVD、再平均**额外outer项的有序谱**，不是平均矩阵的谱、不是每位置有效rank的均值，也不是完整diag+outer矩阵的谱；更不是最终checkpoint之后重新测的自由生成机制。非零次奇异值或梯度都不能证明其携带了有用的新组合信息。

末128步训练CE（含EOS、序列等权）为：

| 臂 | 91042 | 91043 | 91044 |
| --- | ---: | ---: | ---: |
| static_diagonal | 0.016650 | 0.009072 | 0.011087 |
| static_pooled_outer | 0.009743 | 0.008515 | 0.020774 |
| static_block_outer | 0.013861 | 0.008020 | 0.015788 |
| rebind_diagonal | 0.128898 | 0.123798 | 0.117460 |
| rebind_pooled_outer | 0.101995 | 0.093737 | 0.090810 |
| rebind_block_outer | 0.103902 | 0.092485 | 0.085125 |

outer在rebind中的训练拟合优于diagonal，但pooled与block平均CE很近，完整组合结果仍不同。相同步数不保证同样优化程度；CE小也不等于新组合生成成功。不能据固定预算负结果声称容量不可能。

开发事后案例提供一个失败边界：seed91042、known的dataset_index=2（第三条），A为`window lilac eagle`，C为`window lilac badger`，static三reader全程都读取正确事实，仍全部输出旧A；三槽全部被取回没有自动保证编辑词被读出。另一个rebind的dataset_index=0，C为`pencil jade dolphin`，diagonal/pooled输出`pencil jade badger`，block输出`pencil jade salmon`；三者全程读取正确事实且改对第一词，却都损坏未修改的第三词。这是按固定规则挑出的事后示例，不代表总体分母；[公开案例](results/block/examples.json)保留原始行号、来源SHA、完整轨迹与选例规则。

## 6. 成本、执行正确性与复算

已完成训练的总计为36,864更新、294,912目标、589,824序列、3,059,712个含EOS gold tokens、44,845,056实际输入位置、46,396,368 padded位置、36,864次backbone forward。训练过程秒数之和3,410.352；并行运行会重叠，不可称独占GPU耗时或用户等待延迟。

全部正式学生评估实际21,600次生成、106,565个生成tokens、92,502个答案评分tokens、4,032次目标干预及4,032次恢复，共8,064次real group写入。教师448次生成、2,347个生成tokens；正式总生成22,048次。学生generation过程秒数之和4,718.977，教师69.185，同样不可相加解释为独占GPU墙时。初始化填库、控制银行物化与smoke不计入这些在线logical writes或正式成本。

答案NLL不含EOS，是teacher-forced辅助指标；含EOS的训练CE不能直接按同口径比较。rebind known D的逐seedtoken NLL为diagonal 1.25623/1.16319/1.50252、pooled 2.04645/2.90323/2.55018、block 0.75091/0.82072/1.32259；更低NLL伴随部分EM提升，但不是完整组合泛化已经成立。

最终[独立CPU审计](results/block/independent_audit.json)完整通过，耗时116.695秒，无CUDA。它对72个正式学生评估的106,565个保存query逐一重放全库选组与完整三槽读取，同时复核银行写改、恢复、controls和token对齐；selected cosine最大差约2.98e-7，writer重编码key/value最大差约6.16e-7/5.28e-6。正式96任务全部完整，无pending；另6个smoke有39次生成、117个query及12次group写入，单列而未混入正式成本。

审计没有重跑LM生成或重新提取原始冻结特征；query来源仍依靠冻结源码及运行遥测。真实query的全库winner可以离线复核，但这不是独立重训/重生成整个实验；uniform权重也不代表signed outer因子的实际贡献。早期[开发审计](results/block/development_audit.json)保留其partial范围，最终完整性以independent_audit为准。

最终复算入口：

```bash
python scripts/summarize_block.py \
  --runs-root ../runs/xtrah100 \
  --output docs/results/block/summary.json \
  --require-final

python scripts/audit_block.py \
  --runs-root ../runs/xtrah100 \
  --registration docs/results/block/registration.json \
  --output ../plans/block_independent_recheck.json \
  --require-final
```

第二条命令要求输出文件尚不存在，以保留审计回执。

其输出summary、training_convergence、answer_diagnostics和key_facts都绑定实际来源SHA。[测试回执](results/block/tests.json)记录完整仓库1,249项通过；随后注册与新汇总器85项通过，另有CPU审计14项和训练机制诊断10项自测。不是宣称1,334项做过一次统一全量重跑。冻结正式运行源码的49个Python文件由登记与不可变快照核对；准备/预检早于最后遥测补充，各自源快照及回执保留，未强行改写成相同版本。

## 7. 下一轮建议：只作为待验证提案

D延续了有限的开发信号，但未达到完整门槛。如果继续做机制研究，应先留在同一小任务上，增加一个**不先混合原始value、但仍为rank1**的配平对照：

```
M_mean_factors = diag(m) + mean_s[w(v_s)] mean_s[u(v_s)]ᵀ
```

保留现有u/w定义，不对`mean(u)`再作隐式归一化。这能与已有pooled外积比较“先对原始value池化再生成因子”与“先逐槽生成因子再平均”的区别；与block的差恰为因子的跨槽协方差项：

```
mean_s[w_s u_sᵀ] - mean(w) mean(u)ᵀ
  = mean_s[(w_s-mean(w))(u_s-mean(u))ᵀ]
```

这样才比直接把block胜出叫作“rank3胜过rank1”更可判别。另可在只做机制诊断时，固定已选事实及各因子边缘集合，打乱u/w的行配对，测`mean_s[w_s u_perm(s)ᵀ]`的变化；该操作改变关联，不是现有实验结果，也不应用测试分数挑排列。

这些仍需新协议、配平预算和独立确认，不能在本轮追加后冒充预注册臂。本轮未证实先扩大Wikipedia会解决问题。当前更直接的问题是：非线性因子生成顺序、槽内关联与符号系数是否帮助完整内容组合，同时能否保留未改词和无关事实。只有在小任务上形成稳定证据，才讨论增加文本复杂度或规模。
