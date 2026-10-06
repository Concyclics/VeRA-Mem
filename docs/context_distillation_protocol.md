# 有上下文 teacher 到 VDB–VeRA student：训练方案与尝试协议

本协议在本轮模型运行前固定。目标是学习通用的观测写入器和稀疏读取器，让输入生成的 VeRA 参数向量复现原模型看到相关文本上下文后的行为；在线遇到新事实时，仍只写入向量，不更新共享权重。先用既有机制任务作探索性实验，不把开发集改善称为确认性的语义泛化。

## 文献依据和方法边界

| 原始文献 | 对本设计的支持 | 与本项目的区别 |
| --- | --- | --- |
| [Askell et al., 2021](https://arxiv.org/abs/2112.00861) | 用有提示上下文的模型分布训练无该上下文的模型，早期 context distillation | 知识进入模型权重；其附录也报告没有消除某些格式差异 |
| [Agarwal et al., GKD, ICLR 2024](https://arxiv.org/abs/2306.13649) | 在学生自身前缀上接受 teacher 分布监督；轨迹来源和 KL 方向是独立因素 | 不涉及我们的可写 VDB 和 VeRA；OPD 不等于 reverse KL |
| [Ye et al., OPCD, 2026](https://arxiv.org/abs/2602.12275) | teacher 看 context，student 不看；在相同 student 前缀上做 reverse KL；使用过 Qwen3-4B-Instruct-2507 | 未验证 hidden 蒸馏、稀疏检索或 VeRA；论文实验不能当作本架构已有结果 |
| [Romero et al., FitNets, ICLR 2015](https://arxiv.org/abs/1412.6550) | teacher 中间表示作为辅助训练目标 | 图像网络证据；过强表示约束可能伤害性能，不保证跨上下文的 LLM 状态应完全一致 |
| [Mu et al., Gisting, NeurIPS 2023](https://arxiv.org/abs/2304.08467) | 通用压缩器将新 context 一次编码为紧凑表示，无须每条 context 再训练 | gist tokens 是 attention 前缀，并非稀疏召回的参数向量 |
| [Ge et al., ICAE, ICLR 2024](https://arxiv.org/abs/2307.06945) | 联合学习 context 编码与冻结语言模型的读出兼容性 | 使用 memory slots 与重建/续写训练，不是本轮 OPD 目标 |
| [Chari et al., KV-Distill, 2025](https://arxiv.org/abs/2503.10337) | 用完整上下文的冻结模型输出监督压缩表示的输出 | KV 指 attention cache，不是 VDB；未验证 VeRA 的表示容量或寻址 |

因此“context-aware teacher”及“hidden hint”本身不是新概念。我们需要验证的是它们能否训练出可持续写入、可按层输入稀疏检索的参数化记忆接口；是否构成研究创新还需进一步查重和真实任务证据。

## 数据流与监督位置

```text
训练观测 c ──冻结基座提特征── Wk/Wv ──临时可微 memory bank
                                               │
问题 x + 学生前缀 y<t ──冻结基座── Wq→top-k→value→VeRA ── pS, hS
原文 c + 同一问题 x + 同一前缀 y<t ──冻结原始基座────────── pT, hT
```

Teacher 禁用全部 VeRA 增量并停止梯度，读取对应训练事实的完整观测。Student prompt 仅含问题，不拼接该原文；新观测通过 writer 进入向量库。两者共用同一冻结模型与 tokenizer，分别前向，避免无意义的词表对齐。Teacher 的上下文是训练期额外信息，评测时不把 teacher 或正确事实 ID 提供给 student。

每个输出位置比较各自 `prompt_length - 1 + t`，输入相同的 continuation token IDs，不按相同绝对序列下标对齐，不 decode 后重新 tokenize。第一轮 hidden 选 **最终 RMSNorm 后、lm_head 前**的状态，已位于第20层 VeRA 注入之后；不约束无法被该单层 adapter 改变的注入前9728维输入。仅约束答案预测位置，不逐位置强行对齐不同长度的上下文。

隐藏状态损失为 `1 - cosine(hS, stop_gradient(hT))`，权重0.1，无额外可学习投影。它只衡量表征接近，不能作为记忆准确性的替代指标。上下文长度、位置编码和表达格式都可能影响该目标。

## 固定的五臂设计

| 方法名 | 预测位置所见前缀 | 主输出损失 | hidden辅助 |
| --- | --- | --- | --- |
| ce | 正确答案前缀 | 答案 CE | 无 |
| off_kd | 正确答案前缀 | reverse KL(S‖T) | 无 |
| off_kd_hidden | 正确答案前缀 | 同上 | 0.1×cosine距离 |
| on_kd | 当前 student 采样前缀 | reverse KL(S‖T) | 无 |
| on_kd_hidden | 当前 student 采样前缀 | 同上 | 0.1×cosine距离 |

四个 KD 臂均为完整词表 KL，温度1；不将 KL 方向变化混入 on-policy 对照。采用 GKD 式固定采样前缀重算目标，不对离散采样求导，也不声称实现了包含轨迹分布梯度的完整策略梯度估计。Student sampling 温度1、最多4个新 token，保留实际 EOS；截断时不补人工 EOS。逐token批处理未结束的样本，右padding后按各自最后有效位置采样，不使用KV cache；与逐条实现具有相同条件采样分布，但随机数消费顺序不同。off-policy 使用答案 token 加 EOS。损失先在每个样本的 token 上平均，再对样本平均。

所有组同时使用：

```text
L = L_output + 0.2 L_all_style_address + 0.2 L_actual_token_address
    + [hidden arm] 0.1 L_hidden
    + [every fourth update] 0.25 L_canonical_replay_CE
```

每个 episode 包含目标事实与其他不同事实，所有8种 query×4种 support 的风格组合参与事实身份 CE。实际预测位置另用全候选 keys 做地址 CE，弥补 hard top-k 漏召回时输出损失不直接更新未选 key 的缺口。on-policy 出现错误前缀后仍以当前问题的事实作为该位置地址标签，这是明确的训练选择，并不等于模型可以撤销已输出的错误答案。

线上 CPU VDB 路径是 detach 的推理接口。采样时不保留梯度，训练重算时使用等价 top-k 的临时 GPU bank，keys/values 由当前编码器重新产生并保留计算图；不能直接向线上 detach 路径附加 KL 就宣称训练了 Q/K/value。

## 固定预算与数据范围

- 模型：Qwen3-4B-Instruct-2507，revision `cdbee75f17c01a7cc42f958dc650907174af0554`；第20层 `mlp.down_proj`，rank/key=64、top-k=4。
- 共同初始化：上一轮 `factcentric_probe_20261006/anchored_all_view.pt` 的最终 checkpoint；选择它是保留已有 value 坐标兼容性的预定初始化，不是本轮开发集选模。额外评估其未继续训练的结果。
- 使用既有缓存的4096训练实体、8问句/4观测表达。开发实体64个，只访问 train/dev 分支；不读取旧确认集来决定方法或选checkpoint。开发格式已在旧实验中分析过，本轮不能称为盲测。
- 每组固定256次更新、batch8，即2048次主目标事实曝光；每4步一次8事实canonical回放，额外512次回放曝光。统一seed42、事实与视图随机序列。
- Wq/Wk学习率1e-5，Wv为1e-4，共享b为0.005，分组梯度裁剪1。冻结基座、A/B和中心buffer。固定最后一步，不用开发集挑选步数、loss系数或臂。
- 相同更新和事实预算不等于相同 token/FLOPs。OPD多出采样，teacher多出context前向，生成长度也不同；分别记录真实token数、teacher/student前向成本与耗时。
- Smoke只用于检查接口与梯度，和正式结果分开保存；不使用其分数选择配方。

## 评估与停止规则

先评估有context的teacher在原/新问句×原/新观测四象限的生成效果，再评估共同初始化。最终student在相同四象限测试 real / 强制正确value / shuffled value / empty，保留逐题预测、R@1/R@4、NLL、EM、VDB和权重哈希。新观测格式按 `index % 2` 轮转，新问句格式按 `(index // 2) % 2` 轮转：64个实体中四种新问句×新观测组合各16个，避免问题风格与正确记录风格完全相关。不额外把每个表达视为独立实体。统一正确答案前缀上比较最终hidden cosine和reverse KL，以免各臂不同采样状态导致指标不可比。

所有评测只写入观测生成的向量，冻结共享权重。强制正确value是诊断，不是数学上的性能上界；其分布不同于top-k混合。必须同时观察 real−shuffled、real−empty；若只接近teacher hidden而没有事实敏感的生成提升，就不能认定记忆有效。

首轮是单seed、短预算、开发集、16词表的机制尝试。正确答案只有1–2个token，OPD特有轨迹收益可能主要影响第二子词/EOS；不能将其结果外推到长程推理。无论正负结果，完成既定矩阵后保留原始结果，不用同一开发集无限加预算挑选方法。

## 后续验证路线（首轮不混入）

1. 使用独立的新确认实体和结构表达族；多seed，匹配token预算并报告计算量。对已分析的旧XML/CSV/对话集合仅作历史对照。
2. 添加真实多token值或多字段事实回答，避免通过无关冗长解释人为制造OPD优势；分别报告首token、完整事实和多字段正确性。
3. 新事实写入、同实体事实更新、干扰记录和混合格式bank；观察写后保持及正确记忆被替换后答案是否随之变化。
4. 独立比较注入后block hidden、最终hidden、context-effect差分、不同hidden权重；差分仍可能含位置和格式影响，不能直接命名为纯语义向量。
5. 若teacher可靠但student寻址或读出仍失败，分开研究更强写入编码器、非线性Q/K、多层VeRA与容量变化，避免把架构收益归因于蒸馏。
