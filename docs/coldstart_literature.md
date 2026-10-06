# 冷启动、基础字典与稀疏优化：文献映射

2026-10-06 核对。以下依据原论文与作者仓库；详细节号、训练量和推导见
[证据记录](results/coldstart/literature.json)。本轮实际执行范围以
[固定协议](coldstart_protocol.md) 为准，证据记录中的早期建议不等于已实施实验。

## 最接近的证据

| 原始来源 | 已验证的机制与实际训练量 | 对本实验的启示与边界 |
| --- | --- | --- |
| [Large Memory Layers with Product Keys](https://arxiv.org/html/1907.05242)，§3.1/3.3、§4.1/4.3/4.5；[作者代码](https://github.com/facebookresearch/XLM) | 从 Common Crawl 新闻的 **280 亿 words、140GB** 训练乘积 key/value 记忆；随机 key、query BatchNorm、较高的稀疏 value 学习率。不可把 words 改称 tokenizer tokens。 | 支持学习共享基础字典并监测使用率。它没有保证新事实能够单次写入、独立覆盖，也没有验证冻结 4B 模型的一层 rank-64 VeRA。 |
| [Large Product Key Memory for Pretrained Language Models](https://aclanthology.org/2020.findings-emnlp.362/)，§3–4、§5.2/6；[作者代码](https://github.com/clovaai/pkm-transformers) | BERT MLM 使用 **English Wikipedia + BookCorpus 17GB，50 万步、batch 1024**。用已有模型初始化，并以 ResM 保留原 FFN、增加记忆残差；仍是稀疏 value 梯度。 | 最直接支持研究初始化与槽使用退化。文中 top-32 使用率可近 100%，同时 top-1 约 2%；只看 top-k 覆盖会掩盖集中使用。其 backbone 联合训练，不能据此认定冻结 Qwen 的失败就是论文所述漂移。 |
| [Memory Layers at Scale](https://arxiv.org/html/2412.09764v2)，§3/4/5.4；[作者代码](https://github.com/facebookresearch/memory) | Llama 类模型从头预训练，最高 **1T tokens**；base 134M–8B、memory 最高 128B 参数。共享多层 values，增加输入门、输出投影与稳定性处理。 | 支持“训练有用的基础字典”，也提示多层读出与容量的重要性。它不是少量 Wikipedia 热身，更不是自动生成、逐记录编辑的 episodic VDB。 |
| [SHINE](https://arxiv.org/html/2602.06358)，§3.3–3.4、§4.1–4.2；[作者代码](https://github.com/MuLabPKU/SHINE) | 冻结 Qwen3-8B，通过跨层 memory tokens 与 Transformer 生成 LoRA。先在 **TransMLA 6B-token corpus 上 1 epoch** 重构/补全，再做 QA；WikiText-2 是评测，不能说预训练来自 6B Wikipedia tokens。 | 支持先训练“文本→参数→行为”的写读接口。其跨层多 token、超网络容量与本轮单层 pooled rank-64 value 不同；没有稀疏 VDB，因此不是稀疏槽优化的直接证据。 |
| [Token Merging: Your ViT But Faster](https://arxiv.org/html/2210.09461)，§3–4；[作者代码](https://github.com/facebookresearch/ToMe) | 在视觉、视频、音频 Transformer 上按 key 相似度匹配，以 token size 加权合并并做 proportional attention；不是 Wikipedia 语言记忆训练。 | 支持谨慎保留合并的质量与计数信息。不能证明相似 key 的不同事实可以安全平均，也不能证明聚类会修复长期参数槽的稀疏梯度。 |

PKM 与 Memory Layers 的基础槽是独立可训练参数；本方案还有由共享 writer
前向生成的 episodic 记录。一次 writer 更新会影响未来所有记录的生成值，因此
“某条缓存向量没被选中”与“它有一个永远不更新的独立 embedding 参数”并不等价。
本轮将共享基础字典和可独立替换的新事实分开，避免混淆两种记忆。

## 本轮采用的可检验调整

数据控制先比较无梯度热身、同答案的紧凑观测热身、Wikipedia 自然段落热身。
基础字典再比较固定、普通稀疏训练、前密后疏和 straight-through；`wiki_joint`
额外让字典本身参与 Wikipedia 热身。后者同时改变热身路由和基础记忆使用，
不能当作单因素对照。

straight-through 用于**混合后的输出**。若
`y_s = p_sparse @ V`、`y_d = p_dense @ V`，采用
`y_d + stop_gradient(y_s - y_d)`，前向是稀疏输出，反向走 dense Jacobian。
仅令权重 `p_ST = p_dense + stop_gradient(p_sparse - p_dense)` 再乘 `V`，
value 梯度仍由前向稀疏权重决定，不能声称所有 value 都获得梯度。
这个输出级方法是本实验的有偏替代梯度诊断，**不是上述论文共同验证的配方**；
dense backward 有真实计算成本，微小非零梯度也不等于有效学习。

基础 prototypes 的压缩只看已训练的 checkpoint，不使用确认集内容。
在同一个预声明 checkpoint 上比较 full 1024、off、random 256、cluster 256；
cluster 保留算术均值，不额外做 RMS 重标定或 log-count 修正。
这些对照同时改变容量、top-k 竞争和权重质量，不能把分母减小后的覆盖率升高
解释成训练或推理改进。新事实记录保持独立，不与互相冲突的事实平均。

## 结论应保留的限制

本轮不是 Wikipedia 规模预训练：热身仅 1024 步 × 8 targets，必须报告实际
唯一 target、bank 曝光和 token 数；32,768 条候选语料不能写成全部都接受过目标监督。
训练只有 canonical view，P=A，因此 AP 项不提供独立改写增强。
所有主比较共享一个新初始化种子，Qwen backbone 冻结，只在第 20 层使用一组
固定 rank-64 随机投影；增加字典槽不能扩展这组读出投影的输出子空间。

应同时查看教师 contextual A/B 资格、真实新事实的完整 paired EM/NLL、检索召回、
top-1/top-k 槽覆盖和梯度覆盖。若覆盖改善而 HC 的真实读出不变，只能说明优化路径
或槽使用改变，不能说泛化问题已解决。若所有对照均失败，也不能据此否定更大预算、
更强 writer、多层 memory 或更高容量读出的可能性。当前组合和效率优势都仍需实证，
不应宣称单次写入、稀疏记忆或文本生成参数本身是首创。
