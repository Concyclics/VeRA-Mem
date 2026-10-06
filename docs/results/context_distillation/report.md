# Context teacher 蒸馏：已审计开发集结果

六组结果通过逐记录审计；五个训练组具有相同 target/episode schedule SHA，initial 为零更新。开发集 64 个事实，同一事实在四个条件复用。

## 自由生成与检索

单元格：real EM / prefill R@1。Oracle 是强制 value 的诊断，不是数学上界。

| 方法 | 规范支持/规范问句 | 规范支持/新问句 | 新支持/规范问句 | 新支持/新问句 |
|---|---|---|---|---|
| initial | 90.6% / 100.0% | 0.0% / 4.7% | 1.6% / 42.2% | 0.0% / 7.8% |
| ce | 92.2% / 100.0% | 4.7% / 4.7% | 0.0% / 34.4% | 0.0% / 7.8% |
| off_kd | 60.9% / 100.0% | 3.1% / 6.2% | 0.0% / 23.4% | 0.0% / 7.8% |
| off_kd_hidden | 48.4% / 96.9% | 4.7% / 4.7% | 0.0% / 34.4% | 0.0% / 7.8% |
| on_kd | 64.1% / 100.0% | 6.2% / 6.2% | 0.0% / 34.4% | 0.0% / 7.8% |
| on_kd_hidden | 62.5% / 100.0% | 4.7% / 4.7% | 0.0% / 29.7% | 0.0% / 6.2% |

## 记忆因果对照

real − shuffled，单位百分点及 95% fact-paired bootstrap CI；最后一列先按事实平均四个条件，再抽样事实。

| 方法 | 规范支持/规范问句 | 规范支持/新问句 | 新支持/规范问句 | 新支持/新问句 | 四条件 fact-cluster 平均 |
|---|---|---|---|---|---|
| initial | +84.4 [+75.0, +92.2] | +0.0 [+0.0, +0.0] | +0.0 [-4.7, +4.7] | +0.0 [+0.0, +0.0] | +21.1 [+18.8, +23.4] |
| ce | +81.2 [+70.3, +90.6] | +4.7 [+0.0, +10.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +21.5 [+18.4, +24.6] |
| off_kd | +50.0 [+35.9, +64.1] | -1.6 [-4.7, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +12.1 [+8.6, +15.6] |
| off_kd_hidden | +43.8 [+31.2, +54.7] | +1.6 [+0.0, +4.7] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +11.3 [+8.2, +14.5] |
| on_kd | +54.7 [+42.2, +67.2] | +6.2 [+1.6, +12.5] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +15.2 [+11.7, +18.8] |
| on_kd_hidden | +56.2 [+43.8, +68.8] | +4.7 [+0.0, +10.9] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +15.2 [+11.7, +18.8] |

## Teacher 验收与初始学生

Teacher 接收文本事实；student 接收问题与 VDB。此对照用于确认 teacher 是否提供有效监督。

| 条件 | Teacher EM | Initial real EM | Teacher − initial (pp, 95% CI) | Teacher answer NLL |
|---|---|---|---|---|
| 规范支持/规范问句 | 100.0% | 90.6% | +9.4 [+3.1, +17.2] | 0.0000 |
| 规范支持/新问句 | 100.0% | 0.0% | +100.0 [+100.0, +100.0] | 0.0000 |
| 新支持/规范问句 | 100.0% | 1.6% | +98.4 [+95.3, +100.0] | 0.0000 |
| 新支持/新问句 | 100.0% | 0.0% | +100.0 [+100.0, +100.0] | 0.0000 |

## Gold-prefix 对齐诊断

每项先按事实内部 token 平均，再按事实平均。KL 为 reverse KL；hidden 为 cosine。括号内是相同问题的 no-memory 基线；首 token 尚未输入 gold answer。

| 方法 | 条件 | KL (base) | 首 token KL (base) | Hidden cosine (base) | 首 token cosine (base) |
|---|---|---|---|---|---|
| initial | 规范支持/规范问句 | 2.9047 (23.8349) | 3.8056 (34.6873) | 0.6729 (0.5596) | 0.7077 (0.6034) |
| initial | 规范支持/新问句 | 19.2321 (20.4469) | 34.6726 (34.3462) | 0.6128 (0.6039) | 0.5819 (0.5802) |
| initial | 新支持/规范问句 | 19.9523 (23.5545) | 32.7011 (34.1643) | 0.6183 (0.5623) | 0.6698 (0.6121) |
| initial | 新支持/新问句 | 17.0687 (20.0441) | 33.0607 (33.5209) | 0.6443 (0.6049) | 0.6110 (0.5826) |
| ce | 规范支持/规范问句 | 3.1615 (23.8349) | 5.9589 (34.6873) | 0.6846 (0.5596) | 0.7113 (0.6034) |
| ce | 规范支持/新问句 | 16.2236 (20.4469) | 31.9497 (34.3462) | 0.6658 (0.6039) | 0.6486 (0.5802) |
| ce | 新支持/规范问句 | 19.4351 (23.5545) | 32.1716 (34.1643) | 0.6345 (0.5623) | 0.7095 (0.6121) |
| ce | 新支持/新问句 | 17.3656 (20.0441) | 33.0187 (33.5209) | 0.6534 (0.6049) | 0.6418 (0.5826) |
| off_kd | 规范支持/规范问句 | 7.2554 (23.8349) | 16.4076 (34.6873) | 0.7213 (0.5596) | 0.7492 (0.6034) |
| off_kd | 规范支持/新问句 | 14.9906 (20.4469) | 32.3003 (34.3462) | 0.7054 (0.6039) | 0.6720 (0.5802) |
| off_kd | 新支持/规范问句 | 20.9456 (23.5545) | 32.9523 (34.1643) | 0.6087 (0.5623) | 0.6839 (0.6121) |
| off_kd | 新支持/新问句 | 16.6685 (20.0441) | 32.2167 (33.5209) | 0.6714 (0.6049) | 0.6711 (0.5826) |
| off_kd_hidden | 规范支持/规范问句 | 8.0148 (23.8349) | 17.2067 (34.6873) | 0.7104 (0.5596) | 0.7363 (0.6034) |
| off_kd_hidden | 规范支持/新问句 | 14.5528 (20.4469) | 31.3755 (34.3462) | 0.7023 (0.6039) | 0.6723 (0.5802) |
| off_kd_hidden | 新支持/规范问句 | 20.2167 (23.5545) | 32.0652 (34.1643) | 0.6106 (0.5623) | 0.6881 (0.6121) |
| off_kd_hidden | 新支持/新问句 | 16.4994 (20.0441) | 31.5087 (33.5209) | 0.6718 (0.6049) | 0.6781 (0.5826) |
| on_kd | 规范支持/规范问句 | 7.8127 (23.8349) | 13.6852 (34.6873) | 0.6904 (0.5596) | 0.7384 (0.6034) |
| on_kd | 规范支持/新问句 | 16.5977 (20.4469) | 31.8209 (34.3462) | 0.6649 (0.6039) | 0.6530 (0.5802) |
| on_kd | 新支持/规范问句 | 22.5499 (23.5545) | 33.0869 (34.1643) | 0.5718 (0.5623) | 0.6748 (0.6121) |
| on_kd | 新支持/新问句 | 18.7414 (20.0441) | 32.4306 (33.5209) | 0.6445 (0.6049) | 0.6826 (0.5826) |
| on_kd_hidden | 规范支持/规范问句 | 6.9623 (23.8349) | 12.2752 (34.6873) | 0.6868 (0.5596) | 0.7302 (0.6034) |
| on_kd_hidden | 规范支持/新问句 | 17.0843 (20.4469) | 32.5107 (34.3462) | 0.6528 (0.6039) | 0.6370 (0.5802) |
| on_kd_hidden | 新支持/规范问句 | 22.5604 (23.5545) | 32.6101 (34.1643) | 0.5667 (0.5623) | 0.6685 (0.6121) |
| on_kd_hidden | 新支持/新问句 | 18.7063 (20.0441) | 31.6657 (33.5209) | 0.6437 (0.6049) | 0.6862 (0.5826) |

## 实际训练成本

相同步数不代表相同计算成本。Input totals 是未补齐的 student/teacher/rollout/replay 输入位置总数；不等于 FLOPs。

| 方法 | 更新 | Target exposures | Main targets | Teacher targets | Sampled tokens | Replay targets | 总 input tokens | 训练秒数 |
|---|---|---|---|---|---|---|---|---|
| initial | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 |
| ce | 256 | 2048 | 4858 | 0 | 0 | 1213 | 148745 | 26.4 |
| off_kd | 256 | 2048 | 4858 | 4858 | 0 | 1213 | 334809 | 38.6 |
| off_kd_hidden | 256 | 2048 | 4858 | 4858 | 0 | 1213 | 334809 | 32.4 |
| on_kd | 256 | 2048 | 6371 | 6371 | 6371 | 1213 | 701039 | 69.6 |
| on_kd_hidden | 256 | 2048 | 6473 | 6473 | 6473 | 1213 | 706798 | 69.0 |

## 范围与限制

- Exploratory development-set experiment: at most 64 facts, short finite-vocabulary answers, one training seed.
- No held-out confirmation claim, no checkpoint selection from development results, and no evidence of general continual-learning ability.
- Four conditions reuse facts; across-condition uncertainty bootstraps fact clusters, not individual fact/phase records.
- Gold+EOS prefix KL/hidden diagnostics use teacher forcing and are distinct from free-generation EM; first-token metrics expose the answer-free prefix.
- Cosine hidden alignment can be high for the no-memory backbone; compare changes against that baseline instead of absolute cosine alone.
- Forced-value oracle changes the readout distribution and is a diagnostic, not a guaranteed accuracy upper bound.
- Token cost totals count unpadded input positions and omit backward/FLOP costs; on-policy rollouts recompute prompt/prefix positions.
- Bootstrap intervals condition on this trained model and these template families; they do not estimate seed or unseen-family uncertainty.

summary.json 保留全部四象限的 real/oracle/shuffled/empty EM、NLL、R@1/R@4、decode 指标、成对差值及来源 SHA；原始提示、token ID、事实 ID 和本地路径均未发布。
