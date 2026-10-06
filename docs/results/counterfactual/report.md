# 反事实记忆蒸馏：严格审计结果

v2缓存预检通过：固定writer前缀已核验，历史16个训练事实×4种支持表达共64组特征复算的最小cosine=1.00000083，最大relative RMSE=0.00000000；固定门槛为cosine≥0.999、relative RMSE≤0.05。训练/确认缓存文件SHA与运行记录一致；准备过程teacher生成次数为0。v1无效结果不进入本报告，新确认实体seed为37042。

A/B为同一问题、不同目标事实；P只改目标观察的表达。主指标要求A、B两个世界同时正确，不能用输出发生变化代替成功。

共同warm 256步；四组各续训 512步，target/episode schedule SHA一致。Initial是warm前的原始anchored checkpoint。

## 四条件的真实读出

单元格：A/B成对正确率 / A与P同时正确率。每列64个事实。

| 方法 | 规范/规范 | 规范/新问句 | 新支持/规范 | 新支持/新问句 |
|---|---|---|---|---|
| initial | 76.6% / 76.6% | 0.0% / 0.0% | 1.6% / 1.6% | 0.0% / 0.0% |
| base | 89.1% / 85.9% | 0.0% / 0.0% | 0.0% / 1.6% | 0.0% / 0.0% |
| behavior | 93.8% / 87.5% | 0.0% / 0.0% | 0.0% / 1.6% | 0.0% / 0.0% |
| hidden | 90.6% / 85.9% | 0.0% / 0.0% | 0.0% / 1.6% | 0.0% / 0.0% |
| mixed | 92.2% / 82.8% | 0.0% / 0.0% | 0.0% / 1.6% | 0.0% / 0.0% |

## 检索、错误切换与打乱对照

R@1为A/B两世界；swapped指A答B且B答A。wrong/wrong包括两边都错但输出相同的情况；变化且两边都错要求normalize_answer后的输出确实不同。Δ为real−shuffled paired switch，百分点及95%按事实成对bootstrap区间。

| 方法 | 条件 | A/B R@1 | swapped | wrong/wrong | 变化且两边都错 | 变化但未同时正确 | Δreal−shuffled (pp, CI) |
|---|---|---|---|---|---|---|---|
| initial | 规范/规范 | 100.0%/100.0% | 0.0% | 3.1% | 3.1% | 23.4% | +76.6 [+65.6, +87.5] |
| initial | 规范/新问句 | 1.6%/1.6% | 0.0% | 100.0% | 1.6% | 1.6% | +0.0 [+0.0, +0.0] |
| initial | 新支持/规范 | 39.1%/48.4% | 0.0% | 93.8% | 10.9% | 14.1% | +1.6 [+0.0, +4.7] |
| initial | 新支持/新问句 | 1.6%/0.0% | 0.0% | 100.0% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| base | 规范/规范 | 100.0%/100.0% | 0.0% | 1.6% | 1.6% | 10.9% | +89.1 [+81.2, +95.4] |
| base | 规范/新问句 | 0.0%/1.6% | 0.0% | 100.0% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| base | 新支持/规范 | 35.9%/37.5% | 0.0% | 95.3% | 7.8% | 7.8% | +0.0 [+0.0, +0.0] |
| base | 新支持/新问句 | 1.6%/1.6% | 0.0% | 100.0% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| behavior | 规范/规范 | 100.0%/100.0% | 0.0% | 0.0% | 0.0% | 6.2% | +93.8 [+87.5, +98.4] |
| behavior | 规范/新问句 | 1.6%/1.6% | 0.0% | 100.0% | 1.6% | 1.6% | +0.0 [+0.0, +0.0] |
| behavior | 新支持/规范 | 46.9%/48.4% | 0.0% | 95.3% | 7.8% | 7.8% | +0.0 [+0.0, +0.0] |
| behavior | 新支持/新问句 | 1.6%/1.6% | 0.0% | 100.0% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| hidden | 规范/规范 | 100.0%/100.0% | 0.0% | 0.0% | 0.0% | 9.4% | +90.6 [+82.8, +96.9] |
| hidden | 规范/新问句 | 0.0%/1.6% | 0.0% | 100.0% | 3.1% | 3.1% | +0.0 [+0.0, +0.0] |
| hidden | 新支持/规范 | 40.6%/35.9% | 0.0% | 98.4% | 17.2% | 17.2% | +0.0 [+0.0, +0.0] |
| hidden | 新支持/新问句 | 1.6%/1.6% | 0.0% | 100.0% | 0.0% | 0.0% | +0.0 [+0.0, +0.0] |
| mixed | 规范/规范 | 100.0%/100.0% | 0.0% | 0.0% | 0.0% | 7.8% | +92.2 [+84.4, +98.4] |
| mixed | 规范/新问句 | 0.0%/1.6% | 0.0% | 100.0% | 1.6% | 1.6% | +0.0 [+0.0, +0.0] |
| mixed | 新支持/规范 | 43.8%/43.8% | 0.0% | 95.3% | 7.8% | 7.8% | +0.0 [+0.0, +0.0] |
| mixed | 新支持/新问句 | 1.6%/1.6% | 0.0% | 100.0% | 1.6% | 1.6% | +0.0 [+0.0, +0.0] |

## Teacher可用性

所有事实仍在主表分母中。条件分数仅作诊断；teacher正确性是自由生成结果，不等同训练loss的margin/hidden门控。

| 条件 | Teacher A/B joint | Teacher A/P joint | Teacher A/B/P全对数量 |
|---|---|---|---|
| 规范/规范 | 100.0% | 100.0% | 64/64 |
| 规范/新问句 | 0.0% | 0.0% | 0/64 |
| 新支持/规范 | 100.0% | 100.0% | 64/64 |
| 新支持/新问句 | 1.6% | 0.0% | 0/64 |

| 方法 | 条件 | 全部事实A/B joint | Teacher A/B均正确子集 | 子集分母 | A/P joint(全部/teacher合格) |
|---|---|---|---|---|---|
| initial | 规范/规范 | 76.6% | 76.6% | 64 | 76.6%/76.6% |
| initial | 规范/新问句 | 0.0% | — | 0 | 0.0%/— |
| initial | 新支持/规范 | 1.6% | 1.6% | 64 | 1.6%/1.6% |
| initial | 新支持/新问句 | 0.0% | 0.0% | 1 | 0.0%/— |
| base | 规范/规范 | 89.1% | 89.1% | 64 | 85.9%/85.9% |
| base | 规范/新问句 | 0.0% | — | 0 | 0.0%/— |
| base | 新支持/规范 | 0.0% | 0.0% | 64 | 1.6%/1.6% |
| base | 新支持/新问句 | 0.0% | 0.0% | 1 | 0.0%/— |
| behavior | 规范/规范 | 93.8% | 93.8% | 64 | 87.5%/87.5% |
| behavior | 规范/新问句 | 0.0% | — | 0 | 0.0%/— |
| behavior | 新支持/规范 | 0.0% | 0.0% | 64 | 1.6%/1.6% |
| behavior | 新支持/新问句 | 0.0% | 0.0% | 1 | 0.0%/— |
| hidden | 规范/规范 | 90.6% | 90.6% | 64 | 85.9%/85.9% |
| hidden | 规范/新问句 | 0.0% | — | 0 | 0.0%/— |
| hidden | 新支持/规范 | 0.0% | 0.0% | 64 | 1.6%/1.6% |
| hidden | 新支持/新问句 | 0.0% | 0.0% | 1 | 0.0%/— |
| mixed | 规范/规范 | 92.2% | 92.2% | 64 | 82.8%/82.8% |
| mixed | 规范/新问句 | 0.0% | — | 0 | 0.0%/— |
| mixed | 新支持/规范 | 0.0% | 0.0% | 64 | 1.6%/1.6% |
| mixed | 新支持/新问句 | 0.0% | 0.0% | 1 | 0.0%/— |

## 跨条件成对比较

每个实体的四个条件一起重采样，不能把256行当作256个独立事实。以下为相对base的百分点差及95%区间。

| 方法 | A/B joint | A/P joint | swapped | wrong/wrong | 无关实体joint(两个条件) |
|---|---|---|---|---|---|
| base | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] | +0.0 [+0.0, +0.0] |
| behavior | +1.2 [-0.4, +2.7] | +0.4 [-1.2, +2.0] | +0.0 [+0.0, +0.0] | -0.4 [-1.2, +0.0] | +1.6 [+0.0, +3.9] |
| hidden | +0.4 [-1.2, +2.0] | +0.0 [-2.0, +2.0] | +0.0 [+0.0, +0.0] | +0.4 [-0.8, +1.6] | +0.0 [-3.1, +3.1] |
| mixed | +0.8 [-0.8, +2.7] | -0.8 [-2.3, +0.8] | +0.0 [+0.0, +0.0] | -0.4 [-1.2, +0.0] | +0.8 [-1.6, +3.1] |

## 实际训练成本与teacher训练覆盖率

| 阶段 | 更新 | Pair exposures | sampled tokens | 总input tokens | 训练秒数 | Teacher首token A/B全对 | Behavior有效 | Hidden有效 |
|---|---|---|---|---|---|---|---|---|
| 共同warm | 256 | 2048 | 0 | 946208 | 81.3 | 94.8% | 100.0% | 100.0% |
| base | 512 | 4096 | 0 | 1892732 | 156.9 | 94.9% | 100.0% | 100.0% |
| behavior | 512 | 4096 | 0 | 1892732 | 153.0 | 94.9% | 100.0% | 100.0% |
| hidden | 512 | 4096 | 0 | 1892732 | 158.2 | 94.9% | 100.0% | 100.0% |
| mixed | 512 | 4096 | 2676 | 2046139 | 222.6 | 94.9% | 100.0% | 100.0% |

共同warm在物理成本总量中只计一次；每个续训方法的完整训练还需加上该共同warm。初始checkpoint的历史训练和数据缓存准备成本不计入此次成本。

## 范围与限制

- 64 fresh entities, a finite 16-word answer vocabulary, two reserved structure families and one training seed; broader semantic generalization is untested.
- Only v2 caches with the historical writer prefix and a passing 64-feature training-only reproduction check are admitted. Invalid v1 results are excluded; confirmation uses fresh seed37042.
- A/B must both be correct for paired_switch_em. Changed predictions alone, swapped wrong answers, and wrong/wrong changes are not successes.
- Teacher eligibility conditions are reported alongside all-fact scores; no difficult teacher-ineligible facts are silently discarded.
- A/B/P and all four phases share an entity. Bootstrap samples target-fact clusters, not individual world/phase rows; intervals exclude seed/family uncertainty.
- Unrelated reads query the next entity under one target intervention; this limited preservation diagnostic is not a general forgetting benchmark.
- The actual writer may change the target key and value together. These are single-record observation interventions, not fixed-key value-only interventions.
- Oracle forces a value and changes the readout distribution; it is a diagnostic rather than a mathematical upper bound.
- Equal continuation updates/target schedules do not imply equal compute; mixed rollouts add work. Token totals are unpadded positions, not FLOPs.
- Physical training cost counts the common warm-up once and excludes training the inherited anchored initialization and cache preparation.
- This audit verifies prediction/write records, reconstructs CPU bank and shuffle hashes, and checks source/checkpoint/cache lineage; it does not replay model inference.

summary.json包含所有real/shuffled/oracle/empty、teacher、paraphrase、unrelated指标、成对区间、日志成本与SHA。原始问题、观察、token ID、实体ID和本地路径不进入公开输出。
