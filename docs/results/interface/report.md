# VeRA-Mem 接口实验审计

已审计 53 个完成任务；0 个未完成/未启动任务未计入成绩。

主指标为同一事实 A/B 两世界均正确。所有表格由逐条预测重算；仅输出变化不算成功。

## 正式实验

### Step 1：四银行

开发集诊断；模板名中的 confirmation 不代表确认集成绩。

| Support style | Bank | Paired |
|---|---|---|
| cf_confirmation_support_json | KcVc | 53/64 |
| cf_confirmation_support_json | KhVc | 2/64 |
| cf_confirmation_support_json | KcVh | 0/64 |
| cf_confirmation_support_json | KhVh | 0/64 |
| cf_confirmation_support_json | teacher | 64/64 |
| cf_confirmation_support_markdown | KcVc | 53/64 |
| cf_confirmation_support_markdown | KhVc | 1/64 |
| cf_confirmation_support_markdown | KcVh | 1/64 |
| cf_confirmation_support_markdown | KhVh | 0/64 |
| cf_confirmation_support_markdown | teacher | 64/64 |

固定首位置路由一致性检查：512 次。后续自由生成前缀可分叉。


### Step 2：Writer

| Run | Split | Seed | Phase | Real paired | A/B R@1 | Decode A/B | Shuffled | Empty | Teacher 合格/总数 |
|---|---|---:|---|---:|---|---|---:|---:|---|
| baseline_confirm | confirm | 42 | CC | 112/128 (87.5%) | 99.2%/100.0% | 76.0%/78.0% | 0.0% | 0.0% | 128/128 |
| baseline_confirm | confirm | 42 | CH | 0/128 (0.0%) | 3.9%/0.8% | 4.3%/3.7% | — | — | 128/128 |
| baseline_confirm | confirm | 42 | HC | 0/128 (0.0%) | 38.3%/38.3% | 6.2%/6.1% | 0.0% | 0.0% | 128/128 |
| baseline_confirm | confirm | 42 | HH | 0/128 (0.0%) | 1.6%/0.8% | 4.1%/3.8% | — | — | 128/128 |
| mean_42_confirm | confirm | 42 | CC | 105/128 (82.0%) | 99.2%/100.0% | 73.5%/76.5% | 0.0% | 0.0% | 128/128 |
| mean_42_confirm | confirm | 42 | CH | 0/128 (0.0%) | 3.9%/0.8% | 4.4%/3.9% | — | — | 128/128 |
| mean_42_confirm | confirm | 42 | HC | 0/128 (0.0%) | 38.3%/38.3% | 4.3%/5.0% | 0.0% | 0.0% | 128/128 |
| mean_42_confirm | confirm | 42 | HH | 0/128 (0.0%) | 1.6%/0.8% | 3.8%/3.1% | — | — | 128/128 |
| mean_43_confirm | confirm | 43 | CC | 112/128 (87.5%) | 99.2%/100.0% | 74.2%/76.8% | 0.0% | 0.0% | 128/128 |
| mean_43_confirm | confirm | 43 | CH | 0/128 (0.0%) | 3.9%/0.8% | 4.4%/3.9% | — | — | 128/128 |
| mean_43_confirm | confirm | 43 | HC | 0/128 (0.0%) | 38.3%/38.3% | 4.9%/5.4% | 0.0% | 0.0% | 128/128 |
| mean_43_confirm | confirm | 43 | HH | 0/128 (0.0%) | 1.6%/0.8% | 3.6%/3.0% | — | — | 128/128 |
| mean_44_confirm | confirm | 44 | CC | 109/128 (85.2%) | 99.2%/100.0% | 74.7%/76.4% | 0.0% | 0.0% | 128/128 |
| mean_44_confirm | confirm | 44 | CH | 0/128 (0.0%) | 3.9%/0.8% | 4.5%/3.9% | — | — | 128/128 |
| mean_44_confirm | confirm | 44 | HC | 0/128 (0.0%) | 38.3%/38.3% | 4.1%/4.5% | 0.0% | 0.0% | 128/128 |
| mean_44_confirm | confirm | 44 | HH | 0/128 (0.0%) | 1.6%/0.8% | 3.6%/3.0% | — | — | 128/128 |
| baseline_dev | dev | 42 | CC | 23/32 (71.9%) | 100.0%/96.9% | 77.8%/64.6% | 0.0% | 0.0% | 32/32 |
| baseline_dev | dev | 42 | CH | 0/32 (0.0%) | 0.0%/3.1% | 5.4%/5.5% | — | — | 32/32 |
| baseline_dev | dev | 42 | HC | 0/32 (0.0%) | 31.2%/37.5% | 2.2%/6.0% | 0.0% | 0.0% | 32/32 |
| baseline_dev | dev | 42 | HH | 0/32 (0.0%) | 9.4%/0.0% | 0.0%/0.0% | — | — | 32/32 |
| last_42_dev | dev | 42 | CC | 22/32 (68.8%) | 100.0%/96.9% | 77.8%/68.1% | 0.0% | 0.0% | 32/32 |
| last_42_dev | dev | 42 | CH | 0/32 (0.0%) | 0.0%/3.1% | 5.2%/3.4% | — | — | 32/32 |
| last_42_dev | dev | 42 | HC | 0/32 (0.0%) | 31.2%/37.5% | 3.2%/5.8% | 0.0% | 0.0% | 32/32 |
| last_42_dev | dev | 42 | HH | 0/32 (0.0%) | 9.4%/0.0% | 1.0%/1.1% | — | — | 32/32 |
| mean_42_dev | dev | 42 | CC | 24/32 (75.0%) | 100.0%/96.9% | 64.8%/69.6% | 0.0% | 0.0% | 32/32 |
| mean_42_dev | dev | 42 | CH | 0/32 (0.0%) | 0.0%/3.1% | 7.0%/3.6% | — | — | 32/32 |
| mean_42_dev | dev | 42 | HC | 0/32 (0.0%) | 31.2%/37.5% | 3.8%/5.7% | 0.0% | 0.0% | 32/32 |
| mean_42_dev | dev | 42 | HH | 0/32 (0.0%) | 9.4%/0.0% | 1.8%/1.8% | — | — | 32/32 |
| mlp_42_dev | dev | 42 | CC | 24/32 (75.0%) | 100.0%/96.9% | 80.0%/64.6% | 0.0% | 0.0% | 32/32 |
| mlp_42_dev | dev | 42 | CH | 0/32 (0.0%) | 0.0%/3.1% | 5.3%/3.5% | — | — | 32/32 |
| mlp_42_dev | dev | 42 | HC | 0/32 (0.0%) | 31.2%/37.5% | 3.5%/5.8% | 0.0% | 0.0% | 32/32 |
| mlp_42_dev | dev | 42 | HH | 0/32 (0.0%) | 9.4%/0.0% | 1.2%/2.4% | — | — | 32/32 |
| last_43_dev | dev | 43 | CC | 22/32 (68.8%) | 100.0%/96.9% | 77.8%/64.6% | 0.0% | 0.0% | 32/32 |
| last_43_dev | dev | 43 | CH | 0/32 (0.0%) | 0.0%/3.1% | 7.1%/3.5% | — | — | 32/32 |
| last_43_dev | dev | 43 | HC | 0/32 (0.0%) | 31.2%/37.5% | 4.1%/5.9% | 0.0% | 0.0% | 32/32 |
| last_43_dev | dev | 43 | HH | 0/32 (0.0%) | 9.4%/0.0% | 1.4%/1.4% | — | — | 32/32 |
| mean_43_dev | dev | 43 | CC | 24/32 (75.0%) | 100.0%/96.9% | 64.8%/69.6% | 0.0% | 0.0% | 32/32 |
| mean_43_dev | dev | 43 | CH | 0/32 (0.0%) | 0.0%/3.1% | 8.8%/5.4% | — | — | 32/32 |
| mean_43_dev | dev | 43 | HC | 0/32 (0.0%) | 31.2%/37.5% | 2.5%/4.0% | 0.0% | 0.0% | 32/32 |
| mean_43_dev | dev | 43 | HH | 0/32 (0.0%) | 9.4%/0.0% | 1.7%/1.8% | — | — | 32/32 |
| mlp_43_dev | dev | 43 | CC | 24/32 (75.0%) | 100.0%/96.9% | 80.0%/64.6% | 0.0% | 0.0% | 32/32 |
| mlp_43_dev | dev | 43 | CH | 0/32 (0.0%) | 0.0%/3.1% | 5.3%/3.5% | — | — | 32/32 |
| mlp_43_dev | dev | 43 | HC | 0/32 (0.0%) | 31.2%/37.5% | 3.4%/6.1% | 0.0% | 0.0% | 32/32 |
| mlp_43_dev | dev | 43 | HH | 0/32 (0.0%) | 9.4%/0.0% | 3.6%/3.3% | — | — | 32/32 |
| last_44_dev | dev | 44 | CC | 22/32 (68.8%) | 100.0%/96.9% | 77.8%/66.7% | 0.0% | 0.0% | 32/32 |
| last_44_dev | dev | 44 | CH | 0/32 (0.0%) | 0.0%/3.1% | 7.3%/7.4% | — | — | 32/32 |
| last_44_dev | dev | 44 | HC | 0/32 (0.0%) | 31.2%/37.5% | 4.2%/6.4% | 0.0% | 0.0% | 32/32 |
| last_44_dev | dev | 44 | HH | 0/32 (0.0%) | 9.4%/0.0% | 1.4%/1.4% | — | — | 32/32 |
| mean_44_dev | dev | 44 | CC | 24/32 (75.0%) | 100.0%/96.9% | 63.0%/68.9% | 0.0% | 0.0% | 32/32 |
| mean_44_dev | dev | 44 | CH | 0/32 (0.0%) | 0.0%/3.1% | 3.8%/1.8% | — | — | 32/32 |
| mean_44_dev | dev | 44 | HC | 0/32 (0.0%) | 31.2%/37.5% | 1.5%/3.9% | 0.0% | 0.0% | 32/32 |
| mean_44_dev | dev | 44 | HH | 0/32 (0.0%) | 9.4%/0.0% | 3.2%/1.6% | — | — | 32/32 |
| mlp_44_dev | dev | 44 | CC | 24/32 (75.0%) | 100.0%/96.9% | 80.0%/64.6% | 0.0% | 0.0% | 32/32 |
| mlp_44_dev | dev | 44 | CH | 0/32 (0.0%) | 0.0%/3.1% | 5.4%/3.5% | — | — | 32/32 |
| mlp_44_dev | dev | 44 | HC | 0/32 (0.0%) | 31.2%/37.5% | 4.2%/6.6% | 0.0% | 0.0% | 32/32 |
| mlp_44_dev | dev | 44 | HH | 0/32 (0.0%) | 9.4%/0.0% | 2.1%/2.1% | — | — | 32/32 |

Teacher 合格子集的各方法分子/分母、canonical-key 诊断、成本与全部逐条配对向量见 summary.json。

- last_42：seed 42，新增 2000 步，起始步 0，可训练参数 622592；训练 5.1 秒。
- mean_42：seed 42，新增 2000 步，起始步 0，可训练参数 622592；训练 5.3 秒。
- mlp_42：seed 42，新增 2000 步，起始步 0，可训练参数 3129344；训练 5.7 秒。
- last_43：seed 43，新增 2000 步，起始步 0，可训练参数 622592；训练 5.3 秒。
- mean_43：seed 43，新增 2000 步，起始步 0，可训练参数 622592；训练 5.3 秒。
- mlp_43：seed 43，新增 2000 步，起始步 0，可训练参数 3129344；训练 5.8 秒。
- last_44：seed 44，新增 2000 步，起始步 0，可训练参数 622592；训练 5.0 秒。
- mean_44：seed 44，新增 2000 步，起始步 0，可训练参数 622592；训练 5.3 秒。
- mlp_44：seed 44，新增 2000 步，起始步 0，可训练参数 3129344；训练 5.5 秒。

### Step 3：寻址/读出

| Run | Split | Seed | Phase | Real paired | A/B R@1 | Decode A/B | Shuffled | Empty | Teacher 合格/总数 |
|---|---|---:|---|---:|---|---|---:|---:|---|
| coupled_42_confirm | confirm | 42 | CC | 0/256 (0.0%) | 19.9%/19.1% | 3.0%/2.7% | 0.0% | 0.0% | 256/256 |
| coupled_42_confirm | confirm | 42 | CH | 0/256 (0.0%) | 2.0%/0.8% | 2.6%/2.2% | — | — | 256/256 |
| coupled_42_confirm | confirm | 42 | HC | 0/256 (0.0%) | 5.5%/7.4% | 2.3%/2.4% | 0.0% | 0.0% | 256/256 |
| coupled_42_confirm | confirm | 42 | HH | 0/256 (0.0%) | 0.8%/0.8% | 1.2%/1.3% | — | — | 256/256 |
| decoupled_42_confirm | confirm | 42 | CC | 0/256 (0.0%) | 23.8%/19.5% | 2.9%/2.1% | 0.0% | 0.0% | 256/256 |
| decoupled_42_confirm | confirm | 42 | CH | 0/256 (0.0%) | 2.0%/1.6% | 2.6%/2.5% | — | — | 256/256 |
| decoupled_42_confirm | confirm | 42 | HC | 0/256 (0.0%) | 3.9%/6.2% | 2.9%/2.5% | 0.0% | 0.0% | 256/256 |
| decoupled_42_confirm | confirm | 42 | HH | 0/256 (0.0%) | 1.2%/0.8% | 1.2%/1.4% | — | — | 256/256 |
| coupled_43_confirm | confirm | 43 | CC | 0/256 (0.0%) | 18.8%/18.0% | 4.0%/4.0% | 0.0% | 0.0% | 256/256 |
| coupled_43_confirm | confirm | 43 | CH | 0/256 (0.0%) | 2.0%/1.2% | 3.6%/2.8% | — | — | 256/256 |
| coupled_43_confirm | confirm | 43 | HC | 0/256 (0.0%) | 5.1%/2.3% | 2.4%/2.6% | 0.0% | 0.0% | 256/256 |
| coupled_43_confirm | confirm | 43 | HH | 0/256 (0.0%) | 2.0%/2.0% | 1.9%/1.8% | — | — | 256/256 |
| decoupled_43_confirm | confirm | 43 | CC | 0/256 (0.0%) | 23.4%/21.9% | 3.4%/2.7% | 0.0% | 0.0% | 256/256 |
| decoupled_43_confirm | confirm | 43 | CH | 0/256 (0.0%) | 2.3%/1.2% | 3.3%/2.5% | — | — | 256/256 |
| decoupled_43_confirm | confirm | 43 | HC | 0/256 (0.0%) | 3.1%/3.5% | 2.1%/2.5% | 0.0% | 0.0% | 256/256 |
| decoupled_43_confirm | confirm | 43 | HH | 0/256 (0.0%) | 0.8%/2.0% | 1.8%/1.8% | — | — | 256/256 |
| coupled_44_confirm | confirm | 44 | CC | 0/256 (0.0%) | 27.7%/30.1% | 4.1%/3.8% | 0.0% | 0.0% | 256/256 |
| coupled_44_confirm | confirm | 44 | CH | 0/256 (0.0%) | 2.3%/2.0% | 2.5%/2.5% | — | — | 256/256 |
| coupled_44_confirm | confirm | 44 | HC | 0/256 (0.0%) | 5.1%/4.7% | 2.1%/2.1% | 0.0% | 0.0% | 256/256 |
| coupled_44_confirm | confirm | 44 | HH | 0/256 (0.0%) | 1.2%/1.6% | 1.5%/0.7% | — | — | 256/256 |
| decoupled_44_confirm | confirm | 44 | CC | 0/256 (0.0%) | 19.9%/23.8% | 4.2%/4.9% | 0.0% | 0.0% | 256/256 |
| decoupled_44_confirm | confirm | 44 | CH | 0/256 (0.0%) | 2.3%/2.0% | 2.3%/2.5% | — | — | 256/256 |
| decoupled_44_confirm | confirm | 44 | HC | 0/256 (0.0%) | 5.5%/4.7% | 2.2%/2.0% | 0.0% | 0.0% | 256/256 |
| decoupled_44_confirm | confirm | 44 | HH | 0/256 (0.0%) | 1.2%/1.6% | 1.7%/1.2% | — | — | 256/256 |

Teacher 合格子集的各方法分子/分母、canonical-key 诊断、成本与全部逐条配对向量见 summary.json。

- coupled_42：seed 42，新增 512 步，起始步 0，可训练参数 1870336；训练 161.7 秒。
- decoupled_42：seed 42，新增 512 步，起始步 0，可训练参数 1870336；训练 159.0 秒。
- coupled_43：seed 43，新增 512 步，起始步 0，可训练参数 1870336；训练 165.0 秒。
- decoupled_43：seed 43，新增 512 步，起始步 0，可训练参数 1870336；训练 157.5 秒。
- coupled_44：seed 44，新增 512 步，起始步 0，可训练参数 1870336；训练 163.5 秒。
- decoupled_44：seed 44，新增 512 步，起始步 0，可训练参数 1870336；训练 155.8 秒。

### Step 4：KD

| Run | Split | Seed | Phase | Real paired | A/B R@1 | Decode A/B | Shuffled | Empty | Teacher 合格/总数 |
|---|---|---:|---|---:|---|---|---:|---:|---|
| base_42_confirm | confirm | 42 | CC | 0/256 (0.0%) | 35.9%/35.9% | 5.4%/4.0% | 0.0% | 0.0% | 256/256 |
| base_42_confirm | confirm | 42 | CH | 0/256 (0.0%) | 1.6%/0.8% | 2.6%/1.9% | — | — | 256/256 |
| base_42_confirm | confirm | 42 | HC | 0/256 (0.0%) | 6.2%/8.2% | 2.5%/2.6% | 0.0% | 0.0% | 256/256 |
| base_42_confirm | confirm | 42 | HH | 0/256 (0.0%) | 2.3%/1.6% | 1.4%/1.3% | — | — | 256/256 |
| hidden_42_confirm | confirm | 42 | CC | 0/256 (0.0%) | 26.2%/27.0% | 4.3%/3.6% | 0.0% | 0.0% | 256/256 |
| hidden_42_confirm | confirm | 42 | CH | 0/256 (0.0%) | 1.6%/0.8% | 2.4%/2.4% | — | — | 256/256 |
| hidden_42_confirm | confirm | 42 | HC | 0/256 (0.0%) | 3.5%/3.9% | 2.3%/2.5% | 0.0% | 0.0% | 256/256 |
| hidden_42_confirm | confirm | 42 | HH | 0/256 (0.0%) | 2.0%/2.0% | 1.5%/1.3% | — | — | 256/256 |
| clip_42_confirm | confirm | 42 | CC | 0/256 (0.0%) | 30.9%/29.3% | 4.8%/3.4% | 0.0% | 0.0% | 256/256 |
| clip_42_confirm | confirm | 42 | CH | 0/256 (0.0%) | 1.6%/0.8% | 2.6%/2.6% | — | — | 256/256 |
| clip_42_confirm | confirm | 42 | HC | 0/256 (0.0%) | 3.9%/3.9% | 2.3%/2.3% | 0.0% | 0.0% | 256/256 |
| clip_42_confirm | confirm | 42 | HH | 0/256 (0.0%) | 2.0%/1.6% | 1.3%/1.2% | — | — | 256/256 |
| on_policy_42_confirm | confirm | 42 | CC | 0/256 (0.0%) | 34.8%/37.1% | 4.1%/4.3% | 0.0% | 0.0% | 256/256 |
| on_policy_42_confirm | confirm | 42 | CH | 0/256 (0.0%) | 1.2%/0.4% | 3.3%/2.8% | — | — | 256/256 |
| on_policy_42_confirm | confirm | 42 | HC | 0/256 (0.0%) | 5.9%/5.1% | 2.1%/1.6% | 0.0% | 0.0% | 256/256 |
| on_policy_42_confirm | confirm | 42 | HH | 0/256 (0.0%) | 2.0%/2.3% | 1.2%/1.3% | — | — | 256/256 |
| normalized_42_confirm | confirm | 42 | CC | 0/256 (0.0%) | 29.7%/28.9% | 5.2%/4.8% | 0.0% | 0.0% | 256/256 |
| normalized_42_confirm | confirm | 42 | CH | 0/256 (0.0%) | 1.2%/0.8% | 2.0%/2.5% | — | — | 256/256 |
| normalized_42_confirm | confirm | 42 | HC | 0/256 (0.0%) | 3.5%/4.3% | 2.7%/2.5% | 0.0% | 0.0% | 256/256 |
| normalized_42_confirm | confirm | 42 | HH | 0/256 (0.0%) | 1.6%/2.0% | 1.4%/1.5% | — | — | 256/256 |

Teacher 合格子集的各方法分子/分母、canonical-key 诊断、成本与全部逐条配对向量见 summary.json。

- base_42：seed 42，新增 256 步，起始步 512，可训练参数 1870336；训练 86.4 秒。
- hidden_42：seed 42，新增 256 步，起始步 512，可训练参数 1870336；训练 79.9 秒。
- clip_42：seed 42，新增 256 步，起始步 512，可训练参数 1870336；训练 86.3 秒。
- on_policy_42：seed 42，新增 256 步，起始步 512，可训练参数 1870336；训练 169.8 秒。
- normalized_42：seed 42，新增 256 步，起始步 512，可训练参数 1870336；训练 86.4 秒。

### 缓存准备

- prepare：prepare 已完成；源文件哈希和结果见 JSON。
- prepare：prepare 已完成；源文件哈希和结果见 JSON。

## Smoke，仅验证管线

### Step 2：Writer

- last：seed 42，新增 8 步，起始步 0，可训练参数 622592；训练 0.6 秒。
- mean：seed 42，新增 8 步，起始步 0，可训练参数 622592；训练 0.7 秒。
- mlp：seed 42，新增 8 步，起始步 0，可训练参数 3129344；训练 0.5 秒。

### Step 4：KD

- train_smoke：seed 42，新增 4 步，起始步 0，可训练参数 1870336；训练 9.4 秒。

## Preflight，仅作预检

### Teacher/损失校准

- calibrate：calibrate 已完成；源文件哈希和结果见 JSON。

## Seed 稳定性

只在同 scope、step、split、完整配对内容一致时聚合；单 seed 不代表稳定性。

| Scope | Step | Split | Arm | Phase/method | Seeds | Mean / min / max |
|---|---|---|---|---|---|---|
| formal | step2_writer | confirm | baseline | CC/real | [42] | 87.5% / 87.5% / 87.5% |
| formal | step2_writer | confirm | baseline | CH/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | confirm | baseline | HC/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | confirm | baseline | HH/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | confirm | mean | CC/real | [42, 43, 44] | 84.9% / 82.0% / 87.5% |
| formal | step2_writer | confirm | mean | CH/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | confirm | mean | HC/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | confirm | mean | HH/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | dev | baseline | CC/real | [42] | 71.9% / 71.9% / 71.9% |
| formal | step2_writer | dev | baseline | CH/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | dev | baseline | HC/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | dev | baseline | HH/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | dev | last | CC/real | [42, 43, 44] | 68.8% / 68.8% / 68.8% |
| formal | step2_writer | dev | last | CH/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | dev | last | HC/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | dev | last | HH/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | dev | mean | CC/real | [42, 43, 44] | 75.0% / 75.0% / 75.0% |
| formal | step2_writer | dev | mean | CH/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | dev | mean | HC/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | dev | mean | HH/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | dev | mlp | CC/real | [42, 43, 44] | 75.0% / 75.0% / 75.0% |
| formal | step2_writer | dev | mlp | CH/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | dev | mlp | HC/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step2_writer | dev | mlp | HH/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step3_address | confirm | coupled | CC/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step3_address | confirm | coupled | CH/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step3_address | confirm | coupled | HC/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step3_address | confirm | coupled | HH/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step3_address | confirm | decoupled | CC/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step3_address | confirm | decoupled | CH/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step3_address | confirm | decoupled | HC/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step3_address | confirm | decoupled | HH/real | [42, 43, 44] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | base | CC/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | base | CH/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | base | HC/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | base | HH/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | clip | CC/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | clip | CH/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | clip | HC/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | clip | HH/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | hidden | CC/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | hidden | CH/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | hidden | HC/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | hidden | HH/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | normalized | CC/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | normalized | CH/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | normalized | HC/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | normalized | HH/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | on_policy | CC/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | on_policy | CH/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | on_policy | HC/real | [42] | 0.0% / 0.0% / 0.0% |
| formal | step4_kd | confirm | on_policy | HH/real | [42] | 0.0% / 0.0% / 0.0% |

## 成对差异

固定 seed 68043，2000 次按实体簇重采样；数值为 paired EM 百分点差和 95% percentile 区间。仅比较同 split、完整 case IDs/内容/银行背景/预算；跨 run 还要求同优化 seed。

| Scope/split | Left − right | Phase | Δ pp [95% CI] | 簇数 |
|---|---|---|---|---|
| formal/confirm | baseline_confirm:real − baseline_confirm:shuffled | CC | +87.5 [+82.0, +93.0] | 128 |
| formal/confirm | baseline_confirm:real − baseline_confirm:empty | CC | +87.5 [+82.0, +93.0] | 128 |
| formal/confirm | baseline_confirm:real − baseline_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 128 |
| formal/confirm | baseline_confirm:real − baseline_confirm:empty | HC | +0.0 [+0.0, +0.0] | 128 |
| formal/confirm | baseline_confirm:real − baseline_confirm:canonical_key | HC | +0.0 [+0.0, +0.0] | 128 |
| formal/confirm | mean_42_confirm:real − mean_42_confirm:shuffled | CC | +82.0 [+75.0, +88.3] | 128 |
| formal/confirm | mean_42_confirm:real − mean_42_confirm:empty | CC | +82.0 [+75.0, +88.3] | 128 |
| formal/confirm | mean_42_confirm:real − mean_42_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 128 |
| formal/confirm | mean_42_confirm:real − mean_42_confirm:empty | HC | +0.0 [+0.0, +0.0] | 128 |
| formal/confirm | mean_42_confirm:real − mean_42_confirm:canonical_key | HC | -0.8 [-2.3, +0.0] | 128 |
| formal/confirm | mean_43_confirm:real − mean_43_confirm:shuffled | CC | +87.5 [+81.2, +93.0] | 128 |
| formal/confirm | mean_43_confirm:real − mean_43_confirm:empty | CC | +87.5 [+81.2, +93.0] | 128 |
| formal/confirm | mean_43_confirm:real − mean_43_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 128 |
| formal/confirm | mean_43_confirm:real − mean_43_confirm:empty | HC | +0.0 [+0.0, +0.0] | 128 |
| formal/confirm | mean_43_confirm:real − mean_43_confirm:canonical_key | HC | -2.3 [-5.5, +0.0] | 128 |
| formal/confirm | mean_44_confirm:real − mean_44_confirm:shuffled | CC | +85.2 [+78.9, +91.4] | 128 |
| formal/confirm | mean_44_confirm:real − mean_44_confirm:empty | CC | +85.2 [+78.9, +91.4] | 128 |
| formal/confirm | mean_44_confirm:real − mean_44_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 128 |
| formal/confirm | mean_44_confirm:real − mean_44_confirm:empty | HC | +0.0 [+0.0, +0.0] | 128 |
| formal/confirm | mean_44_confirm:real − mean_44_confirm:canonical_key | HC | -0.8 [-2.3, +0.0] | 128 |
| formal/confirm | coupled_42_confirm:real − coupled_42_confirm:shuffled | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_42_confirm:real − coupled_42_confirm:empty | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_42_confirm:real − coupled_42_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_42_confirm:real − coupled_42_confirm:empty | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_42_confirm:real − coupled_42_confirm:canonical_key | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_42_confirm:real − decoupled_42_confirm:shuffled | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_42_confirm:real − decoupled_42_confirm:empty | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_42_confirm:real − decoupled_42_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_42_confirm:real − decoupled_42_confirm:empty | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_42_confirm:real − decoupled_42_confirm:canonical_key | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_43_confirm:real − coupled_43_confirm:shuffled | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_43_confirm:real − coupled_43_confirm:empty | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_43_confirm:real − coupled_43_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_43_confirm:real − coupled_43_confirm:empty | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_43_confirm:real − coupled_43_confirm:canonical_key | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_43_confirm:real − decoupled_43_confirm:shuffled | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_43_confirm:real − decoupled_43_confirm:empty | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_43_confirm:real − decoupled_43_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_43_confirm:real − decoupled_43_confirm:empty | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_43_confirm:real − decoupled_43_confirm:canonical_key | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_44_confirm:real − coupled_44_confirm:shuffled | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_44_confirm:real − coupled_44_confirm:empty | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_44_confirm:real − coupled_44_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_44_confirm:real − coupled_44_confirm:empty | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_44_confirm:real − coupled_44_confirm:canonical_key | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_44_confirm:real − decoupled_44_confirm:shuffled | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_44_confirm:real − decoupled_44_confirm:empty | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_44_confirm:real − decoupled_44_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_44_confirm:real − decoupled_44_confirm:empty | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | decoupled_44_confirm:real − decoupled_44_confirm:canonical_key | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − base_42_confirm:shuffled | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − base_42_confirm:empty | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − base_42_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − base_42_confirm:empty | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − base_42_confirm:canonical_key | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − hidden_42_confirm:shuffled | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − hidden_42_confirm:empty | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − hidden_42_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − hidden_42_confirm:empty | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − hidden_42_confirm:canonical_key | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | clip_42_confirm:real − clip_42_confirm:shuffled | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | clip_42_confirm:real − clip_42_confirm:empty | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | clip_42_confirm:real − clip_42_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | clip_42_confirm:real − clip_42_confirm:empty | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | clip_42_confirm:real − clip_42_confirm:canonical_key | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | on_policy_42_confirm:real − on_policy_42_confirm:shuffled | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | on_policy_42_confirm:real − on_policy_42_confirm:empty | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | on_policy_42_confirm:real − on_policy_42_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | on_policy_42_confirm:real − on_policy_42_confirm:empty | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | on_policy_42_confirm:real − on_policy_42_confirm:canonical_key | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | normalized_42_confirm:real − normalized_42_confirm:shuffled | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | normalized_42_confirm:real − normalized_42_confirm:empty | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | normalized_42_confirm:real − normalized_42_confirm:shuffled | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | normalized_42_confirm:real − normalized_42_confirm:empty | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | normalized_42_confirm:real − normalized_42_confirm:canonical_key | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/dev | baseline_dev:real − baseline_dev:shuffled | CC | +71.9 [+53.1, +87.5] | 32 |
| formal/dev | baseline_dev:real − baseline_dev:empty | CC | +71.9 [+53.1, +87.5] | 32 |
| formal/dev | baseline_dev:real − baseline_dev:shuffled | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | baseline_dev:real − baseline_dev:empty | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | baseline_dev:real − baseline_dev:canonical_key | HC | -3.1 [-9.4, +0.0] | 32 |
| formal/dev | last_42_dev:real − last_42_dev:shuffled | CC | +68.8 [+50.0, +84.4] | 32 |
| formal/dev | last_42_dev:real − last_42_dev:empty | CC | +68.8 [+50.0, +84.4] | 32 |
| formal/dev | last_42_dev:real − last_42_dev:shuffled | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_42_dev:real − last_42_dev:empty | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_42_dev:real − last_42_dev:canonical_key | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_42_dev:real − mean_42_dev:shuffled | CC | +75.0 [+59.4, +90.6] | 32 |
| formal/dev | mean_42_dev:real − mean_42_dev:empty | CC | +75.0 [+59.4, +90.6] | 32 |
| formal/dev | mean_42_dev:real − mean_42_dev:shuffled | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_42_dev:real − mean_42_dev:empty | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_42_dev:real − mean_42_dev:canonical_key | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mlp_42_dev:real − mlp_42_dev:shuffled | CC | +75.0 [+59.4, +90.6] | 32 |
| formal/dev | mlp_42_dev:real − mlp_42_dev:empty | CC | +75.0 [+59.4, +90.6] | 32 |
| formal/dev | mlp_42_dev:real − mlp_42_dev:shuffled | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mlp_42_dev:real − mlp_42_dev:empty | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mlp_42_dev:real − mlp_42_dev:canonical_key | HC | -3.1 [-9.4, +0.0] | 32 |
| formal/dev | last_43_dev:real − last_43_dev:shuffled | CC | +68.8 [+50.0, +84.4] | 32 |
| formal/dev | last_43_dev:real − last_43_dev:empty | CC | +68.8 [+50.0, +84.4] | 32 |
| formal/dev | last_43_dev:real − last_43_dev:shuffled | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_43_dev:real − last_43_dev:empty | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_43_dev:real − last_43_dev:canonical_key | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_43_dev:real − mean_43_dev:shuffled | CC | +75.0 [+59.4, +90.6] | 32 |
| formal/dev | mean_43_dev:real − mean_43_dev:empty | CC | +75.0 [+59.4, +90.6] | 32 |
| formal/dev | mean_43_dev:real − mean_43_dev:shuffled | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_43_dev:real − mean_43_dev:empty | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_43_dev:real − mean_43_dev:canonical_key | HC | -3.1 [-9.4, +0.0] | 32 |
| formal/dev | mlp_43_dev:real − mlp_43_dev:shuffled | CC | +75.0 [+59.4, +90.6] | 32 |
| formal/dev | mlp_43_dev:real − mlp_43_dev:empty | CC | +75.0 [+59.4, +90.6] | 32 |
| formal/dev | mlp_43_dev:real − mlp_43_dev:shuffled | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mlp_43_dev:real − mlp_43_dev:empty | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mlp_43_dev:real − mlp_43_dev:canonical_key | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_44_dev:real − last_44_dev:shuffled | CC | +68.8 [+50.0, +84.4] | 32 |
| formal/dev | last_44_dev:real − last_44_dev:empty | CC | +68.8 [+50.0, +84.4] | 32 |
| formal/dev | last_44_dev:real − last_44_dev:shuffled | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_44_dev:real − last_44_dev:empty | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_44_dev:real − last_44_dev:canonical_key | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_44_dev:real − mean_44_dev:shuffled | CC | +75.0 [+59.4, +90.6] | 32 |
| formal/dev | mean_44_dev:real − mean_44_dev:empty | CC | +75.0 [+59.4, +90.6] | 32 |
| formal/dev | mean_44_dev:real − mean_44_dev:shuffled | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_44_dev:real − mean_44_dev:empty | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_44_dev:real − mean_44_dev:canonical_key | HC | -3.1 [-9.4, +0.0] | 32 |
| formal/dev | mlp_44_dev:real − mlp_44_dev:shuffled | CC | +75.0 [+59.4, +90.6] | 32 |
| formal/dev | mlp_44_dev:real − mlp_44_dev:empty | CC | +75.0 [+59.4, +90.6] | 32 |
| formal/dev | mlp_44_dev:real − mlp_44_dev:shuffled | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mlp_44_dev:real − mlp_44_dev:empty | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mlp_44_dev:real − mlp_44_dev:canonical_key | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/confirm | baseline_confirm:real − mean_42_confirm:real | CC | +5.5 [-1.6, +12.5] | 128 |
| formal/confirm | baseline_confirm:real − mean_42_confirm:real | CH | +0.0 [+0.0, +0.0] | 128 |
| formal/confirm | baseline_confirm:real − mean_42_confirm:real | HC | +0.0 [+0.0, +0.0] | 128 |
| formal/confirm | baseline_confirm:real − mean_42_confirm:real | HH | +0.0 [+0.0, +0.0] | 128 |
| formal/confirm | coupled_42_confirm:real − decoupled_42_confirm:real | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_42_confirm:real − decoupled_42_confirm:real | CH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_42_confirm:real − decoupled_42_confirm:real | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_42_confirm:real − decoupled_42_confirm:real | HH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_43_confirm:real − decoupled_43_confirm:real | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_43_confirm:real − decoupled_43_confirm:real | CH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_43_confirm:real − decoupled_43_confirm:real | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_43_confirm:real − decoupled_43_confirm:real | HH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_44_confirm:real − decoupled_44_confirm:real | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_44_confirm:real − decoupled_44_confirm:real | CH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_44_confirm:real − decoupled_44_confirm:real | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | coupled_44_confirm:real − decoupled_44_confirm:real | HH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − hidden_42_confirm:real | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − hidden_42_confirm:real | CH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − hidden_42_confirm:real | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − hidden_42_confirm:real | HH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − clip_42_confirm:real | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − clip_42_confirm:real | CH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − clip_42_confirm:real | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − clip_42_confirm:real | HH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − on_policy_42_confirm:real | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − on_policy_42_confirm:real | CH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − on_policy_42_confirm:real | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − on_policy_42_confirm:real | HH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − normalized_42_confirm:real | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − normalized_42_confirm:real | CH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − normalized_42_confirm:real | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | base_42_confirm:real − normalized_42_confirm:real | HH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − clip_42_confirm:real | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − clip_42_confirm:real | CH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − clip_42_confirm:real | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − clip_42_confirm:real | HH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − on_policy_42_confirm:real | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − on_policy_42_confirm:real | CH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − on_policy_42_confirm:real | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − on_policy_42_confirm:real | HH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − normalized_42_confirm:real | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − normalized_42_confirm:real | CH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − normalized_42_confirm:real | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | hidden_42_confirm:real − normalized_42_confirm:real | HH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | clip_42_confirm:real − on_policy_42_confirm:real | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | clip_42_confirm:real − on_policy_42_confirm:real | CH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | clip_42_confirm:real − on_policy_42_confirm:real | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | clip_42_confirm:real − on_policy_42_confirm:real | HH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | clip_42_confirm:real − normalized_42_confirm:real | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | clip_42_confirm:real − normalized_42_confirm:real | CH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | clip_42_confirm:real − normalized_42_confirm:real | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | clip_42_confirm:real − normalized_42_confirm:real | HH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | on_policy_42_confirm:real − normalized_42_confirm:real | CC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | on_policy_42_confirm:real − normalized_42_confirm:real | CH | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | on_policy_42_confirm:real − normalized_42_confirm:real | HC | +0.0 [+0.0, +0.0] | 64 |
| formal/confirm | on_policy_42_confirm:real − normalized_42_confirm:real | HH | +0.0 [+0.0, +0.0] | 64 |
| formal/dev | baseline_dev:real − last_42_dev:real | CC | +3.1 [+0.0, +9.4] | 32 |
| formal/dev | baseline_dev:real − last_42_dev:real | CH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | baseline_dev:real − last_42_dev:real | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | baseline_dev:real − last_42_dev:real | HH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | baseline_dev:real − mean_42_dev:real | CC | -3.1 [-18.8, +9.4] | 32 |
| formal/dev | baseline_dev:real − mean_42_dev:real | CH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | baseline_dev:real − mean_42_dev:real | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | baseline_dev:real − mean_42_dev:real | HH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | baseline_dev:real − mlp_42_dev:real | CC | -3.1 [-9.4, +0.0] | 32 |
| formal/dev | baseline_dev:real − mlp_42_dev:real | CH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | baseline_dev:real − mlp_42_dev:real | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | baseline_dev:real − mlp_42_dev:real | HH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_42_dev:real − mean_42_dev:real | CC | -6.2 [-21.9, +9.4] | 32 |
| formal/dev | last_42_dev:real − mean_42_dev:real | CH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_42_dev:real − mean_42_dev:real | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_42_dev:real − mean_42_dev:real | HH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_42_dev:real − mlp_42_dev:real | CC | -6.2 [-15.6, +0.0] | 32 |
| formal/dev | last_42_dev:real − mlp_42_dev:real | CH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_42_dev:real − mlp_42_dev:real | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_42_dev:real − mlp_42_dev:real | HH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_42_dev:real − mlp_42_dev:real | CC | +0.0 [-12.5, +12.5] | 32 |
| formal/dev | mean_42_dev:real − mlp_42_dev:real | CH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_42_dev:real − mlp_42_dev:real | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_42_dev:real − mlp_42_dev:real | HH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_43_dev:real − mean_43_dev:real | CC | -6.2 [-21.9, +9.4] | 32 |
| formal/dev | last_43_dev:real − mean_43_dev:real | CH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_43_dev:real − mean_43_dev:real | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_43_dev:real − mean_43_dev:real | HH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_43_dev:real − mlp_43_dev:real | CC | -6.2 [-15.6, +0.0] | 32 |
| formal/dev | last_43_dev:real − mlp_43_dev:real | CH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_43_dev:real − mlp_43_dev:real | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_43_dev:real − mlp_43_dev:real | HH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_43_dev:real − mlp_43_dev:real | CC | +0.0 [-12.5, +12.5] | 32 |
| formal/dev | mean_43_dev:real − mlp_43_dev:real | CH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_43_dev:real − mlp_43_dev:real | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_43_dev:real − mlp_43_dev:real | HH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_44_dev:real − mean_44_dev:real | CC | -6.2 [-18.8, +6.2] | 32 |
| formal/dev | last_44_dev:real − mean_44_dev:real | CH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_44_dev:real − mean_44_dev:real | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_44_dev:real − mean_44_dev:real | HH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_44_dev:real − mlp_44_dev:real | CC | -6.2 [-15.6, +0.0] | 32 |
| formal/dev | last_44_dev:real − mlp_44_dev:real | CH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_44_dev:real − mlp_44_dev:real | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | last_44_dev:real − mlp_44_dev:real | HH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_44_dev:real − mlp_44_dev:real | CC | +0.0 [-12.5, +12.5] | 32 |
| formal/dev | mean_44_dev:real − mlp_44_dev:real | CH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_44_dev:real − mlp_44_dev:real | HC | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | mean_44_dev:real − mlp_44_dev:real | HH | +0.0 [+0.0, +0.0] | 32 |
| formal/dev | four_bank:KcVh − four_bank:KcVc | cf_confirmation_support_json | -82.8 [-90.6, -73.4] | 64 |
| formal/dev | four_bank:KhVc − four_bank:KcVc | cf_confirmation_support_json | -79.7 [-89.1, -70.3] | 64 |
| formal/dev | four_bank:KhVh − four_bank:KcVc | cf_confirmation_support_json | -82.8 [-90.6, -73.4] | 64 |
| formal/dev | four_bank:KcVh − four_bank:KcVc | cf_confirmation_support_markdown | -81.2 [-90.6, -71.9] | 64 |
| formal/dev | four_bank:KhVc − four_bank:KcVc | cf_confirmation_support_markdown | -81.2 [-90.6, -71.9] | 64 |
| formal/dev | four_bank:KhVh − four_bank:KcVc | cf_confirmation_support_markdown | -82.8 [-90.6, -73.4] | 64 |

## 成本与门控

| Scope | 完成任务 | Job wall 秒 | Train 秒 | Train input tokens | Eval generated tokens | Answer scoring tokens |
|---|---:|---:|---:|---:|---:|---:|
| formal | 48 | 17559.0 | 1519.7 | 22383360 | 409680.0 | 244232.0 |
| smoke | 4 | 151.4 | 11.1 | 28677 | 0 | 0 |
| preflight | 1 | 135.1 | 0.0 | 0 | 0 | 0 |

Job wall 包含初始化并与 train/generation 时间重叠，不能相加。空库 A_and_B 只生成一次，不双计成本；warm 任务只计一次。Writer 缓存样本暴露不是 token 数。

训练 valid-pair 门控和 teacher 自由生成验收分别记录。Dev 仅选择 writer；正式部署门槛在随后 classic confirm 三 seed 对 baseline 判定：HC +20pp、CC 下降≤5pp。未取得 confirm 时保持 pending，不能由 dev 失败判定正式失败，也不能据 confirm 重新选模型。

- Dev 诊断：available。
  - Writer last：diagnostic_only。
  - Writer mean：diagnostic_only。
  - Writer mlp：diagnostic_only。
- 正式 Confirm gate：assessed。
  - Writer mean：fail。
- interface_extended_preflight_20261006/train_smoke：behavior_valid_pairs 32/32；legacy_behavior_clipped_valid_pairs 32/32；hidden_valid_pairs 32/32；teacher_first_token_joint_correct 31/32；student_first_token_joint_correct 0/32
- interface_step3_train_a_20261006/coupled_42：behavior_valid_pairs 4096/4096；legacy_behavior_clipped_valid_pairs 4096/4096；hidden_valid_pairs 4096/4096；teacher_first_token_joint_correct 3947/4096；student_first_token_joint_correct 1/4096
- interface_step3_train_a_20261006/decoupled_42：behavior_valid_pairs 4096/4096；legacy_behavior_clipped_valid_pairs 4096/4096；hidden_valid_pairs 4096/4096；teacher_first_token_joint_correct 3947/4096；student_first_token_joint_correct 1/4096
- interface_step3_train_b_20261006/coupled_43：behavior_valid_pairs 4096/4096；legacy_behavior_clipped_valid_pairs 4096/4096；hidden_valid_pairs 4096/4096；teacher_first_token_joint_correct 3937/4096；student_first_token_joint_correct 2/4096
- interface_step3_train_b_20261006/decoupled_43：behavior_valid_pairs 4096/4096；legacy_behavior_clipped_valid_pairs 4096/4096；hidden_valid_pairs 4096/4096；teacher_first_token_joint_correct 3937/4096；student_first_token_joint_correct 2/4096
- interface_step3_train_c_20261006/coupled_44：behavior_valid_pairs 4096/4096；legacy_behavior_clipped_valid_pairs 4096/4096；hidden_valid_pairs 4096/4096；teacher_first_token_joint_correct 3958/4096；student_first_token_joint_correct 0/4096
- interface_step3_train_c_20261006/decoupled_44：behavior_valid_pairs 4096/4096；legacy_behavior_clipped_valid_pairs 4096/4096；hidden_valid_pairs 4096/4096；teacher_first_token_joint_correct 3958/4096；student_first_token_joint_correct 0/4096
- interface_step4_train_a_20261006/base_42：behavior_valid_pairs 2048/2048；legacy_behavior_clipped_valid_pairs 2048/2048；hidden_valid_pairs 2048/2048；teacher_first_token_joint_correct 1970/2048；student_first_token_joint_correct 2/2048
- interface_step4_train_a_20261006/hidden_42：behavior_valid_pairs 2048/2048；legacy_behavior_clipped_valid_pairs 2048/2048；hidden_valid_pairs 2048/2048；teacher_first_token_joint_correct 1970/2048；student_first_token_joint_correct 3/2048
- interface_step4_train_b_20261006/clip_42：behavior_valid_pairs 2048/2048；legacy_behavior_clipped_valid_pairs 2048/2048；hidden_valid_pairs 2048/2048；teacher_first_token_joint_correct 1970/2048；student_first_token_joint_correct 2/2048
- interface_step4_train_b_20261006/on_policy_42：behavior_valid_pairs 2048/2048；legacy_behavior_clipped_valid_pairs 2048/2048；hidden_valid_pairs 2048/2048；teacher_first_token_joint_correct 1970/2048；student_first_token_joint_correct 4/2048
- interface_step4_train_c_20261006/normalized_42：behavior_valid_pairs 2048/2048；legacy_behavior_clipped_valid_pairs 2048/2048；hidden_valid_pairs 2048/2048；teacher_first_token_joint_correct 1970/2048；student_first_token_joint_correct 0/2048

## 限制

- Only interface_* suite manifests are discovered; historical counterfactual/scaling runs are never silently added.
- Smoke/preflight evidence is displayed separately and excluded from formal seed aggregation and formal costs.
- Development scores support model selection; confirmation scores are distinct. This audit does not authorize selection on confirmation.
- HC canonical_key and step1 four banks are diagnostic interventions, not deployable methods or upper bounds.
- Paired bootstrap clusters observed entity facts. Partial target sampling can omit relations or members of four-entity groups; supplementary group intervals require complete reconstructible groups.
- Development generation chooses the writer family only. The formal writer gate uses the subsequent fixed classic confirmation, HC +20 points and CC degradation at most 5 points in seeds 42/43/44; absent confirmation remains pending. Teacher eligibility is separate.
- Budget-hit means generated length reached the limit, not proof of truncation: EOS could be the last token.
- Evaluation prompt/context tokens, answer-scoring time, writer encoding and VDB write/read timings were not separately logged; generated and target-token counters are not total inference cost.
- SHA/provenance and row arithmetic are audited; this script does not replay tensors, model generation or training, and does not verify external historical checkpoint contents.
