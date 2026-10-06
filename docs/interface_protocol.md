# 向量记忆接口与训练实验

本轮按四银行干预、writer调整、事实地址与多token答案、蒸馏消融四步执行。目的在于定位跨表达失败并测试有限改动，不以重复调整确认集分数筛选方案。基座为冻结的Qwen3-4B-Instruct-2507，revision沿用历史固定版本；第20层down_proj、rank/key64、top-k4保持不变。原始结果在H100 SSD3运行，保存源文件哈希、实际命令与轻量checkpoint，并同步到本地。

## 四银行干预

新开发实体seed47042，共64条。读取原问句；其他记录保持canonical。目标记录分别使用KcVc、KhVc、KcVh、KhVh，两个新support结构为JSON与Markdown。每种A/B世界独立修改一个目标记录。value-only比较在相同问题、相同key下核对实际首位置top-k索引和权重逐位相同；自由生成单独报告。原文teacher预算16 tokens。此阶段是开发诊断，不是后续确认集。

## Writer验证

独立train4096条沿用历史事实；新dev实体seed48042，使用历史自然语言开发模板；新confirm实体seed58042，128条，JSON和Markdown support。结构化query末尾补明确单词输出指令，生成预算32。所有writer重新生成特征，统一Remember前缀；池化只含support正文token，不含指令、system、chat或padding。

从既有base checkpoint初始化，固定读出与地址参数。比较last-linear、mean-linear、last-MLP256；只有Wv和可选MLP训练。目标是同事实同world canonical表达经冻结旧writer得到的value，A/B目标独立。各2000次batch128更新、Adam1e-4、clip1，训练seed42/43/44，固定末步。mean单独拟合train value中心。完整记录实际训练参数量，不能称等参数容量对照。

开发集只查询固定32个目标，银行保持128条；选择以真实HC paired为第一指标，若同分，以canonical-key诊断HC paired为第二指标，再以CC paired排序，仍同分优先更简单last-linear。确认前固定选择；同时保留所有开发结果。进入主线的最低阈值为新support原query相对原模型增加20个百分点、原格式下降不超过5个百分点，三个seed复现。该阈值是项目决策门槛。

## 地址与扩展答案

使用独立entity、每entity四关系、三词答案；train/dev/confirm完整答案无交集。对应split seeds41042/42042/43042。真实A银行包含target A与B答案的其他实体硬负例，并包含同实体其他关系负例。训练只用自然语言模板；确认包括JSON、Markdown、YAML、INI，query/support结构在关系内独立均衡，避免结构绑定关系。

对照同数据、同初始化、同采样日程的公共配对监督与附加同地址A/B key一致性0.05。每组512次batch8更新，三个训练seed。共享参数离线训练、评估全部冻结。此任务与历史16词任务分别报告，不用难度变化推断某单项调整收益。最终confirm共256条事实（64实体），不把同实体四关系当独立实体。

## 蒸馏消融

从扩展任务地址一致性组的共同checkpoint和Adam状态继续，比较base、原clip10差分、归一化差分、归一化加hidden、归一化加25%学生轨迹。教师尺度只用前256个训练事实校准，固定为有效teacher差分绝对值中位数。归一化损失为scale乘SmoothL1，保留教师幅度变化；不逐样本除以自身差分。

五臂使用相同FKL、gold全序列CE、gold实际地址、跨表达地址和A/P一致性；on-policy只替换FKL的前缀来源，其他监督始终使用共同gold分支。每4步从A/B各半混合学生policy采样公共前缀，上限8 tokens，其他分支不伪造EOS；这是world-mixture FKL，不是原论文OPCD复现。hidden与on-policy分别相对归一化臂比较，避免同时新增两项。无周期回放，记录全部额外forward和token成本，不宣称等FLOPs。

若此前部署路径达到继续门槛，五臂各512次更新、三个seed；若未达到，限制为seed42、五臂各256次的诊断性比较，明确不能据此宣称稳定增益。所有训练结束后才执行新confirm生成，不根据confirm挑checkpoint。

## 验收与解释

主指标为同一事实A/B均严格正确，分别报告单边正确、首位置R1/R4、decode驻留、格式或预算失败、teacher合格子集与全体分母。真实、shuffle、empty对照覆盖CC/HC；其他两个query变化条件按实际评估矩阵说明。canonical-key仅是机制诊断，不能作为部署结果。在线修改一条记录不能改变其他记录字节与时间戳。

统计按实体聚类配对，扩展答案的共同构造组作为补充聚类；三个seed同时报告，不能仅将样本bootstrap当训练不确定性。报告写入/读取与训练成本的实测范围，不从小银行精确检索外推生产级VDB规模优势。门槛失败时保留负结果并停止继续叠加当前结构的损失。

## 执行记录补充

正式扩展任务训练开始前，train-only校准得到scale=71.015625；256个有效teacher差分均被旧clip10饱和。teacher接口预检使用16个训练事实、20个表达条件、A/B两个世界，共640次生成，全部严格正确；它们不是640个独立事实。旧clip仍保留样本各自的A/B token标签，不能称为没有内容监督。新版同时恢复未截断目标并改变Huber转折尺度；后续对比识别这项整体修正，不单独识别目标幅度多样性的效果。

dev只用于锁定writer；继续门槛在新classic confirm上逐seed判断。三个训练seed共享历史base初始化，因此衡量的是本轮优化采样日程的不确定性，未覆盖基座、随机投影初始化或数据划分的不确定性。扩展任务完整答案跨split无交集，但构成答案的词汇重叠；不是开放词表或任意文档记忆实验。

确认评估可在同一空闲GPU上并行两个独立进程，模型和银行各自隔离、数量与预算不变。启动前检查GPU无其他计算进程，填位时只接受本runner仍存活的子进程；失败停止派发并让已启动评估自然结束。并行时单任务墙钟受资源竞争影响，不能将它当作单请求服务延迟比较。

## 完成记录

四步已于2026-10-06完成，共22个suite、53个job。正式writer门槛失败，第四步依预定规则执行seed42、每臂256步的诊断预算。结果及核验边界见[四步结果与学生交接](interface_results.md)；本节只记录已执行状态，不改变上面的选型、预算或评估规则。
