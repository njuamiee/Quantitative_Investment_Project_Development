# 调研来源与采用范围（核对日期：2026-10-11）

1. [Microsoft Qlib 官方仓库](https://github.com/microsoft/qlib)：对照监督学习、状态模型与组合评估的研究流程。本轮实际训练 Ridge、LightGBM、LambdaRank、CatBoost、XGBoost、ExtraTrees 和集成；没有把官方其他数据集的收益移植成本项目收益。神经网络、图模型和强化学习列为后续候选，需额外数据与单独验证。
2. [RD-Agent 官方仓库](https://github.com/microsoft/RD-Agent) 与[原始论文](https://arxiv.org/abs/2505.15155)：参考“提出因子假说—实现—反馈”的流程。已做本地受限表达式和候选留痕；没有声称运行其收费大模型后台，也没有复用其论文收益作为本项目证据。
3. [Alpha-GPT 2.0 原始论文](https://arxiv.org/abs/2402.09746)：参考人参与的假说审核、因子解释与迭代。工作区文件名包含“2026”，论文公开编号及初次提交年份为2024，按原始来源引用。
4. [Kronos 官方仓库](https://github.com/shiyu-coder/Kronos) 与[原始论文](https://arxiv.org/abs/2508.02739)：金融K线分词与自回归预测模型。本轮下载公开mini权重，在本机推理；代码与权重提交号写入 kronos_results.json。稀疏截面检验与完整选股资金回测分别陈述；预训练数据版本尚未独立核实。
5. [TypeSafe Jev API](https://docs.typesafe.ai/api) 与[模型说明](https://docs.typesafe.ai/models)：实现 choice、score、noul 三类问题。固定模型标识 jev-1.13.0；score 按等级索引的概率加权值解释，本项目五档为0—4。真实调用按用户要求禁用；测试使用显式合成样本。中文业务的校准与真实增益尚待标签数据验证。
6. [Factor Momentum and the Momentum Factor](https://doi.org/10.1111/jofi.13131)：本地会议PDF为2022年Journal of Finance版本，44页。论文研究因子收益延续及其与个股动量的关系。本轮做63日因子收益均值/波动率驱动的近似择时，未复现其原始跨市场月度组合。不能由论文结论推定北交所短期信号必然有效。
7. [Factor Timing](https://www.nber.org/system/files/working_papers/w26708/w26708.pdf)：本地PDF为Haddad、Kozak、Santosh，NBER WP26708，2020年版，80页。参考因子条件预期收益与配置的思路；本轮HMM状态均值和收缩协方差仅为项目内近似，未声称完整重现论文的经济约束或SDF估计。
8. [北交所2026年休市公告](https://www.bse.cn/important_news/200027428.html)：10月1—7日休市，10月8日起开市。会议中“10月8日仍属国庆休市”的表述需要更正；10月8、9日缺少分钟文件是有效交易日缺口。
9. 本地《任务6：HMM市场状态识别.pdf》4页：五维观测、滚动估计、前向滤波、状态标签、软风险预算、状态条件因子配置。宏观历史版本、价差和融资数据缺失已列出。
10. 本地天软《191因子算法及其计算说明》（2021-11-26版）及 registry.csv：用于供应商表与字段映射、公式和含义核对；当前文件不能替代历史发布时点档案。
11. 1009会议记录/新录音.txt、新录音 2.txt、项目规划.mm：老师要求与任务分工的原始依据。项目规划含连接信息，报告不复制这些凭据。

检索和实测覆盖了与现有数据相匹配的主要模型路线，不宣称穷尽全球项目。任何GitHub星标、论文成绩或供应商宣传均不构成本项目达标证据。
