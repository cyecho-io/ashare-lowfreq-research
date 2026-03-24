# 从基线到 V1：一版模型驱动交易策略的构建记录

> 注：本文记录的是 `v1` 构建阶段的原始实验过程。`2026-03-24` 在组合规则诊断中发现 `rebalance_every` 曾存在实现缺陷，因此文中部分历史绩效数字应视为“修复前结果”，不再代表当前最新基准。修复后的分析见 [strategy-v2-ic-diagnosis.md](/Users/yongqiuwu/works/github/Trade/docs/strategy-v2-ic-diagnosis.md)。

这篇文章记录了本仓库第一版模型驱动交易策略从建立基线、做参数调试、补稳健性验证，到最终收口为 `v1` 的全过程。目标不是证明这已经是一版可直接实盘上线的成熟策略，而是把一条可复现、可解释、可继续迭代的研究链路真正跑通，并沉淀出一个足够清晰的版本节点。

## 摘要

本轮工作最终收口到一版研究型 `v1`：

- 标签：`industry_excess_fwd_return_5`
- 训练：`12` 个月 walk-forward
- 组合：`top_k=5`、`rebalance_every=3`、`min_hold_bars=12`
- 风险约束：`max_names_per_industry=2`
- 流动性过滤：能力已接入，但默认关闭

最终采用的正式配置文件是 [research_industry_v4_v1.toml](/Users/yongqiuwu/works/github/Trade/configs/research_industry_v4_v1.toml)，推荐结果目录是 [model_score_backtest_walk_forward_industry_v4_candidate_a_industry_only](/Users/yongqiuwu/works/github/Trade/results/model_score_backtest_walk_forward_industry_v4_candidate_a_industry_only)。

当前 `v1` 的核心指标：

| 指标 | 数值 |
| --- | ---: |
| 年化收益 | `59.60%` |
| 最大回撤 | `11.31%` |
| Sharpe | `2.01` |

这版策略已经可以被视为一个合格的研究原型，但还不是最终的实盘策略。它解决的是“第一版策略建设有没有真正完成闭环”的问题，而不是“这版策略是否已经完全成熟”的问题。

## 1. 这次要解决什么问题

在这次迭代之前，项目已经具备基础的数据、回测和研究模块，但还缺一个真正完成收口的策略版本。所谓“完成收口”，不是指参数扫一遍就结束，而是至少要满足下面几点：

- 有稳定可复现的输入和输出
- 有明确的标签、训练方式和组合规则
- 有一版正向结果，并且能重复跑出来
- 做过最基本的稳健性检查，而不是只拿单点结果下结论
- 至少补上一层基础风险约束

所以本轮工作的目标很明确：先做出一个研究型 `v1`，把完整链路走通，再决定下一步往稳健性还是执行层继续深化。

## 2. 初始条件

本轮实验的固定前提如下：

- 项目路径：`/Users/yongqiuwu/works/github/Trade`
- 数据源：Parquet 已导入
- 训练方式：LightGBM + 月度 walk-forward
- 训练窗口：`12` 个月
- 主要输出目录：
  - `research/factors/`
  - `research/models/`
  - `results/`

这意味着本轮不再花时间搭底层框架，而是直接在现有研究链路上收口策略版本。

## 3. 最早确认的有效基线

最开始确认的有效基线是：

- 标签：`industry_excess_fwd_return_5`
- 组合参数：
  - `top_k=5`
  - `min_hold_bars=10`
  - `keep_buffer=2`
  - `min_turnover_names=3`

对应参数扫面结果文件为 [model_sweep_industry_v4.csv](/Users/yongqiuwu/works/github/Trade/research/models/model_sweep_industry_v4.csv)。

在这份扫面结果中，最优组合对应：

- `top_k=5`
- `rebalance_every=3`
- `min_hold_bars=10`

对应回测结果为：

| 指标 | 数值 |
| --- | ---: |
| 总收益 | `29.19%` |
| 年化收益 | `47.51%` |
| 最大回撤 | `7.74%` |
| Sharpe | `1.77` |

结果目录： [model_score_backtest_walk_forward_industry_v4_best](/Users/yongqiuwu/works/github/Trade/results/model_score_backtest_walk_forward_industry_v4_best)

这一版的意义在于，它提供了一个清晰的起点：不是空谈策略想法，而是已经有一条能跑出正结果的基线链路。

## 4. 第一步不是继续调参，而是先确认基线可复现

拿到一组“看起来不错”的回测结果后，第一步并没有继续往下扫参数，而是先确认这组结果到底是不是能稳定重跑。

为此，直接使用已有的 walk-forward 分数文件复现了一次最优组合回测：

- 分数文件： [walk_forward_scores_industry_v4.parquet](/Users/yongqiuwu/works/github/Trade/research/models/walk_forward_scores_industry_v4.parquet)
- 指标文件： [walk_forward_metrics_industry_v4.json](/Users/yongqiuwu/works/github/Trade/research/models/walk_forward_metrics_industry_v4.json)

复现结果与扫面文件完全一致。这个动作虽然简单，但非常关键，因为它确认了两件事：

- 当前基线不是一次性跑出来的偶然结果
- 后续调优都可以建立在同一条稳定基线上

只有基线是稳的，后面的任何比较才有意义。

## 5. 为什么没有直接把基线当成 V1

虽然基线结果已经不错，但它还不能直接被当成最终的 `v1`，原因主要有三个：

1. 不知道收益是不是只集中在少数月份
2. 不知道模型信号本身是否稳定
3. 不知道这组参数是不是单点最优

如果这三件事不检查，就很容易把一个“回测样子不错”的结果误判成“策略已经完成”。

## 6. 稳健性检查：先看时间维度，再看参数维度

基线复现完成后，补了三类最基础但信息量很高的检查：

1. 月度收益拆解
2. 月度 IC 检查
3. 邻域参数扫面

### 6.1 月度收益拆解

把基线回测拆成月度之后，发现收益分布并不均匀。主要利润集中在：

- `2025-08`
- `2025-09`

而在以下月份出现了明显回撤：

- `2025-11`
- `2025-12`

这说明单看总收益会高估策略的稳定性，必须把时间结构也看进去。

### 6.2 月度 IC 检查

walk-forward 的月度 IC 均值约为 `0.0067`，而且组合月收益和月度 IC 的相关性并不明显。

这意味着当前现象更像是：

- 模型信号有一定作用
- 但强度还不够高
- 组合规则对最终回测结果的贡献不小

这个结论很重要，因为它告诉我们：这版策略可以继续推进，但不能把它包装成一个已经被充分验证的成熟 alpha。

### 6.3 邻域参数扫面

为了判断当前组合是不是只是单点最优，又围绕原组合做了一轮邻域扫面，输出文件为：

- [model_sweep_industry_v4_stability.csv](/Users/yongqiuwu/works/github/Trade/research/models/model_sweep_industry_v4_stability.csv)

扫面区间包括：

- `top_k in {4,5,6,8}`
- `rebalance_every in {2,3,4,5}`
- `min_hold_bars in {8,10,12}`

这一轮最大的收获不是“找到更高收益的一组参数”，而是更清楚地看到了参数地形：

- 当前基线不是唯一峰值
- `min_hold_bars=12` 明显优于 `10`
- `top_k=5` 的整体稳定性明显优于 `4/6/8`

这意味着策略方向本身是成立的，并不是只在一个参数点偶然有效。

## 7. 从基线走向候选 A/B

在邻域扫面后，正式回测了两组候选组合。

### 候选 A

- `top_k=5`
- `min_hold_bars=12`

结果目录： [model_score_backtest_walk_forward_industry_v4_candidate_a](/Users/yongqiuwu/works/github/Trade/results/model_score_backtest_walk_forward_industry_v4_candidate_a)

核心指标：

| 指标 | 数值 |
| --- | ---: |
| 年化收益 | `60.79%` |
| 最大回撤 | `11.31%` |
| Sharpe | `2.08` |

### 候选 B

- `top_k=4`
- `min_hold_bars=12`

结果目录： [model_score_backtest_walk_forward_industry_v4_candidate_b](/Users/yongqiuwu/works/github/Trade/results/model_score_backtest_walk_forward_industry_v4_candidate_b)

核心指标：

| 指标 | 数值 |
| --- | ---: |
| 年化收益 | `81.42%` |
| 最大回撤 | `16.47%` |
| Sharpe | `2.42` |

### 为什么最终没选 B

候选 B 从收益和 Sharpe 上看更强，但它更像一版进攻型组合：回撤更大，月度表现也更集中。相比之下，候选 A 的收益已经明显优于原始基线，同时回撤还保持在一个更适合作为 `v1` 默认版本的区间。

所以这里的决策不是“谁数值最大就选谁”，而是：

- 候选 B 作为高收益观察组合保留
- 候选 A 作为更稳妥的默认 `v1` 候选继续推进

## 8. 把风险约束真正接进策略层

如果一版策略要从“实验结果”变成“版本结果”，只调组合参数还不够，至少要接入一层基础风险约束。

因此在组合策略层补了两类能力：

1. 流动性过滤：`min_daily_amount`
2. 行业约束：`max_names_per_industry`

相关代码位于：

- [score_strategy.py](/Users/yongqiuwu/works/github/Trade/src/ashare_backtest/research/score_strategy.py)
- [main.py](/Users/yongqiuwu/works/github/Trade/src/ashare_backtest/cli/main.py)
- [sweep.py](/Users/yongqiuwu/works/github/Trade/src/ashare_backtest/research/sweep.py)
- [research_config.py](/Users/yongqiuwu/works/github/Trade/src/ashare_backtest/cli/research_config.py)

这两项都做成了可配置参数，默认关闭，避免破坏历史结果，也方便后续继续做分层测试。

## 9. 风险约束不是“加上去”就行，而是要单独验证

### 9.1 行业约束验证

先单独测试：

- `max_names_per_industry=2`

在候选 A 上的结果变为：

| 指标 | 数值 |
| --- | ---: |
| 年化收益 | `59.60%` |
| 最大回撤 | `11.31%` |
| Sharpe | `2.01` |

结果目录： [model_score_backtest_walk_forward_industry_v4_candidate_a_industry_only](/Users/yongqiuwu/works/github/Trade/results/model_score_backtest_walk_forward_industry_v4_candidate_a_industry_only)

结论很直接：行业上限 2 对收益影响很小，但能提供一层明确的集中度控制，因此这项约束被接受为 `v1` 默认设置。

### 9.2 流动性过滤验证

流动性过滤没有直接拍脑袋定，而是做了一轮扫描。

测试时固定组合为：

- `top_k=5`
- `min_hold_bars=12`
- `max_names_per_industry=2`

然后测试多档 `min_daily_amount`。关键结果如下：

| min_daily_amount | 年化收益 | 最大回撤 | Sharpe |
| --- | ---: | ---: | ---: |
| 0 | `59.60%` | `11.31%` | `2.01` |
| 10000 | `54.73%` | `11.32%` | `1.90` |
| 20000 | `54.73%` | `11.32%` | `1.90` |
| 30000 | `41.09%` | `10.86%` | `1.43` |
| 40000 | `42.73%` | `10.90%` | `1.47` |
| 50000 | `34.82%` | `10.76%` | `1.24` |
| 60000 | `27.73%` | `16.51%` | `1.05` |
| 80000 | `14.74%` | `15.44%` | `0.59` |
| 100000 | `32.74%` | `21.19%` | `1.18` |

这个结果说明一件很现实的事：当前这版策略对低流动性股票的暴露是收益来源的一部分。也就是说，流动性过滤能力必须保留，但现阶段不应该默认启用，否则会明显削弱这版 `v1` 的表现。

## 10. 最终怎么收口为 V1

在经历了基线复现、稳健性检查、候选 A/B 对比和风险约束调试后，最终确认的 `v1` 配置为：

- 标签：`industry_excess_fwd_return_5`
- 训练窗口：`12` 个月 walk-forward
- 组合参数：
  - `top_k=5`
  - `rebalance_every=3`
  - `min_hold_bars=12`
  - `keep_buffer=2`
  - `min_turnover_names=3`
- 风险约束：
  - `max_names_per_industry=2`
  - `min_daily_amount=0`

正式配置文件：

- [research_industry_v4_v1.toml](/Users/yongqiuwu/works/github/Trade/configs/research_industry_v4_v1.toml)

README 也已经切换到这版推荐配置：

- [README.md](/Users/yongqiuwu/works/github/Trade/README.md)

如果只用一句话概括这次收口，可以写成：

> 用更长持有期替代更激进的调仓，用行业约束替代裸暴露扩张，最终得到一版收益仍强、结构更清晰的研究型 `v1`。

## 11. 这版 V1 到底意味着什么

这版策略已经具备：

- 一套稳定的数据与训练链路
- 一套可重复运行的模型选股流程
- 一组经过邻域验证的组合参数
- 一层基础的行业风险约束

但它依然只是研究型 `v1`，而不是最终的实盘策略。主要原因是：

- 月度收益分布仍然偏集中
- 月度 IC 仍然偏弱
- 更真实的执行层约束还没有完全接入
- 更长区间和更多市场状态下的稳健性验证还不充分

换句话说，这一版解决的是“有没有形成第一版策略”的问题，而不是“这版策略是不是已经成熟到可以直接上线”的问题。

## 12. 下一步应该做什么

如果继续推进，优先级建议如下：

1. 做更长时间区间的 walk-forward 与回测验证
2. 增加更真实的容量和成交约束
3. 做行业暴露、风格暴露和收益来源拆解
4. 把 `v1` 以及后续版本的实验结果系统化归档

未来如果继续做 `v2`，最值得优先验证的方向不是继续裸调 `top_k` 或 `min_hold_bars`，而是：

- 提升底层预测信号稳定性
- 让风险约束更真实但不过度伤害收益
- 把策略收益来源从“回测结果”拆解成“可解释的机制”

对应的后续推进计划见：

- [strategy-v2-roadmap.md](/Users/yongqiuwu/works/github/Trade/docs/strategy-v2-roadmap.md)

## 13. 一键复现命令

```bash
PYTHONPATH=src python3 -m ashare_backtest.cli.main run-research-config configs/research_industry_v4_v1.toml
```
