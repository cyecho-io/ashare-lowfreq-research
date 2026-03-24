# 从 V1 到 V1.1：修复调仓实现后的策略重建记录

这份文档记录 `v1` 之后最关键的一次版本修正：在组合规则诊断中发现 `rebalance_every` 没有真正生效，导致历史参数扫面和部分收益结论被污染。修复该问题后，策略基准需要重新建立，因此形成了修复后的正式版本 `v1.1`。

## 摘要

`v1.1` 不是一次常规调参，而是一次“实现修复后的策略重建”。

修复后重新选出的默认基准为：

- 标签：`industry_excess_fwd_return_5`
- 训练：`12` 个月 walk-forward
- 评估窗口：`2025-01` 到 `2026-02`
- 组合：`top_k=6`、`rebalance_every=5`、`min_hold_bars=8`
- 风险约束：`max_names_per_industry=2`

正式配置文件：

- [research_industry_v4_v1_1.toml](/Users/yongqiuwu/works/github/Trade/configs/research_industry_v4_v1_1.toml)

对应结果目录：

- [model_score_backtest_walk_forward_industry_v4_postfix_best_longer](/Users/yongqiuwu/works/github/Trade/results/model_score_backtest_walk_forward_industry_v4_postfix_best_longer)

核心指标：

| 指标 | 数值 |
| --- | ---: |
| 年化收益 | `81.17%` |
| 最大回撤 | `18.15%` |
| Sharpe | `2.26` |

## 1. 为什么需要 V1.1

`v1` 阶段做了完整的策略构建，但在进入 `v2` 早期诊断时，发现一个关键实现问题：

- `rebalance_every` 虽然在参数里存在
- 但在策略实现里没有真正对调仓频率产生作用

直接现象是：

- 不同 `rebalance_every` 的回测结果几乎完全一致
- 历史 sweep 对调仓频率的比较没有意义

这意味着：

- 修复前的很多参数结论不再可信
- 修复前的 `v1` 应被视为“历史研究记录”
- 后续必须以修复后的重扫结果重新建立基准

## 2. 修复了什么

修复点位于：

- [score_strategy.py](/Users/yongqiuwu/works/github/Trade/src/ashare_backtest/research/score_strategy.py)

修复前的问题本质上是：

- 调仓判断依赖截断后的固定长度 `lookback` 历史
- 导致 `rebalance_every` 逻辑失真

修复后，调仓频率终于开始真正影响结果。

快速验证结果：

| 参数 | 年化收益 | 最大回撤 | Sharpe |
| --- | ---: | ---: | ---: |
| `rebalance_every=2` | `29.18%` | `21.26%` | `1.09` |
| `rebalance_every=5` | `34.25%` | `18.43%` | `1.29` |

这一步非常关键，因为它说明：

> 修复不是“代码洁癖”，而是直接改写了策略研究结论。

## 3. 修复后重建基准的方法

修复完成后，没有直接沿用旧 `v1` 配置，而是重新做了两轮 sweep：

### 短窗口重扫

窗口：

- `2025-07` 到 `2026-02`

输出文件：

- [model_sweep_industry_v4_postfix_v1window.csv](/Users/yongqiuwu/works/github/Trade/research/models/model_sweep_industry_v4_postfix_v1window.csv)

最优组合：

- `top_k=4`
- `rebalance_every=3`
- `min_hold_bars=10`

结果目录：

- [model_score_backtest_walk_forward_industry_v4_postfix_best_v1window](/Users/yongqiuwu/works/github/Trade/results/model_score_backtest_walk_forward_industry_v4_postfix_best_v1window)

核心指标：

| 指标 | 数值 |
| --- | ---: |
| 年化收益 | `154.46%` |
| 最大回撤 | `11.12%` |
| Sharpe | `3.31` |

### 长窗口重扫

窗口：

- `2025-01` 到 `2026-02`

输出文件：

- [model_sweep_industry_v4_postfix_longer.csv](/Users/yongqiuwu/works/github/Trade/research/models/model_sweep_industry_v4_postfix_longer.csv)

最优组合：

- `top_k=6`
- `rebalance_every=5`
- `min_hold_bars=8`

结果目录：

- [model_score_backtest_walk_forward_industry_v4_postfix_best_longer](/Users/yongqiuwu/works/github/Trade/results/model_score_backtest_walk_forward_industry_v4_postfix_best_longer)

核心指标：

| 指标 | 数值 |
| --- | ---: |
| 年化收益 | `81.17%` |
| 最大回撤 | `18.15%` |
| Sharpe | `2.26` |

## 4. 为什么最终选长窗口最优作为 V1.1

虽然短窗口最优看起来更强，但它的问题也很明显：

- 样本更短
- 更容易被局部行情放大
- 不能作为修复后正式基准的唯一依据

相比之下，长窗口最优更适合作为 `v1.1` 的默认基准，因为它：

- 使用更长样本期
- 覆盖更多市场状态
- 对调仓频率的依赖已经是真实生效的
- 结果虽然不如短窗口极致，但更适合作为后续版本迭代起点

换句话说：

- 短窗口最优更像“局部最强点”
- 长窗口最优更像“修复后更可信的正式基准”

## 5. V1.1 的核心变化

和修复前 `v1` 相比，`v1.1` 最重要的变化有三点。

### 1. 调仓频率真正进入了参数空间

现在 `rebalance_every` 不再是名义参数，而是真正能改变：

- 交易次数
- 换手率
- 收益路径
- 回撤表现

### 2. 默认组合更偏低频、更分散

修复后长窗口最优不再是更激进的短频组合，而是：

- `top_k=6`
- `rebalance_every=5`
- `min_hold_bars=8`

这说明在更真实的调仓逻辑下，策略更适合：

- 持有更多股票
- 调仓更慢
- 让信号有更充分的实现周期

### 3. 版本叙事从“历史最优”转向“修复后可信基准”

`v1.1` 的核心意义不只是收益数字变化，而是版本管理方式发生了变化：

- `v1` 保留为原始研究构建记录
- `v1.1` 成为修复后正式基准
- 后续 `v2` 优化应默认建立在 `v1.1` 上

## 6. 当前对 V1.1 的判断

`v1.1` 比修复前 `v1` 更可信，但它仍然不是实盘策略。

它当前的定位更准确地说是：

- 一版修复后、重新校准过的研究基准
- 一版可以继续承接 `v2` 优化的正式起点

它解决了一个此前非常严重的问题：

- 现在我们终于可以相信参数比较本身是有意义的

但它仍然保留了后续要继续解决的问题：

- IC 与收益映射关系仍然不稳定
- 更真实的容量和执行约束还没完整接入
- 收益来源和风险来源还需要继续拆解

## 7. 当前推荐配置

当前默认应使用：

- [research_industry_v4_v1_1.toml](/Users/yongqiuwu/works/github/Trade/configs/research_industry_v4_v1_1.toml)

一键复现命令：

```bash
PYTHONPATH=src python3 -m ashare_backtest.cli.main run-research-config configs/research_industry_v4_v1_1.toml
```

## 8. 后续建议

从现在开始，后续路线建议这样切换：

1. `v1` 作为历史构建记录保留  
2. `v1.1` 作为当前正式基准  
3. `v2` 的优化默认都基于 `v1.1` 展开  

如果继续推进，下一步最有价值的工作不是再做一轮大范围盲扫，而是：

- 在 `v1.1` 基础上继续做 IC 稳定性优化
- 分析为什么低频更分散组合在修复后更优
- 继续做收益来源和风险来源拆解

## 一句话总结

`v1.1` 的价值不只是“换了一组更好的参数”，而是：

> 在修复调仓实现缺陷之后，重新建立了一版可信的正式基准，让后续所有优化终于有了可靠起点。
