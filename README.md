# A 股低频策略回测工具

这是一个面向个人使用的 A 股低频策略回测项目，目标是用最小可维护系统完成以下闭环：

- 用“受限 Python”编写策略脚本
- 将策略脚本校验并注册进策略库
- 选择股票池、时间范围、交易成本参数运行日线回测
- 输出基础绩效指标、交易记录和净值序列

当前仓库处于 MVP 骨架阶段，优先解决以下问题：

- 明确策略脚本边界，避免任意 Python 执行
- 把数据读取、调仓调度、成交模拟、持仓管理、绩效统计收口到回测引擎
- 让一个人可以逐步扩展，而不是一次性做成通用量化平台

## MVP 边界

详细设计见 [docs/mvp.md](/Users/yongqiuwu/works/github/Trade/docs/mvp.md)。

第一阶段只覆盖：

- 市场：A 股
- 频率：日线
- 策略类型：多头股票
- 调仓：每 2 到 3 个交易日为主，也支持更稀疏日频调仓
- 成交：按下一交易日开盘或当日收盘的简化成交模型扩展
- 风控：基础仓位约束、停牌与涨跌停不可成交、手续费和滑点

第一阶段明确不做：

- 分钟级、逐笔、盘口回测
- 融资融券、期货、期权
- 分布式任务调度
- 多租户、Web 平台、权限系统
- 任意 Python 研究环境

## 目录

- `docs/`: MVP 边界、架构和流程说明
- `storage/`: 标准化后的本地市场数据
- `src/ashare_backtest/`: 核心代码
- `strategies/`: 已通过协议约束的策略脚本
- `examples/`: 策略模板和示例
- `tests/`: 后续测试

## 快速开始

```bash
python -m ashare_backtest.cli.main validate strategies/buy_and_hold.py
python -m ashare_backtest.cli.main show-template
python -m ashare_backtest.cli.main import-sqlite /path/to/source.db
python -m ashare_backtest.cli.main run-backtest strategies/three_day_momentum.py --storage-root storage --start-date 2024-02-01 --end-date 2024-12-31 --universe 600519.SH,000001.SZ,300750.SZ --output-dir results/three_day_momentum
python -m ashare_backtest.cli.main run-config configs/three_day_momentum.toml
python -m ashare_backtest.cli.main build-factors --storage-root storage --symbols 600519.SH,000001.SZ,300750.SZ --start-date 2024-02-01 --end-date 2024-12-31
python -m ashare_backtest.cli.main run-research-config configs/research_excess_v3.toml
```

后续会补充：

- 数据适配层
- 真正的回测循环
- 成交撮合与绩效统计
- 策略生成与入库命令

## 当前推荐研究链路

当前已验证过一条更有效的研究配置：

- 因子面板：`industry_v4`
- 标签：`industry_excess_fwd_return_5`
- 训练：按月 walk-forward，训练窗口 12 个月
- 组合：`top_k=6`、`rebalance_every=5`、`min_hold_bars=8`、`keep_buffer=2`、`min_turnover_names=3`
- 风险约束：`max_names_per_industry=2`

推荐直接通过 [research_industry_v4_v1_1.toml](/Users/yongqiuwu/works/github/Trade/configs/research_industry_v4_v1_1.toml) 复现。
