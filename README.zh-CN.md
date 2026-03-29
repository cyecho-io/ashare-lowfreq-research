# A 股低频策略回测工具

[English README](README.md)

这是一个面向个人研究使用的 A 股低频回测与盘前准备工具，目标是把数据同步、因子构建、模型训练、分数回测和模拟执行收敛到一条可维护的本地工作流里。

当前仓库重点解决的是：

- 同步并标准化本地 A 股日线数据
- 基于 universe 构建 factor snapshot 并训练分数模型
- 运行 walk-forward / as-of-date 打分与分数驱动回测
- 生成盘前参考、latest 状态和模拟账户计划
- 通过本地 Web 控制台查看数据状态、回测结果和模拟执行历史

这个项目是有边界的，不打算扩展成通用量化平台。

## 当前边界

- 市场：A 股
- 频率：日线
- 策略类型：多头股票组合
- 研究链路：factor build -> model training -> walk-forward / latest inference -> score backtest
- 执行约束：手续费、印花税、滑点、成交参与率上限、挂单保留天数
- 使用方式：CLI + 本地 Web 控制台

当前明确不做：

- 分钟级、逐笔、盘口回测
- 衍生品、融资融券、多资产组合
- 分布式调度和多租户系统
- 任意无约束 Python 策略执行

## 目录

- `src/ashare_backtest/`：核心代码
- `src/ashare_backtest/web/`：数据看板、回测控制台、模拟成交台
- `configs/`：研究和回测配置
- `research/`：本地产生的 factor、model 和 latest 工件
- `storage/`：标准化 parquet 数据和源 SQLite 数据库
- `strategies/`：受协议约束的策略脚本
- `docs/`：设计文档、研究笔记和 runbook
- `tests/`：回归测试

`results/`、`research/factors/`、`research/models/` 下的产物默认视为本地生成文件，已经加入 `.gitignore`，不再作为仓库源码的一部分长期追踪。

## 安装

需要 Python 3.11+。

```bash
python -m pip install -e .
```

安装后会暴露两个命令：

- `ashare-backtest`
- `ashare-backtest-web`

## 快速开始

把本地 SQLite 行情导入 parquet 存储：

```bash
ashare-backtest import-sqlite storage/source/ashare_arena_sync.db --storage-root storage
```

基于指定 universe 构建 factor snapshot：

```bash
ashare-backtest build-factors \
  --storage-root storage \
  --universe-name tradable_core \
  --start-date 2024-02-01 \
  --as-of-date 2024-12-31
```

运行研究配置：

```bash
ashare-backtest run-research-config configs/research_industry_v4_v1_1.toml
```

基于模型分数执行回测：

```bash
ashare-backtest run-model-backtest \
  --scores-path research/models/walk_forward_scores.parquet \
  --storage-root storage \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --output-dir results/model_score_backtest
```

## 数据同步

把 Tushare 日线数据同步到项目源 SQLite：

```bash
ashare-backtest sync-tushare-sqlite --start 20240101 --end 20260331
```

把基准指数历史同步到 parquet：

```bash
ashare-backtest sync-tushare-benchmark --symbol 000300.SH --start 20240101 --end 20260331
```

如果不显式传 `--token`，默认读取环境变量 `TUSHARE_TOKEN`。

## Web 控制台

启动本地控制台：

```bash
ashare-backtest-web
```

当前控制台包含三个主要页面：

- `/`：数据看板，查看交易日历热力图、SQLite 数据源摘要和策略数量
- `/backtest`：回测控制台，选择配置、分数文件和区间后直接发起回测
- `/simulation`：模拟成交台，创建模拟账户、查看账户状态、执行历史和状态演化

模拟页里 `strategy_state.json`、`decision_log.csv` 的 `decision_reason` 常见值包括：

- `initial_entry`
- `empty_universe`
- `insufficient_history`
- `rebalance_schedule`
- `missing_scores`
- `model_score_schedule`

排查模拟结果时，可以优先看 `summary.decision_reason`，判断当前是正常调仓、沿用旧仓位，还是数据准备存在缺口。

## 当前推荐研究配置

当前推荐使用 [`configs/research_industry_v4_v1_1.toml`](/Users/yongqiuwu/works/github/Trade/configs/research_industry_v4_v1_1.toml)：

- 因子面板：`industry_v4`
- 标签：`industry_excess_fwd_return_5`
- 训练：按月 walk-forward，训练窗口 12 个月
- 组合：`top_k=6`、`rebalance_every=5`、`min_hold_bars=8`、`keep_buffer=2`
- 换手控制：`min_turnover_names=3`
- 行业约束：`max_names_per_industry=2`

默认会先在 `universe` 层做股票池门禁，再让因子构建读取指定 universe。导入后通常会维护两个快照池：

- `all_active`：当前 active 股票
- `tradable_core`：当前 active、非 ST、上市满 120 天，并满足基本可交易性和流动性过滤

## 相关文档

- [`docs/mvp.md`](/Users/yongqiuwu/works/github/Trade/docs/mvp.md)
- [`docs/research-pipeline.md`](/Users/yongqiuwu/works/github/Trade/docs/research-pipeline.md)
- [`docs/strategy-v1-1-premarket-runbook.md`](/Users/yongqiuwu/works/github/Trade/docs/strategy-v1-1-premarket-runbook.md)
- [`docs/strategy-v1-1-latest-active-data-source.md`](/Users/yongqiuwu/works/github/Trade/docs/strategy-v1-1-latest-active-data-source.md)
- [`docs/strategy-v2-live-readiness-checklist.md`](/Users/yongqiuwu/works/github/Trade/docs/strategy-v2-live-readiness-checklist.md)
- [`docs/strategy-v2-roadmap.md`](/Users/yongqiuwu/works/github/Trade/docs/strategy-v2-roadmap.md)

## 测试

```bash
python3 -m pytest
```
