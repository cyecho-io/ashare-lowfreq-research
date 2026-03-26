# A 股低频策略回测工具

[English README](README.md)

这是一个面向个人使用的 A 股研究与回测工具，目标是围绕一条可维护、可复现的闭环工作流来做：

- 同步并标准化本地 A 股市场数据
- 构建因子面板并训练打分模型
- 基于分数输出运行带执行约束的回测
- 通过轻量 Web 控制台查看结果
- 生成最新选股、盘前参考和策略状态文件

这个仓库是有边界的，不打算扩展成通用量化平台。

## 当前边界

- 市场：A 股
- 频率：日线
- 策略类型：多头股票组合
- 研究链路：因子构建 -> 模型训练 -> walk-forward / latest inference -> 分数驱动回测
- 执行约束：手续费、印花税、滑点、成交参与率上限、挂单保留天数
- 使用方式：CLI + 本地 Web 回测控制台

当前明确不做：

- 分钟级、逐笔、盘口回测
- 衍生品、融资融券、多资产组合
- 分布式调度和多租户系统
- 任意无约束 Python 策略执行

## 目录

- `src/ashare_backtest/`: 核心代码
- `src/ashare_backtest/web/`: 本地回测与模拟盘控制台
- `configs/`: 回测和研究配置
- `research/`: 因子面板、模型输出和 latest 工件
- `storage/`: 标准化 parquet 数据和源 SQLite 数据库
- `strategies/`: 受协议约束的策略脚本
- `docs/`: 设计文档、研究笔记和 runbook
- `tests/`: 回归测试

## 安装

需要 Python 3.11+。

```bash
python -m pip install -e .
```

安装后会暴露两个命令：

- `ashare-backtest`
- `ashare-backtest-web`

## 快速开始

校验策略脚本：

```bash
ashare-backtest validate strategies/buy_and_hold.py
```

把本地 SQLite 行情导入 parquet 存储：

```bash
ashare-backtest import-sqlite storage/source/ashare_arena_sync.db --storage-root storage
```

基于 universe 构建因子面板：

```bash
ashare-backtest build-factors \
  --storage-root storage \
  --universe-name tradable_core \
  --start-date 2024-02-01 \
  --end-date 2024-12-31
```

运行研究配置：

```bash
ashare-backtest run-research-config configs/research_industry_v4_v1_1.toml
```

基于模型分数执行回测：

```bash
ashare-backtest run-model-backtest \
  --scores-path research/models/latest_scores.parquet \
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

启动本地回测控制台：

```bash
ashare-backtest-web
```

当前控制台支持：

- 从预设配置发起回测
- 浏览结果目录和摘要指标
- 查看带基准对比的权益曲线
- 筛选和检查交易记录
- 查看策略最新信号与模拟盘视图

## 当前推荐研究配置

当前推荐直接使用 [`configs/research_industry_v4_v1_1.toml`](/Users/yongqiuwu/works/github/Trade/configs/research_industry_v4_v1_1.toml)：

- 因子面板：`industry_v4`
- 标签：`industry_excess_fwd_return_5`
- 训练：按月 walk-forward，训练窗口 12 个月
- 组合：`top_k=6`、`rebalance_every=5`、`min_hold_bars=8`、`keep_buffer=2`
- 换手控制：`min_turnover_names=3`
- 行业约束：`max_names_per_industry=2`

当前默认把股票池门禁前置到 `universe` 层，再让因子构建读取指定 universe。导入后会生成两个快照池：

- `all_active`：当前 active 股票
- `tradable_core`：当前 active、非 ST、上市满 120 天，并满足基本可交易性和流动性过滤

## 相关文档

- [`docs/mvp.md`](/Users/yongqiuwu/works/github/Trade/docs/mvp.md)
- [`docs/research-pipeline.md`](/Users/yongqiuwu/works/github/Trade/docs/research-pipeline.md)
- [`docs/strategy-v2-live-readiness-checklist.md`](/Users/yongqiuwu/works/github/Trade/docs/strategy-v2-live-readiness-checklist.md)
- [`docs/strategy-v2-roadmap.md`](/Users/yongqiuwu/works/github/Trade/docs/strategy-v2-roadmap.md)

## 测试

```bash
python3 -m pytest
```
