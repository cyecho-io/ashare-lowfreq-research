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
python -m pip install -e ".[dev]"
```

安装后会暴露两个命令：

- `ashare-backtest`
- `ashare-backtest-web`

建议先复制环境变量模板：

```bash
cp .env.example .env
```

只有在使用 Tushare 同步真实数据时，才需要填写 `TUSHARE_TOKEN`。

## 快速体验

如果你只是想在 clone 之后先体验一遍功能，而不想先准备自己的行情数据，可以直接使用仓库里内置的 `storage/demo/` 极小样例数据。

先初始化环境：

```bash
bash scripts/bootstrap_demo.sh
source .venv/bin/activate
```

运行内置 demo 研究配置：

```bash
ashare-backtest run-research-config configs/demo_research.toml
```

启动本地 Web 控制台：

```bash
ashare-backtest-web
```

然后打开 `http://127.0.0.1:8765`。

这个 demo 路径会给你：

- 一套已跟踪的极小 A 股样例数据
- 一个可直接运行的研究配置
- 一份输出到 `results/demo_backtest` 的回测结果
- 一个可查看数据状态、回测工件和模拟视图的本地 Web 界面

如果后面要切到你自己的本地数据，只需要把配置里的 `storage root` 换掉，再走下面的完整工作流即可。

## 开源发布前建议

如果目标是让别人 clone 后尽快体验，建议至少保证这条最短路径可用：

- 用 `python -m pip install -e ".[dev]"` 完成安装
- 从 `.env.example` 复制出本地 `.env`
- 从 [`examples/demo_research_config.toml`](/Users/yongqiuwu/works/github/Trade/examples/demo_research_config.toml) 开始改自己的研究配置
- 用 [`configs/demo_research.toml`](/Users/yongqiuwu/works/github/Trade/configs/demo_research.toml) 直接跑仓库里跟踪的 `storage/demo/` 极小样例数据
- 如果希望无需 Tushare 账号就能体验 Web 控制台，额外提供一份可追踪的小型 demo 数据

当前仓库的大部分 CLI / Web 流程仍默认依赖 `storage/` 下已有本地数据。现在仓库已经提供了一个可提交的 `storage/demo/` 最小样例数据集，方便对外演示。

也可以先用这个初始化脚本快速搭本地环境：

```bash
bash scripts/bootstrap_demo.sh
```

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

如果只是做模板复制或对外示例，可以先从这个示例配置开始：

```bash
cp examples/demo_research_config.toml configs/demo_research.toml
```

如果只是验证公开仓库是否能跑通，可以直接运行内置 demo 配置：

```bash
ashare-backtest run-research-config configs/demo_research.toml
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
- [`CONTRIBUTING.md`](/Users/yongqiuwu/works/github/Trade/CONTRIBUTING.md)
- [`storage/demo/README.md`](/Users/yongqiuwu/works/github/Trade/storage/demo/README.md)

## 测试

```bash
python3 -m pytest
```

首次运行前建议先安装本包，否则测试阶段无法导入 `ashare_backtest`：

```bash
python -m pip install -e ".[dev]"
python3 -m pytest
```
