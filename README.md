# A 股低频策略回测工具

[English README](README.en.md)

这是一个面向个人研究使用的 A 股低频回测与盘前准备工具，目标是把数据同步、因子构建、模型训练、分数回测和模拟执行收敛到一条可维护的本地工作流里。

当前仓库重点解决的是：

- 同步并标准化本地 A 股日线数据
- 基于 universe 构建 factor snapshot，并支持 native / qlib 双研究链路
- 运行 walk-forward / as-of-date 打分与分数驱动回测
- 生成盘前参考、latest 状态和模拟账户计划
- 通过本地 Web 控制台查看数据状态、研究记录、回测结果和模拟执行历史

这个项目是有边界的，不打算扩展成通用量化平台。

## 当前边界

- 市场：A 股
- 频率：日线
- 策略类型：多头股票组合
- 研究链路：factor build -> model training -> walk-forward / latest inference -> score backtest
- 执行约束：手续费、印花税、滑点、成交参与率上限、挂单保留天数
- 使用方式：CLI + 本地 Web 控制台 + 工作区切换

当前明确不做：

- 分钟级、逐笔、盘口回测
- 衍生品、融资融券、多资产组合
- 分布式调度和多租户系统
- 任意无约束 Python 策略执行

## 目录

- `src/ashare_backtest/`：核心代码
- `src/ashare_backtest/web/`：数据看板、回测控制台、模型研究台、模拟成交台
- `src/ashare_backtest/qlib_integration/`：Qlib research path、特征选择、训练与分数导出
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

如果你准备启用 Qlib 研究路径，再额外安装：

```bash
python -m pip install -e ".[qlib]"
```

Qlib 集成说明见：[docs/qlib-integration.md](/Users/yongqiuwu/works/github/Trade/docs/qlib-integration.md)

## 双链路与工作区

当前仓库已经从单一 research path 扩展为两条并行研究链路：

- `native`：继续使用项目内置的因子构建、LightGBM 训练和分析逻辑
- `qlib`：使用 Qlib 表达式、数据接口和训练流程生成兼容的 `scores.parquet`

两条链路共享下游回测、分层分析、盘前参考和模拟执行逻辑，但会通过 `workspace` 明确隔离生成结果，避免产物混在一起。

当前约定的工作区产物目录包括：

- `research/native/models/` 与 `research/qlib/models/`
- `results/native/` 与 `results/qlib/`

Web 控制台顶部现在带有全局 `工作区` 切换器，`/backtest`、`/research`、`/simulation`、`/` 都会跟随当前 workspace 刷新对应的配置列表、研究记录、分数文件和结果摘要。

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

然后打开 `http://127.0.0.1:8888`。

这个 demo 路径会给你：

- 一套已跟踪的极小 A 股样例数据
- 一个可直接运行的研究配置
- 一条会在本地生成因子、分数和回测结果的完整示例链路
- 一份输出到 `results/demo_backtest` 的回测结果
- 一个可查看数据状态、研究记录、回测工件和模拟视图的本地 Web 界面

在打开 `/backtest` 页面之前，请先执行一次 `ashare-backtest run-research-config configs/demo_research.toml`，让 demo 分数文件先在本地生成出来。

如果后面要切到你自己的本地数据，只需要把配置里的 `storage root` 换掉，再走下面的完整工作流即可。

## 快速开始

如果你要在自己的数据上跑完整工作流，建议先把本地数据底座准备好。

推荐顺序如下：

1. 安装项目并从 `.env.example` 复制出 `.env`
2. 填写 `TUSHARE_TOKEN`
3. 先用 Tushare 同步出第一版本地 SQLite 行情库
4. 再把 SQLite 导入成 Parquet 存储
5. 最后再跑因子、研究配置、回测或 Web 控制台

### 准备第一版 SQLite 行情数据

仓库里的本地数据分成两层：

- `storage/source/`：可写的源 SQLite 数据库
- `storage/parquet/`：研究和回测使用的 Parquet 快照

第一次准备本地 SQLite 行情库时，可以运行：

```bash
ashare-backtest sync-tushare-sqlite \
  --sqlite-path storage/source/ashare_arena_sync.db \
  --start 20240101 \
  --end 20260331
```

这个命令会：

- 在不存在时创建 `storage/source/ashare_arena_sync.db`
- 同步交易日历
- 同步股票主数据
- 同步日线行情到 SQLite
- 刷新派生的 `all_active` 股票池

如果还希望补齐 Web 和报表里常用的基准指数历史，可以继续执行：

```bash
ashare-backtest sync-tushare-benchmark \
  --symbol 000300.SH \
  --start 20240101 \
  --end 20260331
```

### 把 SQLite 导入为 Parquet

SQLite 准备好之后，再导入到 Parquet 分析层：

```bash
ashare-backtest import-sqlite storage/source/ashare_arena_sync.db --storage-root storage
```

这一步会在 `storage/parquet/` 下生成项目后续流程所需的标准文件，并刷新 `storage/catalog.json`。

### 运行研究流程

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

如果你要新建自己的配置，可以先从模板开始：

```bash
cp examples/demo_research_config.toml configs/demo_research.toml
```

如果你准备通过 CLI 或 Web 的 `/research` 页面运行 Qlib 研究任务，可以直接使用仓库新增的 [`configs/research_industry_v4_v1_1_qlib.toml`](/Users/yongqiuwu/works/github/Trade/configs/research_industry_v4_v1_1_qlib.toml)，或者在现有研究配置里补充一个可选的 `[qlib]` 段，例如：

```toml
[qlib]
provider_uri = "~/.qlib/qlib_data/cn_data"
region = "cn"
market = "csi300"
model_name = "lgbm"
config_id = "qlib_smoke"
```

然后在页面里把 `研究后端` 切到 `qlib` 即可。相关说明见：[docs/qlib-integration.md](/Users/yongqiuwu/works/github/Trade/docs/qlib-integration.md)

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

### 启动 Web 控制台

当 `storage/` 下已经有导入后的数据，并且你至少跑过一次研究或回测后，可以启动本地 Web 控制台：

```bash
ashare-backtest-web
```

默认访问地址是 `http://127.0.0.1:8888`。

当前控制台包含四个主要页面，并统一支持 workspace 切换：

- `/`：数据看板，查看交易日历热力图、SQLite 数据源摘要、策略数量和最近运行概览
- `/backtest`：回测控制台，选择配置、分数文件和区间后直接发起回测
- `/research`：模型研究台，直接编辑 `configs/*.toml`、执行研究流水线、查看日志、产物路径和关键指标
- `/simulation`：模拟成交台，创建模拟账户、查看账户状态、执行历史和状态演化

`/research` 页面当前已经支持选择 `native` 或 `qlib` 研究后端；研究记录、分数文件、回测结果和模拟执行结果都会按 `workspace` / `backend` 带出 provenance 信息。

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

## Qlib Research Path

当前仓库已经增加了可选的 Qlib research path，用于生成与现有下游兼容的 `scores.parquet`，但不会替换现有 native research path，也不会替换当前回测执行层。

当前可用命令包括：

- `ashare-backtest qlib-train-walk-forward`
- `ashare-backtest qlib-train-as-of-date`
- `ashare-backtest qlib-train-single-date`

这条链路当前已经补齐了几类能力：

- 可选安装 `.[qlib]`，不污染默认依赖
- 基于 Qlib provider 的特征选择、表达式解析和数据读取
- 将 Qlib 训练结果导出为兼容现有下游的 `scores.parquet`
- 在 Web 模型研究台里直接触发 `backend=qlib` 的研究任务
- 通过 workspace 把 native / qlib 的研究记录、分数文件和结果目录隔离开

完整环境准备、数据目录要求和命令示例见：[docs/qlib-integration.md](/Users/yongqiuwu/works/github/Trade/docs/qlib-integration.md)

## Web 控制台

启动本地控制台：

```bash
ashare-backtest-web
```

当前控制台包含四个主要页面：

- `/`：数据看板，查看交易日历热力图、SQLite 数据源摘要、策略数量和最近运行概览
- `/backtest`：回测控制台，选择配置、分数文件和区间后直接发起回测
- `/research`：模型研究台，编辑研究配置、运行完整研究流水线并查看执行日志
- `/simulation`：模拟成交台，创建模拟账户、查看账户状态、执行历史和状态演化

控制台顶部的 `工作区` 切换器会把页面数据切到当前 workspace，对应读取各自的：

- 研究配置候选
- 研究记录
- 分数文件与 lineage
- 回测结果
- 模拟计划与执行结果

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
