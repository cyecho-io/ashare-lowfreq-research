# Strategy v1.1 Premarket Runbook

这份文档用于在交易日开盘前，基于 `v1.1` 策略生成最新选股和调仓参考。

当前默认股票池：

- `tradable_core`

当前默认策略参数来自：

- [research_industry_v4_v1_1.toml](/Users/yongqiuwu/works/github/Trade/configs/research_industry_v4_v1_1.toml)

## 目标

以 `T` 日收盘后的完整数据为输入，生成 `T+1` 日开盘前的调仓清单。

例如：

- `2026-03-25` 收盘后更新数据
- 生成 `2026-03-26` 开盘前参考

## 一键脚本

如果你不想手动逐条执行命令，可以直接运行：

```bash
./scripts/run_v1_1_premarket.sh
```

这个脚本会：

- 自动加载项目根目录 `.env` 里的环境变量
- 检查 SQLite 是否需要从 Tushare 续更
- 检查 Parquet 是否需要从 SQLite 刷新
- 检查因子面板、latest scores、盘前参考文件是否已经存在且日期匹配
- 对已经满足条件的步骤自动跳过
- 在控制台输出每一步的检查结果和执行结果

补充说明：

- 如果未显式传 `--trade-date`，脚本会在同步 Tushare 时把日历向信号日后额外扩几天，用来自动推导下一个开市日

常用参数：

```bash
./scripts/run_v1_1_premarket.sh --signal-date 2026-03-25
./scripts/run_v1_1_premarket.sh --signal-date 2026-03-25 --trade-date 2026-03-26
./scripts/run_v1_1_premarket.sh --signal-date 2026-03-25 --force-inference
./scripts/run_v1_1_premarket.sh --signal-date 2026-03-25 --force-premarket
```

## 前置检查

先确认本地 SQLite 源库和 `storage` 快照都已经同步到最新收盘日。

### 0. 从 Tushare 更新 SQLite 源库

先配置 token：

```bash
export TUSHARE_TOKEN=your_token
```

然后把最新数据写入：

```bash
./.venv/bin/ashare-backtest sync-tushare-sqlite \
  --sqlite-path storage/source/ashare_arena_sync.db \
  --end 2026-03-25
```

日常续更时，通常可以不传 `--start`，让命令从 SQLite 当前最大 `trade_date + 1` 自动续更：

```bash
./.venv/bin/ashare-backtest sync-tushare-sqlite \
  --sqlite-path storage/source/ashare_arena_sync.db
```

这一步会更新：

- `trading_calendar`
- `equity_instruments`
- `equity_universe_memberships` 里的 `all_active`
- `equity_daily_bars`

说明：

- 这一步更新的是 SQLite 源数据库，不会自动刷新 `storage/parquet/`
- 同一天重复执行不会额外插入重复日线，而是按主键覆盖更新

### 1. 从 SQLite 刷新 Parquet 快照

更新 `storage`：

```bash
./.venv/bin/ashare-backtest import-sqlite storage/source/ashare_arena_sync.db --storage-root storage
```

查看 universe：

```bash
./.venv/bin/ashare-backtest list-universes --storage-root storage
```

如果要做 `T+1` 日盘前选股，必须保证 [daily.parquet](/Users/yongqiuwu/works/github/Trade/storage/parquet/bars/daily.parquet) 已经包含 `T` 日收盘后的完整数据。

## 执行步骤

下面以：

- 信号日 `T = 2026-03-25`
- 执行日 `T+1 = 2026-03-26`

为例。

### 2. 构建信号日 factor snapshot

```bash
./.venv/bin/ashare-backtest build-factors \
  --storage-root storage \
  --factor-spec-id research_industry_v4_v1_1 \
  --universe-name tradable_core \
  --start-date 2024-01-02 \
  --as-of-date 2026-03-25
```

输出文件：

- [asof_2026-03-25.parquet](/Users/yongqiuwu/works/github/Trade/research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-25.parquet)

### 3. 用 walk_forward 窗口对信号日打分

```bash
./.venv/bin/ashare-backtest train-lgbm-walk-forward-as-of-date \
  --factor-panel-path research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-25.parquet \
  --label-column industry_excess_fwd_return_5 \
  --train-window-months 12 \
  --as-of-date 2026-03-25 \
  --output-scores-path research/models/walk_forward_scores_industry_v4_v1_1_2026-03-25.parquet \
  --output-metrics-path research/models/walk_forward_metrics_industry_v4_v1_1_2026-03-25.json
```

输出文件：

- [walk_forward_scores_industry_v4_v1_1_2026-03-25.parquet](/Users/yongqiuwu/works/github/Trade/research/models/walk_forward_scores_industry_v4_v1_1_2026-03-25.parquet)
- [walk_forward_metrics_industry_v4_v1_1_2026-03-25.json](/Users/yongqiuwu/works/github/Trade/research/models/walk_forward_metrics_industry_v4_v1_1_2026-03-25.json)

### 4. 生成下一交易日盘前调仓参考

```bash
./.venv/bin/ashare-backtest generate-premarket-reference \
  --scores-path research/models/walk_forward_scores_industry_v4_v1_1_2026-03-25.parquet \
  --storage-root storage \
  --trade-date 2026-03-26 \
  --top-k 6 \
  --rebalance-every 5 \
  --lookback-window 20 \
  --min-hold-bars 8 \
  --keep-buffer 2 \
  --min-turnover-names 3 \
  --max-names-per-industry 2 \
  --output-path research/models/premarket_reference_industry_v4_v1_1_2026-03-26.json
```

输出文件：

- [premarket_reference_industry_v4_v1_1_2026-03-26.json](/Users/yongqiuwu/works/github/Trade/research/models/premarket_reference_industry_v4_v1_1_2026-03-26.json)

## 如何理解输出

盘前参考 JSON 里重点看这几个字段：

- `summary`
- `selected_symbols`
- `target_weights`
- `actions`
- `risk_flags`

含义：

- `selected_symbols`：模型选出的目标持仓股票
- `target_weights`：目标权重
- `actions`：相对当前组合的动作建议，例如 `BUY`、`SELL`、`ADD`、`TRIM`、`HOLD`
- `risk_flags`：流动性、短期涨幅过快、行业集中等风险提示

## 每日复用模板

把日期替换成当天即可：

- `T`：最新完整收盘日
- `T+1`：下一交易日

命令模板：

```bash
./.venv/bin/ashare-backtest sync-tushare-sqlite \
  --sqlite-path storage/source/ashare_arena_sync.db \
  --end T

./.venv/bin/ashare-backtest import-sqlite storage/source/ashare_arena_sync.db --storage-root storage

./.venv/bin/ashare-backtest build-factors \
  --storage-root storage \
  --factor-spec-id research_industry_v4_v1_1 \
  --universe-name tradable_core \
  --start-date 2024-01-02 \
  --as-of-date T

./.venv/bin/ashare-backtest train-lgbm-walk-forward-as-of-date \
  --factor-panel-path research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_T.parquet \
  --label-column industry_excess_fwd_return_5 \
  --train-window-months 12 \
  --as-of-date T \
  --output-scores-path research/models/walk_forward_scores_industry_v4_v1_1_T.parquet \
  --output-metrics-path research/models/walk_forward_metrics_industry_v4_v1_1_T.json

./.venv/bin/ashare-backtest generate-premarket-reference \
  --scores-path research/models/walk_forward_scores_industry_v4_v1_1_T.parquet \
  --storage-root storage \
  --trade-date T+1 \
  --top-k 6 \
  --rebalance-every 5 \
  --lookback-window 20 \
  --min-hold-bars 8 \
  --keep-buffer 2 \
  --min-turnover-names 3 \
  --max-names-per-industry 2 \
  --output-path research/models/premarket_reference_industry_v4_v1_1_T_plus_1.json
```

## 注意事项

- 如果 `import-sqlite` 后最新日线还没有更新到 `T`，不要继续生成盘前信号。
- 当前 `v1.1` 已默认使用 `tradable_core`，不是全量 `all_active`。
- `generate-premarket-reference` 给的是盘前执行参考，不等于实际成交结果。
- 如果后续需要固定成日常例行流程，建议再封装成单独脚本。
