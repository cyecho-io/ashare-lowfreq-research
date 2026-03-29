# A-Share Low-Frequency Backtesting Toolkit

[中文说明](README.zh-CN.md)

This repository is a personal A-share research and backtesting toolkit focused on a narrow, maintainable workflow:

- sync and standardize local A-share market data
- build factor panels and train score models
- run score-driven backtests with realistic execution constraints
- inspect results in a lightweight web console
- generate latest stock selection, premarket reference, and simulation artifacts

The project is intentionally opinionated. It is not trying to become a general-purpose quant platform.

## Current Scope

- Market: mainland China A-shares
- Frequency: daily bars
- Strategy shape: long-only stock portfolios
- Research loop: factor build -> model training -> walk-forward / latest inference -> score backtest
- Execution controls: commission, stamp tax, slippage, participation cap, pending-order handling
- Interfaces: CLI plus a local backtest web console

Out of scope for now:

- intraday / tick-level simulation
- derivatives, margin, or multi-asset portfolios
- distributed scheduling and multi-tenant infrastructure
- arbitrary unrestricted Python strategy execution

## Repository Layout

- `src/ashare_backtest/`: core package
- `src/ashare_backtest/web/`: local dashboard, backtest, and simulation console
- `configs/`: runnable backtest and research configs
- `research/`: local factor panels, model outputs, and latest artifacts
- `storage/`: normalized parquet market data and source SQLite database
- `strategies/`: protocol-constrained strategy scripts
- `docs/`: research notes, runbooks, and design docs
- `tests/`: regression tests

Generated outputs under `results/`, `research/factors/`, and `research/models/` are treated as local artifacts and are ignored by Git.

## Installation

Requires Python 3.11+.

```bash
python -m pip install -e .
```

This exposes:

- `ashare-backtest`
- `ashare-backtest-web`

## Quick Start

Validate a strategy script:

```bash
ashare-backtest validate strategies/buy_and_hold.py
```

Import local SQLite market data into parquet storage:

```bash
ashare-backtest import-sqlite storage/source/ashare_arena_sync.db --storage-root storage
```

Build a factor snapshot from a named universe:

```bash
ashare-backtest build-factors \
  --storage-root storage \
  --universe-name tradable_core \
  --start-date 2024-02-01 \
  --as-of-date 2024-12-31
```

Run a configured research pipeline:

```bash
ashare-backtest run-research-config configs/research_industry_v4_v1_1.toml
```

Run a backtest from exported model scores:

```bash
ashare-backtest run-model-backtest \
  --scores-path research/models/walk_forward_scores.parquet \
  --storage-root storage \
  --start-date 2025-01-01 \
  --end-date 2025-12-31 \
  --output-dir results/model_score_backtest
```

## Data Sync

Sync daily bars from Tushare into the project source SQLite database:

```bash
ashare-backtest sync-tushare-sqlite --start 20240101 --end 20260331
```

Sync benchmark index history into parquet storage:

```bash
ashare-backtest sync-tushare-benchmark --symbol 000300.SH --start 20240101 --end 20260331
```

`TUSHARE_TOKEN` is used by default when `--token` is not provided.

## Web Console

Start the local web console:

```bash
ashare-backtest-web
```

The console provides:

- a landing dashboard for trading-calendar and data-source readiness
- backtest run submission from configured presets
- result browsing and summary metrics
- equity curve visualization with optional benchmark overlay
- trade log inspection
- simulation-account creation, lineage inspection, and execution history views

## Recommended Research Preset

The current recommended preset is centered on [`configs/research_industry_v4_v1_1.toml`](/Users/yongqiuwu/works/github/Trade/configs/research_industry_v4_v1_1.toml):

- factor panel: `industry_v4`
- label: `industry_excess_fwd_return_5`
- training: monthly walk-forward with a 12-month training window
- portfolio: `top_k=6`, `rebalance_every=5`, `min_hold_bars=8`, `keep_buffer=2`
- turnover control: `min_turnover_names=3`
- industry constraint: `max_names_per_industry=2`

The default tradable universe workflow gates stocks at the `universe` layer before factor construction. After import, the project generates:

- `all_active`: all currently active stocks
- `tradable_core`: active, non-ST names listed for at least 120 days, with tradability and liquidity filters applied

## Useful Documents

- [`docs/mvp.md`](/Users/yongqiuwu/works/github/Trade/docs/mvp.md)
- [`docs/research-pipeline.md`](/Users/yongqiuwu/works/github/Trade/docs/research-pipeline.md)
- [`docs/strategy-v1-1-premarket-runbook.md`](/Users/yongqiuwu/works/github/Trade/docs/strategy-v1-1-premarket-runbook.md)
- [`docs/strategy-v1-1-latest-active-data-source.md`](/Users/yongqiuwu/works/github/Trade/docs/strategy-v1-1-latest-active-data-source.md)
- [`docs/strategy-v2-live-readiness-checklist.md`](/Users/yongqiuwu/works/github/Trade/docs/strategy-v2-live-readiness-checklist.md)
- [`docs/strategy-v2-roadmap.md`](/Users/yongqiuwu/works/github/Trade/docs/strategy-v2-roadmap.md)

## Testing

Run the test suite with:

```bash
python3 -m pytest
```
