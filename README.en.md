# A-Share Low-Frequency Backtesting Toolkit

[中文说明](README.md)

Full usage guide: [Documentation](https://cyecho-io.github.io/ashare-lowfreq-research/).

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
python -m pip install -e ".[dev]"
```

This exposes:

- `ashare-backtest`
- `ashare-backtest-web`

Copy the environment template before using Tushare-backed commands:

```bash
cp .env.example .env
```

Then fill in `TUSHARE_TOKEN` when you want to sync real market data.

## Quick Demo

If you want to evaluate the repository after clone without preparing your own market data, use the bundled tiny dataset under `storage/demo/`.

Set up the environment:

```bash
bash scripts/bootstrap_demo.sh
source .venv/bin/activate
```

Run the demo research preset:

```bash
ashare-backtest run-research-config configs/qlib/demo_strategy.toml
```

Start the local web console:

```bash
ashare-backtest-web
```

Then open `http://127.0.0.1:8888`.

What this demo gives you:

- a tracked tiny A-share sample dataset with project parquet data and a small qlib provider
- runnable native / qlib demo research presets
- one end-to-end example that generates factors, model scores, and backtest outputs locally
- generated backtest outputs under `results/demo_backtest`
- a local web UI for dashboard, backtest artifacts, and simulation views

Before opening `/backtest`, run `ashare-backtest run-research-config configs/qlib/demo_strategy.toml` once so the demo score file is generated locally.

If you want to switch from demo data to your own local dataset later, update `[storage].root` in your config. For qlib workflows, also update `[qlib].provider_uri` and `[qlib].market`. See [real data setup](https://cyecho-io.github.io/ashare-lowfreq-research/real-data/).

## Quick Start

Before running the full workflow on your own data, prepare the local data root first.

Recommended preparation order:

1. Install the project and copy `.env.example` to `.env`
2. Fill in `TUSHARE_TOKEN`
3. Create the first version of the local SQLite source database with Tushare sync
4. Import SQLite into Parquet storage
5. Run factor build, research, backtest, or the web console

### Prepare Initial SQLite Market Data

The repository uses a two-layer local data layout:

- `storage/source/`: writable source SQLite database
- `storage/parquet/`: analysis-friendly Parquet snapshots used by research and backtests

To create the first local SQLite market snapshot, run:

```bash
ashare-backtest sync-tushare-sqlite \
  --sqlite-path storage/source/ashare_arena_sync.db \
  --start 20240101 \
  --end 20260331
```

This command will:

- create `storage/source/ashare_arena_sync.db` if it does not already exist
- sync the trading calendar
- sync stock master data
- sync daily bars into SQLite
- refresh the derived `all_active` universe

If you also want benchmark history for web and reporting views, run:

```bash
ashare-backtest sync-tushare-benchmark \
  --symbol 000300.SH \
  --start 20240101 \
  --end 20260331
```

### Import SQLite Into Parquet

After SQLite is ready, import it into the Parquet storage layer:

```bash
ashare-backtest import-sqlite storage/source/ashare_arena_sync.db --storage-root storage
```

This generates the standard files expected by the rest of the project under `storage/parquet/` and refreshes `storage/catalog.json`.

### Run The Research Flow

Validate a strategy script:

```bash
ashare-backtest validate strategies/buy_and_hold.py
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
ashare-backtest run-research-config configs/qlib/research_industry_v4_v1_1_qlib.toml
```

Run the tracked demo preset against the bundled tiny dataset:

```bash
ashare-backtest run-research-config configs/qlib/demo_strategy.toml
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

### Start The Web Console

Once `storage/` contains imported data and you have at least one research or backtest run, start the local web console:

```bash
ashare-backtest-web
```

The default address is `http://127.0.0.1:8888`.

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

The current recommended preset is centered on [`configs/qlib/research_industry_v4_v1_1_qlib.toml`](/Users/yongqiuwu/works/github/Trade/configs/qlib/research_industry_v4_v1_1_qlib.toml):

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
- [`CONTRIBUTING.md`](/Users/yongqiuwu/works/github/Trade/CONTRIBUTING.md)
- [`storage/demo/README.md`](/Users/yongqiuwu/works/github/Trade/storage/demo/README.md)

## Testing

Run the test suite with:

```bash
python3 -m pytest
```

Install the package first so `ashare_backtest` is importable:

```bash
python -m pip install -e ".[dev]"
python3 -m pytest
```
