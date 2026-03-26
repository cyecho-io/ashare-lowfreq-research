# Storage Data Source Notes

## Purpose

`storage/` is the local A-share research data root for this repository.

It is organized as a two-layer setup:

- `storage/source/`: writable source database used for data sync and maintenance
- `storage/parquet/`: analysis-friendly snapshots used by backtests, factor research, and reporting

The current source-of-truth database is:

- [ashare_arena_sync.db](/Users/yongqiuwu/works/github/Trade/storage/source/ashare_arena_sync.db)

The current imported snapshot index is:

- [catalog.json](/Users/yongqiuwu/works/github/Trade/storage/catalog.json)

## Current Coverage

As of the current import recorded in [catalog.json](/Users/yongqiuwu/works/github/Trade/storage/catalog.json):

- `bars.daily`: `1,609,537` rows, `2024-01-02` to `2026-03-24`
- `instruments.ashare`: `5,493` rows
- `calendar.ashare`: `542` rows
- `universe.memberships`: `5,493` rows

Notes:

- The latest daily bar is currently `2026-03-24`.
- This is expected when the current date is `2026-03-25`, because `2026-03-25` daily bars are not complete until after market close.

## Directory Layout

```text
storage/
├── README.md
├── catalog.json
├── parquet/
│   ├── bars/
│   │   └── daily.parquet
│   ├── calendar/
│   │   └── ashare_trading_calendar.parquet
│   ├── instruments/
│   │   └── ashare_instruments.parquet
│   └── universe/
│       └── memberships.parquet
└── source/
    └── ashare_arena_sync.db
```

## Dataset Roles

### 1. Source SQLite

File:

- [ashare_arena_sync.db](/Users/yongqiuwu/works/github/Trade/storage/source/ashare_arena_sync.db)

Purpose:

- acts as the local operational database for data sync
- suitable for incremental updates, corrections, and schema-managed storage
- should be treated as the writable master copy inside this repository

Key tables:

- `equity_daily_bars`
- `equity_instruments`
- `trading_calendar`
- `equity_universe_memberships`

### 2. Parquet Snapshots

Files:

- [daily.parquet](/Users/yongqiuwu/works/github/Trade/storage/parquet/bars/daily.parquet)
- [ashare_instruments.parquet](/Users/yongqiuwu/works/github/Trade/storage/parquet/instruments/ashare_instruments.parquet)
- [ashare_trading_calendar.parquet](/Users/yongqiuwu/works/github/Trade/storage/parquet/calendar/ashare_trading_calendar.parquet)
- [memberships.parquet](/Users/yongqiuwu/works/github/Trade/storage/parquet/universe/memberships.parquet)

Purpose:

- optimized for research reads
- suitable for pandas, DuckDB, Polars, and backtest ingestion
- should be treated as imported analysis snapshots, not the primary sync target

## Main Tables and Meaning

### Daily Bars

File:

- [daily.parquet](/Users/yongqiuwu/works/github/Trade/storage/parquet/bars/daily.parquet)

Main fields:

- `symbol`
- `trade_date`
- `open`
- `high`
- `low`
- `close`
- `prev_close`
- `adj_factor`
- `volume`
- `amount`
- `turnover_rate`
- `limit_up_price`
- `limit_down_price`
- `is_suspended`
- `is_limit_up`
- `is_limit_down`
- `close_adj`

Current implementation note:

- `adj_factor` is currently imported as `1.0` from the sync path used in this repo.
- That means this is usable for many daily research tasks, but not yet a full adjusted-price history source.

### Instruments

File:

- [ashare_instruments.parquet](/Users/yongqiuwu/works/github/Trade/storage/parquet/instruments/ashare_instruments.parquet)

Main fields:

- `symbol`
- `exchange`
- `name`
- `listing_date`
- `delisting_date`
- `board`
- `industry_level_1`
- `industry_level_2`
- `is_st`
- `is_active`

Use cases:

- stock master data
- board filtering
- industry grouping
- active security filtering

### Trading Calendar

File:

- [ashare_trading_calendar.parquet](/Users/yongqiuwu/works/github/Trade/storage/parquet/calendar/ashare_trading_calendar.parquet)

Main fields:

- `trade_date`
- `is_open`
- `has_night_session`
- `notes`

Use cases:

- trading day alignment
- rebalance scheduling
- detecting the latest completed trading date

### Universe Memberships

File:

- [memberships.parquet](/Users/yongqiuwu/works/github/Trade/storage/parquet/universe/memberships.parquet)

Main fields:

- `universe_name`
- `symbol`
- `effective_date`
- `expiry_date`
- `source`

Current contents:

- one broad stock pool: `all_active`

Use cases:

- defining eligible stock pools
- reusing the same universe across multiple strategies
- future extension to custom pools such as board-specific or liquidity-screened universes

## Update Workflow

Current workflow used in this repo:

1. Sync or backfill source data into [ashare_arena_sync.db](/Users/yongqiuwu/works/github/Trade/storage/source/ashare_arena_sync.db)
2. Import SQLite into `storage/parquet/`
3. Refresh downstream factor panels, scores, and backtests

The import command is:

```bash
PYTHONPATH=src ./.venv/bin/python -m ashare_backtest.cli.main import-sqlite storage/source/ashare_arena_sync.db --storage-root storage
```

## Can Other Projects Reuse This?

Yes.

This storage layout is suitable as a shared local research data foundation for other A-share projects.

Recommended reuse pattern:

- use SQLite as the writable source database
- use Parquet as the read-optimized analysis layer

Good fit:

- daily factor research
- low-frequency backtests
- signal generation
- feature engineering
- ad hoc data analysis in pandas, DuckDB, or Polars

Less complete today:

- adjusted-price history
- minute or tick data
- richer historical universe definitions
- benchmark and index constituent history
- fundamentals beyond the current schema

## Recommended Rules

- Do not edit Parquet files directly.
- Apply data sync or fixes to SQLite first, then reimport.
- Treat [catalog.json](/Users/yongqiuwu/works/github/Trade/storage/catalog.json) as the first place to check freshness and coverage.
- If another repo wants to reuse this data, prefer mounting or referencing `storage/` directly instead of copying partial files.

## Practical Interpretation

For this repository:

- SQLite is the maintained data base
- Parquet is the research-facing snapshot

For future repositories:

- this `storage/` directory can serve as a reusable local A-share data warehouse
- but it should still be versioned and refreshed intentionally, not assumed to be self-updating
