# Demo Data Contract

`storage/demo/` is the recommended location for a small tracked demo dataset for public evaluation.

## Goal

The demo dataset should let a new user:

- start the web console
- inspect dashboard data readiness
- run at least one small backtest
- inspect generated results

without requiring a Tushare account or a private local database.

## Recommended Contents

Suggested tracked files:

- `storage/demo/catalog.json`
- `storage/demo/parquet/bars/daily.parquet`
- `storage/demo/parquet/calendar/ashare_trading_calendar.parquet`
- `storage/demo/parquet/instruments/ashare_instruments.parquet`
- `storage/demo/parquet/universe/memberships.parquet`
- optional `storage/demo/parquet/benchmarks/000300.SH.parquet`

## Recommended Size

Keep the dataset intentionally small:

- 20 to 100 symbols
- 40 to 120 trading days
- one or two universe names

This is enough to demonstrate the product while keeping the repository lightweight.

## Notes

- Do not place secrets or private vendor data in this directory.
- Document the data source and time range if demo data is later committed.
- If tracked demo data is not committed, provide a download or generation script with the same directory layout.
