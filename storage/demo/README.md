# Demo Data Contract

`storage/demo/` is the recommended location for a small tracked demo dataset for public evaluation.

## Goal

The demo dataset should let a new user:

- start the web console
- inspect dashboard data readiness
- run native model research and backtests
- run qlib model research and backtests
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
- `storage/demo/qlib_data/cn_data/`

The qlib provider is generated from the tracked demo parquet files:

```bash
python scripts/build_demo_qlib_provider.py \
  --storage-root storage/demo \
  --provider-uri storage/demo/qlib_data/cn_data \
  --market demo
```

## Recommended Size

Keep the dataset intentionally small:

- 20 to 100 symbols
- 40 to 120 trading days
- one or two universe names

This is enough to demonstrate the product while keeping the repository lightweight.

## Notes

- Do not place secrets or private vendor data in this directory.
- Document the data source and time range if demo data is later committed.
- The demo data is intentionally tiny and is only for product evaluation.
- For real research, configure your own storage root and qlib provider in `configs/native/*.toml` and `configs/qlib/*.toml`.
