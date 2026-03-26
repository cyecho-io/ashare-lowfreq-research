# Tushare SQLite Sync

这份文档说明如何在 `Trade` 仓库内，直接通过 Tushare 更新本地 SQLite 源数据库：

- [ashare_arena_sync.db](/Users/yongqiuwu/works/github/Trade/storage/source/ashare_arena_sync.db)

## 命令

先配置 token：

```bash
export TUSHARE_TOKEN=your_token
```

然后执行同步：

```bash
./.venv/bin/ashare-backtest sync-tushare-sqlite \
  --sqlite-path storage/source/ashare_arena_sync.db \
  --start 20260325 \
  --end 20260325
```

也可以不传 `--start`，让命令从数据库里已有的最新 `trade_date + 1` 自动续更：

```bash
./.venv/bin/ashare-backtest sync-tushare-sqlite \
  --sqlite-path storage/source/ashare_arena_sync.db \
  --end 20260325
```

如果不传 `--end`，默认使用当天日期。

## 会更新什么

这个命令会直接写入 SQLite 里的 4 张表：

- `trading_calendar`
- `equity_instruments`
- `equity_universe_memberships`
- `equity_daily_bars`

具体行为：

- 从 Tushare `trade_cal` 更新交易日历
- 从 Tushare `stock_basic` 更新股票主数据
- 重建 `all_active` universe membership
- 从 Tushare `daily`、`daily_basic`、`adj_factor`、`stk_limit`、`suspend_d` 更新日线行情

## 命令输出

命令成功后会返回一个 JSON 摘要，例如：

```json
{
  "start_date": "2026-03-25",
  "end_date": "2026-03-25",
  "open_trade_dates": 1,
  "stock_basic_rows": 5493,
  "active_symbols": 5493,
  "daily_rows": 5493,
  "daily_trade_dates": 1
}
```

含义：

- `open_trade_dates`: 本次同步范围内的开市日数量
- `stock_basic_rows`: 本次写入/更新的股票主数据行数
- `active_symbols`: 当前 `is_active=true` 的股票数
- `daily_rows`: 本次写入/更新的日线行情行数
- `daily_trade_dates`: 实际处理的开市日数

## 日常流程

每天建议按这个顺序执行：

1. 从 Tushare 更新 SQLite

```bash
./.venv/bin/ashare-backtest sync-tushare-sqlite \
  --sqlite-path storage/source/ashare_arena_sync.db
```

2. 从 SQLite 刷新 Parquet 快照

```bash
./.venv/bin/ashare-backtest import-sqlite storage/source/ashare_arena_sync.db --storage-root storage
```

3. 检查 universe

```bash
./.venv/bin/ashare-backtest list-universes --storage-root storage
```

4. 再执行因子、打分、盘前参考生成

## 注意事项

- 这个命令只负责把 Tushare 数据写入 SQLite，不会自动刷新 `storage/parquet/`
- 如果不接着跑 `import-sqlite`，研究和回测仍然读的是旧快照
- 当前 `all_active` membership 会按最新 `stock_basic` 结果重建
