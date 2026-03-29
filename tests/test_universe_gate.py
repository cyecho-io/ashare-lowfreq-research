from __future__ import annotations

import sqlite3

import pandas as pd

from ashare_backtest.data import SQLiteParquetImporter, load_universe_symbols
from ashare_backtest.factors import FactorBuildConfig, FactorBuilder


def test_import_sqlite_builds_tradable_core_universe(tmp_path) -> None:
    sqlite_path = tmp_path / "source.db"
    storage_root = tmp_path / "storage"

    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            """
            create table equity_instruments (
                symbol text,
                exchange text,
                name text,
                listing_date text,
                delisting_date text,
                board text,
                industry_level_1 text,
                industry_level_2 text,
                is_st integer,
                is_active integer
            )
            """
        )
        conn.execute(
            """
            create table equity_daily_bars (
                symbol text,
                trade_date text,
                open_price real,
                high_price real,
                low_price real,
                close_price real,
                prev_close_price real,
                adj_factor real,
                volume real,
                turnover_amount real,
                turnover_rate real,
                limit_up_price real,
                limit_down_price real,
                is_suspended integer,
                is_limit_up integer,
                is_limit_down integer
            )
            """
        )
        conn.execute(
            """
            create table trading_calendar (
                trade_date text,
                is_open integer,
                has_night_session integer,
                notes text
            )
            """
        )
        conn.execute(
            """
            create table equity_universe_memberships (
                universe_name text,
                symbol text,
                effective_date text,
                expiry_date text,
                source text
            )
            """
        )

        instruments = [
            ("AAA", "SSE", "AAA", "2024-01-01", None, "main", "Tech", "App", 0, 1),
            ("BBB", "SSE", "BBB", "2024-01-01", None, "main", "Tech", "App", 1, 1),
            ("CCC", "SSE", "CCC", "2025-12-20", None, "main", "Tech", "App", 0, 1),
            ("DDD", "SSE", "DDD", "2024-01-01", None, "main", "Tech", "App", 0, 1),
            ("EEE", "SSE", "EEE", "2024-01-01", None, "main", "Tech", "App", 0, 0),
        ]
        conn.executemany("insert into equity_instruments values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", instruments)

        trade_dates = pd.bdate_range("2026-02-23", periods=20)
        calendar_rows = [(item.date().isoformat(), 1, 0, "") for item in trade_dates]
        conn.executemany("insert into trading_calendar values (?, ?, ?, ?)", calendar_rows)

        bar_rows: list[tuple[object, ...]] = []
        for idx, item in enumerate(trade_dates):
            trade_date = item.date().isoformat()
            for symbol, amount, suspended in (
                ("AAA", 30_000_000.0, 0),
                ("BBB", 30_000_000.0, 0),
                ("CCC", 30_000_000.0, 0),
                ("DDD", 100_000.0, 0),
                ("EEE", 30_000_000.0, 0),
            ):
                bar_rows.append(
                    (
                        symbol,
                        trade_date,
                        10.0 + idx,
                        10.5 + idx,
                        9.5 + idx,
                        10.2 + idx,
                        10.0 + idx,
                        1.0,
                        1_000_000.0,
                        amount,
                        0.01,
                        11.0 + idx,
                        9.0 + idx,
                        suspended,
                        0,
                        0,
                    )
                )
        conn.executemany("insert into equity_daily_bars values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", bar_rows)
        conn.commit()

    SQLiteParquetImporter(sqlite_path=sqlite_path, storage_root=storage_root).run()

    all_active = load_universe_symbols(storage_root, "all_active", as_of_date="2026-03-20")
    tradable_core = load_universe_symbols(storage_root, "tradable_core", as_of_date="2026-03-20")

    assert all_active == ("AAA", "BBB", "CCC", "DDD")
    assert tradable_core == ("AAA",)


def test_factor_builder_filters_by_universe_name(tmp_path) -> None:
    storage_root = tmp_path / "storage"
    bars_path = storage_root / "parquet" / "bars" / "daily.parquet"
    instruments_path = storage_root / "parquet" / "instruments" / "ashare_instruments.parquet"
    universe_path = storage_root / "parquet" / "universe" / "memberships.parquet"
    output_path = tmp_path / "factor_panel.parquet"

    bars_path.parent.mkdir(parents=True, exist_ok=True)
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    universe_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": close,
                "prev_close": close - 0.1,
                "adj_factor": 1.0,
                "volume": 1_000_000.0,
                "amount": 30_000_000.0,
                "turnover_rate": 0.01,
                "limit_up_price": 11.0,
                "limit_down_price": 9.0,
                "is_suspended": False,
                "is_limit_up": False,
                "is_limit_down": False,
            }
            for symbol in ("AAA", "BBB")
            for trade_date, close in zip(pd.bdate_range("2026-01-01", periods=70), range(70), strict=False)
        ]
    ).to_parquet(bars_path, index=False)

    pd.DataFrame(
        [
            {"symbol": "AAA", "industry_level_1": "Tech"},
            {"symbol": "BBB", "industry_level_1": "Finance"},
        ]
    ).to_parquet(instruments_path, index=False)

    pd.DataFrame(
        [
            {
                "universe_name": "tradable_core",
                "symbol": "AAA",
                "effective_date": "2026-03-31",
                "expiry_date": None,
                "source": "test",
            }
        ]
    ).to_parquet(universe_path, index=False)

    panel = FactorBuilder(
        FactorBuildConfig(
            storage_root=storage_root.as_posix(),
            output_path=output_path.as_posix(),
            universe_name="tradable_core",
            start_date="2026-01-01",
            as_of_date="2026-03-31",
        )
    ).build()

    assert panel["symbol"].nunique() == 1
    assert set(panel["symbol"]) == {"AAA"}


def test_load_universe_symbols_derives_tradable_core_for_historical_date_without_matching_membership(tmp_path) -> None:
    storage_root = tmp_path / "storage"
    bars_path = storage_root / "parquet" / "bars" / "daily.parquet"
    instruments_path = storage_root / "parquet" / "instruments" / "ashare_instruments.parquet"
    universe_path = storage_root / "parquet" / "universe" / "memberships.parquet"

    bars_path.parent.mkdir(parents=True, exist_ok=True)
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    universe_path.parent.mkdir(parents=True, exist_ok=True)

    trade_dates = pd.bdate_range("2026-01-05", periods=30)
    pd.DataFrame(
        [
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "open": 10.0,
                "high": 10.5,
                "low": 9.5,
                "close": close,
                "prev_close": close - 0.1,
                "adj_factor": 1.0,
                "volume": 1_000_000.0,
                "amount": amount,
                "turnover_rate": 0.01,
                "limit_up_price": 11.0,
                "limit_down_price": 9.0,
                "is_suspended": False,
                "is_limit_up": False,
                "is_limit_down": False,
            }
            for symbol, amount in (("AAA", 30_000_000.0), ("BBB", 30_000_000.0), ("CCC", 100_000.0))
            for close, trade_date in zip(range(30), trade_dates, strict=False)
        ]
    ).to_parquet(bars_path, index=False)

    pd.DataFrame(
        [
            {"symbol": "AAA", "listing_date": "2024-01-01", "delisting_date": None, "is_st": False, "is_active": True},
            {"symbol": "BBB", "listing_date": "2024-01-01", "delisting_date": None, "is_st": True, "is_active": True},
            {"symbol": "CCC", "listing_date": "2024-01-01", "delisting_date": None, "is_st": False, "is_active": True},
        ]
    ).to_parquet(instruments_path, index=False)

    pd.DataFrame(
        [
            {
                "universe_name": "tradable_core",
                "symbol": "AAA",
                "effective_date": "2026-03-27",
                "expiry_date": None,
                "source": "latest_snapshot_only",
            }
        ]
    ).to_parquet(universe_path, index=False)

    symbols = load_universe_symbols(storage_root, "tradable_core", as_of_date="2026-02-02")

    assert symbols == ("AAA",)
