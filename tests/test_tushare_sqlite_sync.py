from __future__ import annotations

import sqlite3

import pandas as pd

from ashare_backtest.data.tushare_sync import TushareSQLiteSync


class FakeTushareClient:
    def trade_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": "20260325", "is_open": 1, "pretrade_date": "20260324"},
                {"exchange": "SSE", "cal_date": "20260326", "is_open": 0, "pretrade_date": "20260325"},
            ]
        )

    def stock_basic(self, list_status: str) -> pd.DataFrame:
        data = {
            "L": [
                {
                    "ts_code": "000001.SZ",
                    "symbol": "000001",
                    "name": "平安银行",
                    "area": "深圳",
                    "industry": "银行",
                    "market": "主板",
                    "list_date": "19910403",
                    "delist_date": None,
                    "list_status": "L",
                },
                {
                    "ts_code": "600000.SH",
                    "symbol": "600000",
                    "name": "ST浦发",
                    "area": "上海",
                    "industry": "银行",
                    "market": "主板",
                    "list_date": "19991110",
                    "delist_date": None,
                    "list_status": "L",
                },
            ],
            "D": [
                {
                    "ts_code": "000002.SZ",
                    "symbol": "000002",
                    "name": "退市样本",
                    "area": "深圳",
                    "industry": "地产",
                    "market": "主板",
                    "list_date": "19910101",
                    "delist_date": "20260320",
                    "list_status": "D",
                }
            ],
            "P": [],
        }
        return pd.DataFrame(data[list_status])

    def daily(self, trade_date: str) -> pd.DataFrame:
        if trade_date != "20260325":
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": trade_date,
                    "open": 10.0,
                    "high": 10.2,
                    "low": 9.9,
                    "close": 10.1,
                    "pre_close": 9.8,
                    "vol": 100000.0,
                    "amount": 500000.0,
                },
                {
                    "ts_code": "600000.SH",
                    "trade_date": trade_date,
                    "open": 8.0,
                    "high": 8.1,
                    "low": 7.9,
                    "close": 8.0,
                    "pre_close": 8.0,
                    "vol": 80000.0,
                    "amount": 300000.0,
                },
            ]
        )

    def daily_basic(self, trade_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": trade_date, "turnover_rate": 0.5},
                {"ts_code": "600000.SH", "trade_date": trade_date, "turnover_rate": 0.3},
            ]
        )

    def adj_factor(self, trade_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": trade_date, "adj_factor": 1.1},
                {"ts_code": "600000.SH", "trade_date": trade_date, "adj_factor": 1.0},
            ]
        )

    def stk_limit(self, trade_date: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "trade_date": trade_date, "up_limit": 10.9, "down_limit": 8.9},
                {"ts_code": "600000.SH", "trade_date": trade_date, "up_limit": 8.8, "down_limit": 7.2},
            ]
        )

    def suspend_d(self, trade_date: str) -> pd.DataFrame:
        return pd.DataFrame([{"ts_code": "600000.SH", "trade_date": trade_date, "suspend_type": "S"}])


def _create_schema(path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            create table equity_daily_bars (
                id integer primary key autoincrement,
                symbol text not null,
                trade_date text not null,
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
                is_suspended integer not null,
                is_limit_up integer not null,
                is_limit_down integer not null,
                unique(symbol, trade_date)
            );
            create table equity_instruments (
                id integer primary key autoincrement,
                symbol text not null unique,
                exchange text not null,
                name text not null,
                listing_date text,
                delisting_date text,
                board text,
                industry_level_1 text,
                industry_level_2 text,
                is_st integer not null,
                is_active integer not null,
                created_at text not null,
                updated_at text not null
            );
            create table trading_calendar (
                id integer primary key autoincrement,
                trade_date text not null unique,
                is_open integer not null,
                has_night_session integer not null,
                notes text
            );
            create table equity_universe_memberships (
                id integer primary key autoincrement,
                universe_name text not null,
                symbol text not null,
                effective_date text not null,
                expiry_date text,
                source text,
                unique(universe_name, symbol, effective_date)
            );
            """
        )
        conn.commit()


def test_tushare_sqlite_sync_writes_calendar_instruments_bars_and_universe(tmp_path) -> None:
    sqlite_path = tmp_path / "ashare.db"
    _create_schema(sqlite_path)

    summary = TushareSQLiteSync(sqlite_path=sqlite_path, client=FakeTushareClient()).sync(
        start_date="20260325",
        end_date="20260326",
    )

    assert summary.open_trade_dates == 1
    assert summary.stock_basic_rows == 3
    assert summary.active_symbols == 2
    assert summary.daily_rows == 2

    with sqlite3.connect(sqlite_path) as conn:
        instruments = pd.read_sql_query(
            "select symbol, exchange, board, industry_level_1, is_st, is_active, delisting_date from equity_instruments order by symbol",
            conn,
        )
        assert instruments["symbol"].tolist() == ["000001.SZ", "000002.SZ", "600000.SH"]
        assert instruments.loc[instruments["symbol"] == "000001.SZ", "is_active"].iloc[0] == 1
        assert instruments.loc[instruments["symbol"] == "000002.SZ", "is_active"].iloc[0] == 0
        assert instruments.loc[instruments["symbol"] == "600000.SH", "is_st"].iloc[0] == 1

        memberships = pd.read_sql_query(
            "select universe_name, symbol, source from equity_universe_memberships order by symbol",
            conn,
        )
        assert memberships["symbol"].tolist() == ["000001.SZ", "600000.SH"]
        assert set(memberships["universe_name"]) == {"all_active"}

        bars = pd.read_sql_query(
            "select symbol, trade_date, turnover_rate, is_suspended from equity_daily_bars order by symbol",
            conn,
        )
        assert bars["symbol"].tolist() == ["000001.SZ", "600000.SH"]
        assert bars.loc[bars["symbol"] == "000001.SZ", "turnover_rate"].iloc[0] == 0.5
        assert bars.loc[bars["symbol"] == "600000.SH", "is_suspended"].iloc[0] == 1
