from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from ashare_backtest.data.catalog import DatasetSummary, build_catalog, write_catalog


DEFAULT_SQLITE_SOURCE = (
    "/Users/yongqiuwu/works/github/Hyper-Alpha-Arena/ashare-arena/backend/ashare_arena.db"
)


class SQLiteParquetImporter:
    def __init__(self, sqlite_path: str | Path, storage_root: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.storage_root = Path(storage_root)
        self.parquet_root = self.storage_root / "parquet"

    def run(self) -> list[DatasetSummary]:
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as conn:
            bars_frame = self._load_bars(conn)
            datasets = [
                self._export_bars(bars_frame),
                self._export_instruments(conn),
                self._export_calendar(conn, bars_frame),
                self._export_universe_memberships(conn),
            ]
        catalog = build_catalog(
            source_type="sqlite",
            source_path=str(self.sqlite_path),
            datasets=datasets,
        )
        write_catalog(self.storage_root / "catalog.json", catalog)
        return datasets

    def _load_bars(self, conn: sqlite3.Connection) -> pd.DataFrame:
        query = """
        select
            symbol,
            trade_date,
            open_price as open,
            high_price as high,
            low_price as low,
            close_price as close,
            prev_close_price as prev_close,
            adj_factor,
            volume,
            turnover_amount as amount,
            turnover_rate,
            limit_up_price,
            limit_down_price,
            is_suspended,
            is_limit_up,
            is_limit_down
        from equity_daily_bars
        order by trade_date, symbol
        """
        frame = pd.read_sql_query(query, conn)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame["close_adj"] = frame["close"] * frame["adj_factor"].fillna(1.0)
        frame["is_suspended"] = frame["is_suspended"].astype(bool)
        frame["is_limit_up"] = frame["is_limit_up"].astype(bool)
        frame["is_limit_down"] = frame["is_limit_down"].astype(bool)
        return frame

    def _export_bars(self, frame: pd.DataFrame) -> DatasetSummary:
        target = self.parquet_root / "bars" / "daily.parquet"
        return self._write_dataset(frame, target, "bars.daily", "trade_date")

    def _export_instruments(self, conn: sqlite3.Connection) -> DatasetSummary:
        query = """
        select
            symbol,
            exchange,
            name,
            listing_date,
            delisting_date,
            board,
            industry_level_1,
            industry_level_2,
            is_st,
            is_active
        from equity_instruments
        order by symbol
        """
        frame = pd.read_sql_query(query, conn)
        for column in ("listing_date", "delisting_date"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        frame["is_st"] = frame["is_st"].astype(bool)
        frame["is_active"] = frame["is_active"].astype(bool)
        target = self.parquet_root / "instruments" / "ashare_instruments.parquet"
        return self._write_dataset(frame, target, "instruments.ashare", "listing_date")

    def _export_calendar(self, conn: sqlite3.Connection, bars_frame: pd.DataFrame) -> DatasetSummary:
        query = """
        select
            trade_date,
            is_open,
            has_night_session,
            notes
        from trading_calendar
        order by trade_date
        """
        frame = pd.read_sql_query(query, conn)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        if frame.empty:
            frame = self._derive_calendar_from_bars(bars_frame)
        else:
            frame["is_open"] = frame["is_open"].astype(bool)
            frame["has_night_session"] = frame["has_night_session"].astype(bool)
            bars_dates = set(bars_frame["trade_date"].dropna().unique().tolist())
            calendar_open_dates = set(frame.loc[frame["is_open"], "trade_date"].dropna().unique().tolist())
            if len(calendar_open_dates) < len(bars_dates):
                frame = self._derive_calendar_from_bars(bars_frame)
        target = self.parquet_root / "calendar" / "ashare_trading_calendar.parquet"
        return self._write_dataset(frame, target, "calendar.ashare", "trade_date")

    def _export_universe_memberships(self, conn: sqlite3.Connection) -> DatasetSummary:
        query = """
        select
            universe_name,
            symbol,
            effective_date,
            expiry_date,
            source
        from equity_universe_memberships
        order by universe_name, effective_date, symbol
        """
        frame = pd.read_sql_query(query, conn)
        for column in ("effective_date", "expiry_date"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
        target = self.parquet_root / "universe" / "memberships.parquet"
        return self._write_dataset(frame, target, "universe.memberships", "effective_date")

    @staticmethod
    def _derive_calendar_from_bars(bars_frame: pd.DataFrame) -> pd.DataFrame:
        dates = sorted(pd.Series(bars_frame["trade_date"].dropna().unique()))
        return pd.DataFrame(
            {
                "trade_date": dates,
                "is_open": True,
                "has_night_session": False,
                "notes": "derived_from_equity_daily_bars",
            }
        )

    @staticmethod
    def _write_dataset(frame: pd.DataFrame, path: Path, name: str, date_column: str) -> DatasetSummary:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        min_date = None
        max_date = None
        if date_column in frame.columns and not frame.empty:
            series = frame[date_column].dropna()
            if not series.empty:
                min_date = series.min().date().isoformat()
                max_date = series.max().date().isoformat()
        return DatasetSummary(
            name=name,
            path=str(path),
            rows=len(frame),
            min_date=min_date,
            max_date=max_date,
        )
