from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from ashare_backtest.data.catalog import DatasetSummary, build_catalog, write_catalog


DEFAULT_SQLITE_SOURCE = (
    "/Users/yongqiuwu/works/github/Hyper-Alpha-Arena/ashare-arena/backend/ashare_arena.db"
)
DERIVED_UNIVERSE_ACTIVE = "all_active"
DERIVED_UNIVERSE_TRADABLE = "tradable_core"
TRADABLE_MIN_LISTING_DAYS = 120
TRADABLE_RECENT_WINDOW = 20
TRADABLE_MIN_RECENT_TRADING_DAYS = 15
TRADABLE_MAX_RECENT_SUSPENDED_DAYS = 5
TRADABLE_MIN_MEDIAN_DAILY_AMOUNT = 200_000.0


class SQLiteParquetImporter:
    def __init__(self, sqlite_path: str | Path, storage_root: str | Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.storage_root = Path(storage_root)
        self.parquet_root = self.storage_root / "parquet"

    def run(self) -> list[DatasetSummary]:
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as conn:
            bars_frame = self._load_bars(conn)
            instruments_frame = self._load_instruments(conn)
            datasets = [
                self._export_bars(bars_frame),
                self._export_instruments(instruments_frame),
                self._export_calendar(conn, bars_frame),
                self._export_universe_memberships(conn, instruments_frame, bars_frame),
            ]
        catalog = build_catalog(
            source_type="sqlite",
            source_path=str(self.sqlite_path),
            datasets=datasets,
            sqlite_summary=self._build_sqlite_summary(bars_frame, instruments_frame),
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

    def _load_instruments(self, conn: sqlite3.Connection) -> pd.DataFrame:
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
        return frame

    def _export_instruments(self, frame: pd.DataFrame) -> DatasetSummary:
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
            missing_bar_dates = sorted(bars_dates - calendar_open_dates)
            if missing_bar_dates:
                derived_rows = pd.DataFrame(
                    {
                        "trade_date": missing_bar_dates,
                        "is_open": True,
                        "has_night_session": False,
                        "notes": "derived_from_equity_daily_bars",
                    }
                )
                frame = (
                    pd.concat([frame, derived_rows], ignore_index=True)
                    .sort_values("trade_date")
                    .drop_duplicates(subset=["trade_date"], keep="first")
                    .reset_index(drop=True)
                )
        target = self.parquet_root / "calendar" / "ashare_trading_calendar.parquet"
        return self._write_dataset(frame, target, "calendar.ashare", "trade_date")

    def _export_universe_memberships(
        self,
        conn: sqlite3.Connection,
        instruments_frame: pd.DataFrame,
        bars_frame: pd.DataFrame,
    ) -> DatasetSummary:
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
        derived = self._build_derived_universe_memberships(instruments_frame, bars_frame)
        if not derived.empty:
            frame = frame.loc[~frame["universe_name"].isin(derived["universe_name"].unique())]
            frame = pd.concat([frame, derived], ignore_index=True).sort_values(
                ["universe_name", "effective_date", "symbol"]
            )
        target = self.parquet_root / "universe" / "memberships.parquet"
        return self._write_dataset(frame, target, "universe.memberships", "effective_date")

    def _build_derived_universe_memberships(
        self,
        instruments_frame: pd.DataFrame,
        bars_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        if instruments_frame.empty:
            return pd.DataFrame(columns=["universe_name", "symbol", "effective_date", "expiry_date", "source"])

        snapshot_date = bars_frame["trade_date"].max()
        if pd.isna(snapshot_date):
            return pd.DataFrame(columns=["universe_name", "symbol", "effective_date", "expiry_date", "source"])

        active = instruments_frame.loc[instruments_frame["is_active"]].copy()
        if active.empty:
            return pd.DataFrame(columns=["universe_name", "symbol", "effective_date", "expiry_date", "source"])

        rows = [self._membership_frame(active["symbol"], DERIVED_UNIVERSE_ACTIVE, snapshot_date, "derived_active_gate")]

        tradable = active.loc[~active["is_st"]].copy()
        tradable["listing_days"] = (snapshot_date - tradable["listing_date"]).dt.days
        tradable = tradable.loc[tradable["listing_days"].fillna(-1) >= TRADABLE_MIN_LISTING_DAYS].copy()
        if tradable.empty:
            return pd.concat(rows, ignore_index=True)

        recent_start = snapshot_date - pd.Timedelta(days=TRADABLE_RECENT_WINDOW * 2)
        recent_bars = bars_frame.loc[
            (bars_frame["symbol"].isin(tradable["symbol"])) & (bars_frame["trade_date"] >= recent_start)
        ].copy()
        if recent_bars.empty:
            return pd.concat(rows, ignore_index=True)

        recent_bars = recent_bars.sort_values(["symbol", "trade_date"])
        recent_bars = recent_bars.groupby("symbol", group_keys=False).tail(TRADABLE_RECENT_WINDOW)
        recent_stats = (
            recent_bars.groupby("symbol")
            .agg(
                recent_trading_days=("trade_date", "nunique"),
                recent_suspended_days=("is_suspended", "sum"),
                median_amount=("amount", "median"),
                latest_suspended=("is_suspended", "last"),
            )
            .reset_index()
        )
        tradable = tradable.merge(recent_stats, on="symbol", how="left")
        tradable = tradable.loc[
            (tradable["recent_trading_days"].fillna(0) >= TRADABLE_MIN_RECENT_TRADING_DAYS)
            & (tradable["recent_suspended_days"].fillna(TRADABLE_RECENT_WINDOW) <= TRADABLE_MAX_RECENT_SUSPENDED_DAYS)
            & (~tradable["latest_suspended"].fillna(True))
            & (tradable["median_amount"].fillna(0.0) >= TRADABLE_MIN_MEDIAN_DAILY_AMOUNT)
        ].copy()
        if not tradable.empty:
            rows.append(
                self._membership_frame(
                    tradable["symbol"],
                    DERIVED_UNIVERSE_TRADABLE,
                    snapshot_date,
                    "derived_tradable_gate",
                )
            )
        return pd.concat(rows, ignore_index=True)

    @staticmethod
    def _membership_frame(
        symbols: pd.Series,
        universe_name: str,
        effective_date: pd.Timestamp,
        source: str,
    ) -> pd.DataFrame:
        values = sorted(pd.Series(symbols).astype(str).unique().tolist())
        return pd.DataFrame(
            {
                "universe_name": universe_name,
                "symbol": values,
                "effective_date": effective_date,
                "expiry_date": pd.NaT,
                "source": source,
            }
        )

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

    @staticmethod
    def _build_sqlite_summary(bars_frame: pd.DataFrame, instruments_frame: pd.DataFrame) -> dict[str, int | str]:
        trade_dates = pd.to_datetime(bars_frame.get("trade_date"), errors="coerce").dropna()
        return {
            "equity_symbol_count": int(bars_frame["symbol"].nunique()) if "symbol" in bars_frame.columns else 0,
            "instrument_count": int(len(instruments_frame)),
            "date_min": trade_dates.min().date().isoformat() if not trade_dates.empty else "",
            "date_max": trade_dates.max().date().isoformat() if not trade_dates.empty else "",
        }
