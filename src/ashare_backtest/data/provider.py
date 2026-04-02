from __future__ import annotations

from bisect import bisect_right
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ashare_backtest.protocol import Bar


class DataProvider(ABC):
    @abstractmethod
    def get_trade_dates(self, start_date: date, end_date: date) -> list[date]:
        raise NotImplementedError

    @abstractmethod
    def get_history(
        self,
        symbols: tuple[str, ...],
        end_date: date,
        lookback: int,
    ) -> dict[str, list[Bar]]:
        raise NotImplementedError

    @abstractmethod
    def get_bars_on_date(self, symbols: tuple[str, ...], trade_date: date) -> dict[str, Bar]:
        raise NotImplementedError

    def preload(self, symbols: tuple[str, ...], start_date: date, end_date: date, lookback: int) -> None:
        """Optional hook for providers that can preload a bounded data slice."""


class InMemoryDataProvider(DataProvider):
    """Minimal placeholder provider used before wiring a real data adapter."""

    def __init__(self, bars: dict[str, list[Bar]]) -> None:
        self._bars = bars

    def get_trade_dates(self, start_date: date, end_date: date) -> list[date]:
        dates = {
            bar.trade_date
            for bars in self._bars.values()
            for bar in bars
            if start_date <= bar.trade_date <= end_date
        }
        return sorted(dates)

    def get_history(
        self,
        symbols: tuple[str, ...],
        end_date: date,
        lookback: int,
    ) -> dict[str, list[Bar]]:
        result: dict[str, list[Bar]] = {}
        for symbol in symbols:
            bars = [bar for bar in self._bars.get(symbol, []) if bar.trade_date <= end_date]
            result[symbol] = bars[-lookback:]
        return result

    def get_bars_on_date(self, symbols: tuple[str, ...], trade_date: date) -> dict[str, Bar]:
        result: dict[str, Bar] = {}
        for symbol in symbols:
            for bar in self._bars.get(symbol, []):
                if bar.trade_date == trade_date:
                    result[symbol] = bar
                    break
        return result


class ParquetDataProvider(DataProvider):
    BAR_COLUMNS = [
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "is_suspended",
        "is_limit_up",
        "is_limit_down",
    ]

    @dataclass(frozen=True)
    class _CacheWindow:
        symbols: tuple[str, ...]
        start_date: date
        end_date: date
        lookback: int

    def __init__(self, storage_root: str | Path) -> None:
        root = Path(storage_root)
        self.bars_path = root / "parquet" / "bars" / "daily.parquet"
        self.calendar_path = root / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
        self._cache_window: ParquetDataProvider._CacheWindow | None = None
        self._calendar_dates: list[date] | None = None
        self._all_open_dates: list[date] | None = None
        self._bars_by_symbol: dict[str, list[Bar]] = {}
        self._bar_dates_by_symbol: dict[str, list[date]] = {}
        self._bars_by_date: dict[date, dict[str, Bar]] = {}

    def preload(self, symbols: tuple[str, ...], start_date: date, end_date: date, lookback: int) -> None:
        normalized_symbols = tuple(sorted(set(symbols)))
        requested = self._CacheWindow(
            symbols=normalized_symbols,
            start_date=start_date,
            end_date=end_date,
            lookback=lookback,
        )
        if self._cache_window == requested:
            return

        all_open_dates = self._load_all_open_dates()
        self._calendar_dates = [item for item in all_open_dates if start_date <= item <= end_date]

        bars_frame = pd.read_parquet(self.bars_path, columns=self.BAR_COLUMNS)
        if normalized_symbols:
            bars_frame = bars_frame.loc[bars_frame["symbol"].isin(normalized_symbols)]
        cutoff_start = start_date
        if lookback > 0 and all_open_dates:
            start_index = next((index for index, item in enumerate(all_open_dates) if item >= start_date), None)
            if start_index is not None:
                cutoff_start = all_open_dates[max(0, start_index - lookback)]
        bars_mask = (
            (bars_frame["trade_date"] >= pd.Timestamp(cutoff_start))
            & (bars_frame["trade_date"] <= pd.Timestamp(end_date))
        )
        filtered = bars_frame.loc[bars_mask].sort_values(["symbol", "trade_date"])

        self._set_cache_from_frame(filtered)
        self._cache_window = requested

    def get_trade_dates(self, start_date: date, end_date: date) -> list[date]:
        if self._calendar_dates is not None:
            return [item for item in self._calendar_dates if start_date <= item <= end_date]
        return [item for item in self._load_all_open_dates() if start_date <= item <= end_date]

    def get_history(
        self,
        symbols: tuple[str, ...],
        end_date: date,
        lookback: int,
    ) -> dict[str, list[Bar]]:
        if self._bars_by_symbol:
            result: dict[str, list[Bar]] = {}
            for symbol in symbols:
                bars = self._bars_by_symbol.get(symbol, [])
                dates = self._bar_dates_by_symbol.get(symbol, [])
                end_index = bisect_right(dates, end_date)
                result[symbol] = bars[max(0, end_index - lookback) : end_index]
            return result
        if not symbols:
            return {}
        frame = pd.read_parquet(self.bars_path, columns=self.BAR_COLUMNS)
        mask = (frame["symbol"].isin(symbols)) & (frame["trade_date"] <= pd.Timestamp(end_date))
        filtered = frame.loc[mask].sort_values(["symbol", "trade_date"])
        result: dict[str, list[Bar]] = {}
        for symbol in symbols:
            rows = filtered.loc[filtered["symbol"] == symbol].tail(lookback)
            result[symbol] = [self._row_to_bar(row) for _, row in rows.iterrows()]
        return result

    def get_bars_on_date(self, symbols: tuple[str, ...], trade_date: date) -> dict[str, Bar]:
        if self._bars_by_date:
            day_map = self._bars_by_date.get(trade_date, {})
            return {symbol: day_map[symbol] for symbol in symbols if symbol in day_map}
        if not symbols:
            return {}
        frame = pd.read_parquet(self.bars_path, columns=self.BAR_COLUMNS)
        mask = (frame["symbol"].isin(symbols)) & (frame["trade_date"] == pd.Timestamp(trade_date))
        filtered = frame.loc[mask]
        return {row["symbol"]: self._row_to_bar(row) for _, row in filtered.iterrows()}

    def _load_all_open_dates(self) -> list[date]:
        if self._all_open_dates is not None:
            return self._all_open_dates
        calendar_frame = pd.read_parquet(self.calendar_path, columns=["trade_date", "is_open"])
        open_calendar = calendar_frame.loc[calendar_frame["is_open"], ["trade_date"]].sort_values("trade_date")
        self._all_open_dates = [item.date() for item in open_calendar["trade_date"].tolist()]
        return self._all_open_dates

    def _set_cache_from_frame(self, filtered: pd.DataFrame) -> None:
        bars_by_symbol: dict[str, list[Bar]] = {}
        bar_dates_by_symbol: dict[str, list[date]] = {}
        bars_by_date: dict[date, dict[str, Bar]] = {}
        for _, row in filtered.iterrows():
            bar = self._row_to_bar(row)
            bars_by_symbol.setdefault(bar.symbol, []).append(bar)
            bar_dates_by_symbol.setdefault(bar.symbol, []).append(bar.trade_date)
            bars_by_date.setdefault(bar.trade_date, {})[bar.symbol] = bar

        self._bars_by_symbol = bars_by_symbol
        self._bar_dates_by_symbol = bar_dates_by_symbol
        self._bars_by_date = bars_by_date

    @staticmethod
    def _row_to_bar(row: pd.Series) -> Bar:
        return Bar(
            symbol=row["symbol"],
            trade_date=row["trade_date"].date(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]) if pd.notna(row["volume"]) else 0.0,
            amount=float(row["amount"]) if "amount" in row and pd.notna(row["amount"]) else 0.0,
            paused=bool(row["is_suspended"]),
            limit_up=bool(row["is_limit_up"]),
            limit_down=bool(row["is_limit_down"]),
        )
