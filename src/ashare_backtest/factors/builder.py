from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ashare_backtest.data import filter_universe_frame, load_universe_symbols
from ashare_backtest.logging_utils import get_logger


def resolve_factor_snapshot_path(
    factor_spec_id: str,
    as_of_date: str,
    universe_name: str = "",
    start_date: str = "",
    root: str | Path = "research/factors",
) -> str:
    base = Path(root) / factor_spec_id
    if universe_name and start_date:
        return (base / universe_name / f"start_{start_date}" / f"asof_{as_of_date}.parquet").as_posix()
    return (base / f"{as_of_date}.parquet").as_posix()


@dataclass(frozen=True)
class FactorBuildConfig:
    storage_root: str
    output_path: str
    symbols: tuple[str, ...] = ()
    universe_name: str = ""
    start_date: str | None = None
    as_of_date: str | None = None


class FactorBuilder:
    BARS_COLUMNS = ["symbol", "trade_date", "high", "low", "close", "volume", "amount"]
    INSTRUMENT_COLUMNS = ["symbol", "industry_level_1"]

    def __init__(self, config: FactorBuildConfig) -> None:
        self.config = config
        self.storage_root = Path(config.storage_root)
        self.bars_path = self.storage_root / "parquet" / "bars" / "daily.parquet"
        self.instruments_path = self.storage_root / "parquet" / "instruments" / "ashare_instruments.parquet"
        self.logger = get_logger("factors.builder")

    def build(self) -> pd.DataFrame:
        self.logger.info(
            "build factor panel start output=%s universe_name=%s as_of_date=%s start_date=%s",
            self.config.output_path,
            self.config.universe_name or "-",
            self.config.as_of_date or "-",
            self.config.start_date or "-",
        )
        symbols = self.config.symbols
        as_of_date = self.config.as_of_date
        frame = pd.read_parquet(self.bars_path, columns=self.BARS_COLUMNS)
        if self.config.start_date:
            # Keep pre-start history so long-window features are available on the first in-sample dates.
            feature_history_start = pd.Timestamp(self.config.start_date) - pd.Timedelta(days=180)
            frame = frame.loc[frame["trade_date"] >= feature_history_start]
        if as_of_date:
            frame = frame.loc[frame["trade_date"] <= pd.Timestamp(as_of_date)]
        if self.config.universe_name:
            symbols = load_universe_symbols(
                storage_root=self.storage_root,
                universe_name=self.config.universe_name,
                as_of_date=as_of_date,
            )
        if symbols:
            frame = frame.loc[frame["symbol"].isin(symbols)]
        frame = frame.sort_values(["symbol", "trade_date"]).copy()

        instruments = pd.read_parquet(self.instruments_path, columns=self.INSTRUMENT_COLUMNS)
        frame = frame.merge(instruments, on="symbol", how="left")

        panel = self._build_factor_panel(frame)
        if self.config.universe_name:
            panel = filter_universe_frame(
                panel,
                storage_root=self.storage_root,
                universe_name=self.config.universe_name,
            )
        if self.config.start_date:
            panel = panel.loc[panel["trade_date"] >= pd.Timestamp(self.config.start_date)].copy()
        if as_of_date:
            panel = panel.loc[panel["trade_date"] <= pd.Timestamp(as_of_date)].copy()
        panel = panel.drop(columns=["is_suspended"], errors="ignore")
        output_path = Path(self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(output_path, index=False)
        self.logger.info(
            "build factor panel complete output=%s rows=%s symbols=%s",
            output_path.as_posix(),
            len(panel),
            panel["symbol"].nunique() if not panel.empty else 0,
        )
        return panel

    @staticmethod
    def _build_factor_panel(frame: pd.DataFrame) -> pd.DataFrame:
        grouped = frame.groupby("symbol", group_keys=False)

        frame["ret_1"] = grouped["close"].pct_change()
        frame["mom_5"] = grouped["close"].pct_change(5)
        frame["mom_10"] = grouped["close"].pct_change(10)
        frame["mom_20"] = grouped["close"].pct_change(20)
        frame["mom_60"] = grouped["close"].pct_change(60)

        ma_5 = grouped["close"].transform(lambda s: s.rolling(5).mean())
        ma_10 = grouped["close"].transform(lambda s: s.rolling(10).mean())
        ma_20 = grouped["close"].transform(lambda s: s.rolling(20).mean())
        ma_60 = grouped["close"].transform(lambda s: s.rolling(60).mean())
        frame["ma_gap_5"] = frame["close"] / ma_5 - 1
        frame["ma_gap_10"] = frame["close"] / ma_10 - 1
        frame["ma_gap_20"] = frame["close"] / ma_20 - 1
        frame["ma_gap_60"] = frame["close"] / ma_60 - 1

        frame["volatility_10"] = grouped["ret_1"].transform(lambda s: s.rolling(10).std())
        frame["volatility_20"] = grouped["ret_1"].transform(lambda s: s.rolling(20).std())
        frame["volatility_60"] = grouped["ret_1"].transform(lambda s: s.rolling(60).std())
        intraday_range = frame["high"] / frame["low"] - 1
        frame["range_ratio_5"] = intraday_range.groupby(frame["symbol"]).transform(lambda s: s.rolling(5).mean())

        volume_ma_5 = grouped["volume"].transform(lambda s: s.rolling(5).mean())
        volume_ma_20 = grouped["volume"].transform(lambda s: s.rolling(20).mean())
        amount_ma_5 = grouped["amount"].transform(lambda s: s.rolling(5).mean())
        amount_ma_20 = grouped["amount"].transform(lambda s: s.rolling(20).mean())
        frame["volume_ratio_5_20"] = volume_ma_5 / volume_ma_20
        frame["amount_ratio_5_20"] = amount_ma_5 / amount_ma_20
        frame["amount_mom_10"] = grouped["amount"].pct_change(10)

        rolling_high_20 = grouped["close"].transform(lambda s: s.rolling(20).max())
        rolling_low_20 = grouped["close"].transform(lambda s: s.rolling(20).min())
        frame["price_pos_20"] = (frame["close"] - rolling_low_20) / (rolling_high_20 - rolling_low_20)
        frame["volatility_ratio_10_60"] = frame["volatility_10"] / frame["volatility_60"]
        frame["trend_strength_20"] = frame["mom_20"] / frame["volatility_20"]

        frame["cross_rank_mom_20"] = frame.groupby("trade_date")["mom_20"].rank(pct=True)
        frame["cross_rank_amount_ratio_5_20"] = frame.groupby("trade_date")["amount_ratio_5_20"].rank(pct=True)
        frame["cross_rank_volatility_20"] = frame.groupby("trade_date")["volatility_20"].rank(pct=True, ascending=True)

        frame["fwd_return_3"] = grouped["close"].transform(lambda s: s.shift(-3) / s - 1)
        frame["fwd_return_5"] = grouped["close"].transform(lambda s: s.shift(-5) / s - 1)
        frame["fwd_return_10"] = grouped["close"].transform(lambda s: s.shift(-10) / s - 1)
        frame["excess_fwd_return_3"] = frame["fwd_return_3"] - frame.groupby("trade_date")["fwd_return_3"].transform(
            "mean"
        )
        frame["excess_fwd_return_5"] = frame["fwd_return_5"] - frame.groupby("trade_date")["fwd_return_5"].transform(
            "mean"
        )
        frame["excess_fwd_return_10"] = frame["fwd_return_10"] - frame.groupby("trade_date")[
            "fwd_return_10"
        ].transform("mean")
        frame["industry_excess_fwd_return_3"] = frame["fwd_return_3"] - frame.groupby(
            ["trade_date", "industry_level_1"]
        )["fwd_return_3"].transform("mean")
        frame["industry_excess_fwd_return_5"] = frame["fwd_return_5"] - frame.groupby(
            ["trade_date", "industry_level_1"]
        )["fwd_return_5"].transform("mean")
        frame["industry_excess_fwd_return_10"] = frame["fwd_return_10"] - frame.groupby(
            ["trade_date", "industry_level_1"]
        )["fwd_return_10"].transform("mean")

        columns = [
            "trade_date",
            "symbol",
            "industry_level_1",
            "close",
            "volume",
            "amount",
            "is_suspended",
            "mom_5",
            "mom_10",
            "mom_20",
            "mom_60",
            "ma_gap_5",
            "ma_gap_10",
            "ma_gap_20",
            "ma_gap_60",
            "volatility_10",
            "volatility_20",
            "volatility_60",
            "range_ratio_5",
            "volume_ratio_5_20",
            "amount_ratio_5_20",
            "amount_mom_10",
            "price_pos_20",
            "volatility_ratio_10_60",
            "trend_strength_20",
            "cross_rank_mom_20",
            "cross_rank_amount_ratio_5_20",
            "cross_rank_volatility_20",
            "fwd_return_3",
            "fwd_return_5",
            "fwd_return_10",
            "excess_fwd_return_3",
            "excess_fwd_return_5",
            "excess_fwd_return_10",
            "industry_excess_fwd_return_3",
            "industry_excess_fwd_return_5",
            "industry_excess_fwd_return_10",
        ]
        return frame.loc[:, columns].reset_index(drop=True)
