from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ashare_backtest.data.importers import (
    DERIVED_UNIVERSE_ACTIVE,
    DERIVED_UNIVERSE_TRADABLE,
    TRADABLE_MAX_RECENT_SUSPENDED_DAYS,
    TRADABLE_MIN_LISTING_DAYS,
    TRADABLE_MIN_MEDIAN_DAILY_AMOUNT,
    TRADABLE_MIN_RECENT_TRADING_DAYS,
    TRADABLE_RECENT_WINDOW,
)


def _derive_universe_symbols(
    storage_root: str | Path,
    universe_name: str,
    as_of_date: str | date,
) -> tuple[str, ...]:
    instruments_path = Path(storage_root) / "parquet" / "instruments" / "ashare_instruments.parquet"
    if not instruments_path.exists():
        return ()

    instruments = pd.read_parquet(instruments_path).copy()
    if instruments.empty or "symbol" not in instruments.columns:
        return ()

    target_date = pd.Timestamp(as_of_date)
    if "listing_date" in instruments.columns:
        instruments["listing_date"] = pd.to_datetime(instruments["listing_date"], errors="coerce")
    else:
        instruments["listing_date"] = pd.NaT
    if "delisting_date" in instruments.columns:
        instruments["delisting_date"] = pd.to_datetime(instruments["delisting_date"], errors="coerce")
    else:
        instruments["delisting_date"] = pd.NaT
    if "is_active" not in instruments.columns:
        instruments["is_active"] = True
    else:
        instruments["is_active"] = instruments["is_active"].fillna(False).astype(bool)
    if "is_st" not in instruments.columns:
        instruments["is_st"] = False
    else:
        instruments["is_st"] = instruments["is_st"].fillna(False).astype(bool)

    active = instruments.loc[
        instruments["is_active"]
        & (instruments["listing_date"].isna() | (instruments["listing_date"] <= target_date))
        & (instruments["delisting_date"].isna() | (instruments["delisting_date"] >= target_date))
    ].copy()
    if active.empty:
        return ()

    if universe_name == DERIVED_UNIVERSE_ACTIVE:
        return tuple(sorted(active["symbol"].astype(str).unique().tolist()))

    if universe_name != DERIVED_UNIVERSE_TRADABLE:
        return ()

    tradable = active.loc[~active["is_st"]].copy()
    tradable["listing_days"] = (target_date - tradable["listing_date"]).dt.days
    tradable = tradable.loc[tradable["listing_days"].fillna(-1) >= TRADABLE_MIN_LISTING_DAYS].copy()
    if tradable.empty:
        return ()

    bars_path = Path(storage_root) / "parquet" / "bars" / "daily.parquet"
    if not bars_path.exists():
        return ()

    bars = pd.read_parquet(
        bars_path,
        columns=["symbol", "trade_date", "amount", "is_suspended"],
    ).copy()
    if bars.empty:
        return ()

    bars["trade_date"] = pd.to_datetime(bars["trade_date"], errors="coerce")
    bars["is_suspended"] = bars["is_suspended"].fillna(False).astype(bool)
    recent_start = target_date - pd.Timedelta(days=TRADABLE_RECENT_WINDOW * 2)
    recent_bars = bars.loc[
        (bars["symbol"].isin(tradable["symbol"]))
        & (bars["trade_date"].notna())
        & (bars["trade_date"] >= recent_start)
        & (bars["trade_date"] <= target_date)
    ].copy()
    if recent_bars.empty:
        return ()

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
    return tuple(sorted(tradable["symbol"].astype(str).unique().tolist()))


def filter_universe_frame(
    frame: pd.DataFrame,
    *,
    storage_root: str | Path,
    universe_name: str,
    date_column: str = "trade_date",
) -> pd.DataFrame:
    if frame.empty or not universe_name:
        return frame
    if date_column not in frame.columns or "symbol" not in frame.columns:
        raise ValueError(f"frame must include {date_column} and symbol columns")

    working = frame.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working = working.dropna(subset=[date_column]).copy()
    if working.empty:
        return working

    if universe_name in {DERIVED_UNIVERSE_ACTIVE, DERIVED_UNIVERSE_TRADABLE}:
        return _filter_derived_universe_frame(
            working,
            storage_root=storage_root,
            universe_name=universe_name,
            date_column=date_column,
        )
    return _filter_membership_universe_frame(
        working,
        storage_root=storage_root,
        universe_name=universe_name,
        date_column=date_column,
    )


def _filter_derived_universe_frame(
    frame: pd.DataFrame,
    *,
    storage_root: str | Path,
    universe_name: str,
    date_column: str,
) -> pd.DataFrame:
    instruments_path = Path(storage_root) / "parquet" / "instruments" / "ashare_instruments.parquet"
    if not instruments_path.exists():
        return frame.iloc[0:0].copy()

    instruments = pd.read_parquet(
        instruments_path,
        columns=["symbol", "listing_date", "delisting_date", "is_st", "is_active"],
    ).copy()
    if instruments.empty:
        return frame.iloc[0:0].copy()

    instruments["listing_date"] = pd.to_datetime(instruments["listing_date"], errors="coerce")
    instruments["delisting_date"] = pd.to_datetime(instruments["delisting_date"], errors="coerce")
    instruments["is_st"] = instruments["is_st"].fillna(False).astype(bool)
    instruments["is_active"] = instruments["is_active"].fillna(False).astype(bool)

    working = frame.merge(instruments, on="symbol", how="left")
    active_mask = (
        working["is_active"].fillna(False)
        & (working["listing_date"].isna() | (working["listing_date"] <= working[date_column]))
        & (working["delisting_date"].isna() | (working["delisting_date"] >= working[date_column]))
    )
    if universe_name == DERIVED_UNIVERSE_ACTIVE:
        return working.loc[active_mask, frame.columns].copy()

    working = working.loc[active_mask].copy()
    if working.empty:
        return working.loc[:, frame.columns].copy()
    if "amount" not in working.columns or "is_suspended" not in working.columns:
        raise ValueError("derived tradable universe filtering requires amount and is_suspended columns")

    working = working.sort_values(["symbol", date_column]).copy()
    working["is_suspended"] = pd.Series(working["is_suspended"], index=working.index, dtype="boolean").fillna(False)
    working["listing_days"] = (working[date_column] - working["listing_date"]).dt.days
    grouped = working.groupby("symbol", group_keys=False)
    working["recent_trading_days"] = grouped[date_column].transform(
        lambda s: s.rolling(TRADABLE_RECENT_WINDOW, min_periods=1).count()
    )
    working["recent_suspended_days"] = grouped["is_suspended"].transform(
        lambda s: s.astype(int).rolling(TRADABLE_RECENT_WINDOW, min_periods=1).sum()
    )
    working["median_amount"] = grouped["amount"].transform(
        lambda s: s.rolling(TRADABLE_RECENT_WINDOW, min_periods=1).median()
    )
    tradable_mask = (
        (~working["is_st"].fillna(False))
        & (working["listing_days"].fillna(-1) >= TRADABLE_MIN_LISTING_DAYS)
        & (working["recent_trading_days"].fillna(0) >= TRADABLE_MIN_RECENT_TRADING_DAYS)
        & (working["recent_suspended_days"].fillna(TRADABLE_RECENT_WINDOW) <= TRADABLE_MAX_RECENT_SUSPENDED_DAYS)
        & (~working["is_suspended"].fillna(True).astype(bool))
        & (working["median_amount"].fillna(0.0) >= TRADABLE_MIN_MEDIAN_DAILY_AMOUNT)
    )
    return working.loc[tradable_mask, frame.columns].copy()


def _filter_membership_universe_frame(
    frame: pd.DataFrame,
    *,
    storage_root: str | Path,
    universe_name: str,
    date_column: str,
) -> pd.DataFrame:
    memberships_path = Path(storage_root) / "parquet" / "universe" / "memberships.parquet"
    if not memberships_path.exists():
        raise FileNotFoundError(f"universe memberships not found: {memberships_path}")

    memberships = pd.read_parquet(
        memberships_path,
        columns=["universe_name", "symbol", "effective_date", "expiry_date"],
    ).copy()
    memberships = memberships.loc[memberships["universe_name"] == universe_name].copy()
    if memberships.empty:
        return frame.iloc[0:0].copy()

    memberships["effective_date"] = pd.to_datetime(memberships["effective_date"], errors="coerce")
    memberships["expiry_date"] = pd.to_datetime(memberships["expiry_date"], errors="coerce")

    indexed = frame.reset_index(drop=False).rename(columns={"index": "_row_id"})
    merged = indexed[["_row_id", "symbol", date_column]].merge(memberships, on="symbol", how="left")
    active_mask = (
        merged["universe_name"].notna()
        & (merged["effective_date"].isna() | (merged["effective_date"] <= merged[date_column]))
        & (merged["expiry_date"].isna() | (merged["expiry_date"] >= merged[date_column]))
    )
    valid_ids = merged.loc[active_mask, "_row_id"].unique().tolist()
    if not valid_ids:
        return frame.iloc[0:0].copy()
    return indexed.loc[indexed["_row_id"].isin(valid_ids), frame.columns].copy()


def load_universe_symbols(
    storage_root: str | Path,
    universe_name: str,
    as_of_date: str | date | None = None,
) -> tuple[str, ...]:
    if as_of_date is not None and universe_name in {DERIVED_UNIVERSE_ACTIVE, DERIVED_UNIVERSE_TRADABLE}:
        derived_symbols = _derive_universe_symbols(storage_root, universe_name, as_of_date)
        if derived_symbols:
            return derived_symbols

    memberships_path = Path(storage_root) / "parquet" / "universe" / "memberships.parquet"
    if not memberships_path.exists():
        raise FileNotFoundError(f"universe memberships not found: {memberships_path}")

    frame = pd.read_parquet(
        memberships_path,
        columns=["universe_name", "symbol", "effective_date", "expiry_date"],
    )
    if frame.empty:
        return ()

    filtered = frame.loc[frame["universe_name"] == universe_name].copy()
    if filtered.empty:
        if as_of_date is not None and universe_name in {DERIVED_UNIVERSE_ACTIVE, DERIVED_UNIVERSE_TRADABLE}:
            return _derive_universe_symbols(storage_root, universe_name, as_of_date)
        return ()

    for column in ("effective_date", "expiry_date"):
        filtered[column] = pd.to_datetime(filtered[column], errors="coerce")

    target_date: pd.Timestamp | None = None
    if as_of_date is not None:
        target_date = pd.Timestamp(as_of_date)
        filtered = filtered.loc[
            (filtered["effective_date"].isna() | (filtered["effective_date"] <= target_date))
            & (filtered["expiry_date"].isna() | (filtered["expiry_date"] >= target_date))
        ]

    if filtered.empty:
        if as_of_date is not None and universe_name in {DERIVED_UNIVERSE_ACTIVE, DERIVED_UNIVERSE_TRADABLE}:
            return _derive_universe_symbols(storage_root, universe_name, as_of_date)
        return ()

    if target_date is None:
        filtered = filtered.sort_values(["symbol", "effective_date"])
        latest = filtered.groupby("symbol", as_index=False).tail(1)
        return tuple(sorted(latest["symbol"].astype(str).unique().tolist()))

    return tuple(sorted(filtered["symbol"].astype(str).unique().tolist()))
