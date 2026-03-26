from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def load_universe_symbols(
    storage_root: str | Path,
    universe_name: str,
    as_of_date: str | date | None = None,
) -> tuple[str, ...]:
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
        return ()

    if target_date is None:
        filtered = filtered.sort_values(["symbol", "effective_date"])
        latest = filtered.groupby("symbol", as_index=False).tail(1)
        return tuple(sorted(latest["symbol"].astype(str).unique().tolist()))

    return tuple(sorted(filtered["symbol"].astype(str).unique().tolist()))
