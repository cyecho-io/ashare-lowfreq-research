from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


FIELDS = ("open", "high", "low", "close", "volume", "amount")


def _write_feature_bin(path: Path, values: pd.Series, start_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = np.hstack([[float(start_index)], values.astype("float32").to_numpy(dtype=np.float32)]).astype("<f4")
    payload.tofile(path)


def build_provider(storage_root: Path, provider_uri: Path, market: str) -> None:
    bars_path = storage_root / "parquet" / "bars" / "daily.parquet"
    calendar_path = storage_root / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    if not bars_path.exists():
        raise FileNotFoundError(f"bars parquet not found: {bars_path}")
    if not calendar_path.exists():
        raise FileNotFoundError(f"calendar parquet not found: {calendar_path}")

    bars = pd.read_parquet(bars_path, columns=["symbol", "trade_date", *FIELDS]).copy()
    calendar = pd.read_parquet(calendar_path, columns=["trade_date", "is_open"]).copy()
    bars["trade_date"] = pd.to_datetime(bars["trade_date"], errors="coerce")
    calendar["trade_date"] = pd.to_datetime(calendar["trade_date"], errors="coerce")
    bars = bars.dropna(subset=["symbol", "trade_date"]).sort_values(["symbol", "trade_date"])
    calendar = calendar.dropna(subset=["trade_date"])
    open_dates = (
        calendar.loc[calendar["is_open"].astype(bool), "trade_date"]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    if open_dates.empty:
        raise ValueError("demo calendar contains no open trade dates")

    if provider_uri.exists():
        shutil.rmtree(provider_uri)
    (provider_uri / "calendars").mkdir(parents=True, exist_ok=True)
    (provider_uri / "instruments").mkdir(parents=True, exist_ok=True)
    (provider_uri / "features").mkdir(parents=True, exist_ok=True)

    open_dates.dt.strftime("%Y-%m-%d").to_csv(provider_uri / "calendars" / "day.txt", index=False, header=False)
    date_index = pd.Index(open_dates)
    instrument_rows: list[tuple[str, str, str]] = []

    for symbol, symbol_frame in bars.groupby("symbol", sort=True):
        symbol_text = str(symbol).upper()
        symbol_frame = symbol_frame.drop_duplicates("trade_date").set_index("trade_date").sort_index()
        valid_dates = symbol_frame.index.intersection(date_index)
        if valid_dates.empty:
            continue

        start_date = pd.Timestamp(valid_dates.min())
        end_date = pd.Timestamp(valid_dates.max())
        start_index = int(date_index.get_loc(start_date))
        end_index = int(date_index.get_loc(end_date))
        aligned_index = date_index[start_index : end_index + 1]
        aligned = symbol_frame.reindex(aligned_index)

        feature_dir = provider_uri / "features" / symbol_text.lower()
        for field in FIELDS:
            _write_feature_bin(feature_dir / f"{field}.day.bin", aligned[field], start_index)

        instrument_rows.append((symbol_text, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))

    if not instrument_rows:
        raise ValueError("demo bars produced no qlib instruments")

    instrument_frame = pd.DataFrame(instrument_rows, columns=["instrument", "start", "end"])
    for name in (market, "all"):
        instrument_frame.to_csv(provider_uri / "instruments" / f"{name}.txt", sep="\t", index=False, header=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a tiny qlib provider from storage/demo parquet files.")
    parser.add_argument("--storage-root", default="storage/demo")
    parser.add_argument("--provider-uri", default="storage/demo/qlib_data/cn_data")
    parser.add_argument("--market", default="demo")
    args = parser.parse_args()

    build_provider(Path(args.storage_root), Path(args.provider_uri), args.market)
    print(f"demo qlib provider written to {args.provider_uri}")


if __name__ == "__main__":
    main()
