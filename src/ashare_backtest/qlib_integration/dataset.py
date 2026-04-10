from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_backtest.data import filter_universe_frame
from ashare_backtest.qlib_integration.config import QlibBaseConfig
from ashare_backtest.qlib_integration.features import build_feature_field_map
from ashare_backtest.qlib_integration.export import normalize_qlib_symbol


def require_qlib() -> tuple[object, object]:
    try:
        import qlib  # type: ignore[import-not-found]
        from qlib.data import D  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Qlib support requires the optional `pyqlib` dependency. "
            "Install it with `python -m pip install -e \".[qlib]\"` before running qlib commands."
        ) from exc
    return qlib, D


def initialize_qlib(config: QlibBaseConfig) -> None:
    qlib, _ = require_qlib()
    provider_uri = Path(config.provider_uri).expanduser().as_posix()
    init_kwargs = {"provider_uri": provider_uri}
    region = config.region.strip().lower()
    if region == "cn":
        try:
            from qlib.constant import REG_CN  # type: ignore[import-not-found]

            init_kwargs["region"] = REG_CN
        except Exception:
            init_kwargs["region"] = config.region
    else:
        init_kwargs["region"] = config.region
    qlib.init(**init_kwargs)


def load_qlib_market_frame(
    config: QlibBaseConfig,
    *,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    initialize_qlib(config)
    _, dataset_provider = require_qlib()

    field_map = build_feature_field_map(config.feature_specs)
    field_names = [name for name, _ in field_map] + ["label"]
    field_exprs = [expression for _, expression in field_map] + [config.label_expression]
    raw = dataset_provider.features(
        dataset_provider.instruments(config.market),
        field_exprs,
        start_time=start_date,
        end_time=end_date,
        freq=config.freq,
    )
    if raw.empty:
        raise ValueError(f"qlib dataset returned no rows for market={config.market} between {start_date} and {end_date}")

    frame = raw.reset_index()
    rename_map = {}
    if "instrument" in frame.columns:
        rename_map["instrument"] = "symbol"
    if "datetime" in frame.columns:
        rename_map["datetime"] = "trade_date"
    frame = frame.rename(columns=rename_map)

    derived_column_names = ["symbol", "trade_date", *field_names]
    if len(frame.columns) != len(derived_column_names):
        raise ValueError(
            "unexpected qlib dataset shape; expected symbol/trade_date plus feature and label columns"
        )
    frame.columns = derived_column_names
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["symbol"] = frame["symbol"].map(normalize_qlib_symbol)
    frame = _apply_project_universe_filter(frame, config)
    return frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _apply_project_universe_filter(frame: pd.DataFrame, config: QlibBaseConfig) -> pd.DataFrame:
    universe_name = str(config.universe_name or "").strip()
    if frame.empty or not universe_name:
        return frame

    bars_path = Path(config.storage_root) / "parquet" / "bars" / "daily.parquet"
    if not bars_path.exists():
        raise FileNotFoundError(f"bars parquet not found for universe filter: {bars_path}")

    bars = pd.read_parquet(bars_path, columns=["symbol", "trade_date", "amount", "is_suspended"]).copy()
    if bars.empty:
        return frame.iloc[0:0].copy()
    bars["trade_date"] = pd.to_datetime(bars["trade_date"], errors="coerce")
    bars["symbol"] = bars["symbol"].astype(str)
    merged = frame.merge(bars, on=["symbol", "trade_date"], how="left")
    filtered = filter_universe_frame(
        merged,
        storage_root=config.storage_root,
        universe_name=universe_name,
        date_column="trade_date",
    )
    return filtered.loc[:, frame.columns].copy()
