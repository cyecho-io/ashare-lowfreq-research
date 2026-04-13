from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_backtest.data import filter_universe_frame
from ashare_backtest.qlib_integration.config import QlibBaseConfig
from ashare_backtest.qlib_integration.features import build_feature_field_map
from ashare_backtest.qlib_integration.export import normalize_qlib_symbol


_DERIVED_FEATURE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "amount_ratio_5_20": (),
    "amount_mom_10": (),
    "cross_rank_mom_20": ("mom_20",),
    "cross_rank_amount_ratio_5_20": ("amount_ratio_5_20",),
    "cross_rank_volatility_20": ("volatility_20",),
}


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

    requested_field_map = build_feature_field_map(config.feature_specs)
    requested_field_names = [name for name, _ in requested_field_map]

    query_field_map: list[tuple[str, str]] = []
    for name, expression in requested_field_map:
        if name in _DERIVED_FEATURE_DEPENDENCIES:
            continue
        query_field_map.append((name, expression))
    for derived_name, dependencies in _DERIVED_FEATURE_DEPENDENCIES.items():
        if derived_name not in requested_field_names:
            continue
        available_names = set(requested_field_names)
        missing_dependencies = [dependency for dependency in dependencies if dependency not in available_names]
        if missing_dependencies:
            raise ValueError(
                f"derived qlib feature {derived_name} requires dependencies: {', '.join(missing_dependencies)}"
            )

    query_label_name = _resolve_query_label_name(config)
    query_field_names = [name for name, _ in query_field_map] + [query_label_name]
    field_exprs = [expression for _, expression in query_field_map] + [config.label_expression]
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

    derived_column_names = ["symbol", "trade_date", *query_field_names]
    if len(frame.columns) != len(derived_column_names):
        raise ValueError(
            "unexpected qlib dataset shape; expected symbol/trade_date plus feature and label columns"
        )
    frame.columns = derived_column_names
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["symbol"] = frame["symbol"].map(normalize_qlib_symbol)
    if query_label_name != "label":
        frame = frame.rename(columns={query_label_name: "label"})
    frame = _materialize_project_time_series_features(frame, requested_field_names, config.storage_root)
    frame = _apply_project_universe_filter(frame, config)
    frame = _materialize_project_cross_sectional_features(frame, requested_field_names)
    frame = _materialize_project_label(frame, config, storage_root=config.storage_root)
    return frame.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _resolve_query_label_name(config: QlibBaseConfig) -> str:
    if str(config.label_mode).strip().lower() == "industry_excess_fwd_return_5":
        return "raw_label"
    return "label"


def _materialize_project_time_series_features(
    frame: pd.DataFrame,
    requested_field_names: list[str],
    storage_root: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    enriched = frame.copy()
    amount_requested = any(
        name in requested_field_names
        for name in ("amount_ratio_5_20", "amount_mom_10", "cross_rank_amount_ratio_5_20")
    )
    if amount_requested:
        bars_path = Path(storage_root) / "parquet" / "bars" / "daily.parquet"
        bars = pd.read_parquet(bars_path, columns=["symbol", "trade_date", "amount"]).copy()
        bars["trade_date"] = pd.to_datetime(bars["trade_date"], errors="coerce")
        bars["symbol"] = bars["symbol"].astype(str)
        enriched = enriched.merge(bars, on=["symbol", "trade_date"], how="left", suffixes=("", "_project"))
        grouped = enriched.groupby("symbol", group_keys=False)
        if "amount_ratio_5_20" in requested_field_names or "cross_rank_amount_ratio_5_20" in requested_field_names:
            amount_ma_5 = grouped["amount"].transform(lambda s: s.rolling(5).mean())
            amount_ma_20 = grouped["amount"].transform(lambda s: s.rolling(20).mean())
            enriched["amount_ratio_5_20"] = amount_ma_5 / amount_ma_20
        if "amount_mom_10" in requested_field_names:
            enriched["amount_mom_10"] = grouped["amount"].pct_change(10, fill_method=None)
        enriched = enriched.drop(columns=["amount"], errors="ignore")
    return enriched


def _materialize_project_cross_sectional_features(
    frame: pd.DataFrame,
    requested_field_names: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return frame

    enriched = frame.copy()
    if "cross_rank_mom_20" in requested_field_names:
        enriched["cross_rank_mom_20"] = enriched.groupby("trade_date")["mom_20"].rank(pct=True)
    if "cross_rank_amount_ratio_5_20" in requested_field_names:
        enriched["cross_rank_amount_ratio_5_20"] = enriched.groupby("trade_date")["amount_ratio_5_20"].rank(pct=True)
    if "cross_rank_volatility_20" in requested_field_names:
        enriched["cross_rank_volatility_20"] = enriched.groupby("trade_date")["volatility_20"].rank(
            pct=True,
            ascending=True,
        )
    return enriched


def _materialize_project_label(frame: pd.DataFrame, config: QlibBaseConfig, *, storage_root: str) -> pd.DataFrame:
    label_mode = str(config.label_mode).strip().lower()
    if label_mode in {"", "raw_fwd_return_5"}:
        return frame
    if label_mode != "industry_excess_fwd_return_5":
        raise ValueError(f"unsupported qlib label_mode: {config.label_mode}")

    if "label" not in frame.columns:
        raise ValueError("qlib frame is missing label column")

    instruments_path = Path(storage_root) / "parquet" / "instruments" / "ashare_instruments.parquet"
    instruments = pd.read_parquet(instruments_path, columns=["symbol", "industry_level_1"]).copy()
    instruments["symbol"] = instruments["symbol"].astype(str)

    enriched = frame.merge(instruments, on="symbol", how="left")
    if "industry_level_1" not in enriched.columns:
        raise ValueError("industry_level_1 is required for industry_excess_fwd_return_5 label_mode")
    industry_mean = enriched.groupby(["trade_date", "industry_level_1"], dropna=False)["label"].transform("mean")
    enriched["label"] = enriched["label"] - industry_mean
    return enriched


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
