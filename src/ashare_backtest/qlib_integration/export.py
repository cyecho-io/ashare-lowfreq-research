from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


_CANONICAL_SYMBOL_RE = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")
_QLIB_PREFIX_SYMBOL_RE = re.compile(r"^(SH|SZ|BJ)(\d{6})$", re.IGNORECASE)


def normalize_qlib_symbol(symbol: str) -> str:
    value = str(symbol).strip().upper()
    if _CANONICAL_SYMBOL_RE.fullmatch(value):
        return value
    match = _QLIB_PREFIX_SYMBOL_RE.fullmatch(value)
    if match:
        exchange, code = match.groups()
        return f"{code}.{exchange}"
    raise ValueError(f"unsupported qlib symbol format: {symbol}")


def export_score_frame(
    frame: pd.DataFrame,
    output_path: str,
    *,
    backend: str = "qlib",
    model: str = "",
    config_id: str = "",
) -> pd.DataFrame:
    required = {"trade_date", "symbol", "prediction"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"score frame is missing required columns: {', '.join(missing)}")

    exported = frame.copy()
    exported["trade_date"] = pd.to_datetime(exported["trade_date"]).dt.date.astype(str)
    exported["symbol"] = exported["symbol"].map(normalize_qlib_symbol)
    if "backend" not in exported.columns:
        exported["backend"] = backend
    if model and "model" not in exported.columns:
        exported["model"] = model
    if config_id and "config_id" not in exported.columns:
        exported["config_id"] = config_id

    ordered_columns = [
        column
        for column in (
            "trade_date",
            "symbol",
            "prediction",
            "label",
            "train_end_date",
            "validation_end_date",
            "backend",
            "model",
            "config_id",
        )
        if column in exported.columns
    ]
    ordered_columns.extend(column for column in exported.columns if column not in ordered_columns)
    exported = exported.loc[:, ordered_columns].sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    exported.to_parquet(destination, index=False)
    return exported
