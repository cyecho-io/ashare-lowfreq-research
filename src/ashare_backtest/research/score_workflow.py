from __future__ import annotations

from dataclasses import fields
from datetime import date

import pandas as pd

from ashare_backtest.data import DataProvider, ParquetDataProvider, load_universe_symbols
from ashare_backtest.protocol import BacktestConfig
from ashare_backtest.research.score_strategy import ScoreStrategyConfig, ScoreTopKStrategy


def load_score_symbols(scores_path: str) -> tuple[str, ...]:
    scores = pd.read_parquet(scores_path, columns=["symbol"])
    return tuple(sorted(scores["symbol"].astype(str).unique().tolist()))


def build_score_strategy_config(config: object) -> ScoreStrategyConfig:
    values = {
        "scores_path": str(getattr(config, "scores_path")),
        "storage_root": str(getattr(config, "storage_root", "storage")),
    }
    for field in fields(ScoreStrategyConfig):
        if field.name in values:
            continue
        if hasattr(config, field.name):
            values[field.name] = getattr(config, field.name)
    return ScoreStrategyConfig(**values)


def build_score_strategy(config: object) -> ScoreTopKStrategy:
    return ScoreTopKStrategy(build_score_strategy_config(config))


def build_score_backtest_config(
    config: object,
    universe: tuple[str, ...],
    start_date: date,
    end_date: date,
    *,
    strategy_path: str = "__model_score__",
) -> BacktestConfig:
    values = {
        "strategy_path": strategy_path,
        "start_date": start_date,
        "end_date": end_date,
        "universe": universe,
    }
    for field in fields(BacktestConfig):
        if field.name in values:
            continue
        if hasattr(config, field.name):
            values[field.name] = getattr(config, field.name)
    return BacktestConfig(**values)


def resolve_score_universe(
    config: object,
    scores: pd.DataFrame,
    *,
    as_of_date: date,
) -> tuple[str, ...]:
    score_symbols = tuple(sorted(scores["symbol"].astype(str).unique().tolist()))
    universe_name = str(getattr(config, "universe_name", "") or "")
    if not universe_name:
        return score_symbols

    storage_root = str(getattr(config, "storage_root", "storage"))
    allowed_symbols = set(load_universe_symbols(storage_root, universe_name, as_of_date=as_of_date.isoformat()))
    if not allowed_symbols:
        raise ValueError(f"universe {universe_name} has no members on {as_of_date.isoformat()}")

    filtered = tuple(symbol for symbol in score_symbols if symbol in allowed_symbols)
    if not filtered:
        raise ValueError(f"scores parquet has no symbols in universe {universe_name} on {as_of_date.isoformat()}")
    return filtered


def build_preloaded_score_provider(
    *,
    storage_root: str,
    universe: tuple[str, ...],
    start_date: date,
    end_date: date,
    lookback: int,
    provider: DataProvider | None = None,
) -> DataProvider:
    resolved_provider = provider or ParquetDataProvider(storage_root)
    resolved_provider.preload(
        symbols=universe,
        start_date=start_date,
        end_date=end_date,
        lookback=lookback,
    )
    return resolved_provider
