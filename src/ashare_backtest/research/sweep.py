from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ashare_backtest.engine import BacktestEngine
from ashare_backtest.research.score_workflow import (
    build_preloaded_score_provider,
    build_score_backtest_config,
    build_score_strategy,
)


@dataclass(frozen=True)
class SweepConfig:
    scores_path: str
    storage_root: str
    start_date: str
    end_date: str
    output_csv_path: str
    top_k_values: tuple[int, ...]
    rebalance_every_values: tuple[int, ...]
    min_hold_bars_values: tuple[int, ...]
    keep_buffer: int = 2
    min_turnover_names: int = 3
    min_daily_amount: float = 0.0
    max_names_per_industry: int = 0
    lookback_window: int = 20
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage_rate: float = 0.0005


@dataclass(frozen=True)
class _SweepRunConfig:
    scores_path: str
    storage_root: str
    top_k: int
    rebalance_every: int
    lookback_window: int
    min_hold_bars: int
    keep_buffer: int
    min_turnover_names: int
    min_daily_amount: float
    max_names_per_industry: int
    initial_cash: float
    commission_rate: float
    stamp_tax_rate: float
    slippage_rate: float


def run_model_sweep(config: SweepConfig) -> list[dict[str, float | int]]:
    scores = pd.read_parquet(config.scores_path)
    universe = tuple(sorted(scores["symbol"].astype(str).unique().tolist()))
    provider = build_preloaded_score_provider(
        storage_root=config.storage_root,
        universe=universe,
        start_date=date.fromisoformat(config.start_date),
        end_date=date.fromisoformat(config.end_date),
        lookback=config.lookback_window,
    )
    engine = BacktestEngine(provider)

    rows: list[dict[str, float | int]] = []
    for top_k in config.top_k_values:
        for rebalance_every in config.rebalance_every_values:
            for min_hold_bars in config.min_hold_bars_values:
                run_config = _SweepRunConfig(
                    scores_path=config.scores_path,
                    storage_root=config.storage_root,
                    top_k=top_k,
                    rebalance_every=rebalance_every,
                    lookback_window=config.lookback_window,
                    min_hold_bars=min_hold_bars,
                    keep_buffer=config.keep_buffer,
                    min_turnover_names=config.min_turnover_names,
                    min_daily_amount=config.min_daily_amount,
                    max_names_per_industry=config.max_names_per_industry,
                    initial_cash=config.initial_cash,
                    commission_rate=config.commission_rate,
                    stamp_tax_rate=config.stamp_tax_rate,
                    slippage_rate=config.slippage_rate,
                )
                strategy = build_score_strategy(run_config)
                backtest = build_score_backtest_config(
                    run_config,
                    universe=universe,
                    start_date=date.fromisoformat(config.start_date),
                    end_date=date.fromisoformat(config.end_date),
                    strategy_path="__model_score_sweep__",
                )
                result = engine.run_with_strategy(backtest, strategy)
                rows.append(
                    {
                        "top_k": top_k,
                        "rebalance_every": rebalance_every,
                        "min_hold_bars": min_hold_bars,
                        "total_return": result.total_return,
                        "annual_return": result.annual_return,
                        "max_drawdown": result.max_drawdown,
                        "sharpe_ratio": result.sharpe_ratio,
                        "turnover_ratio": result.turnover_ratio,
                        "trade_count": len(result.trades),
                    }
                )

    output_path = Path(config.output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "top_k",
                "rebalance_every",
                "min_hold_bars",
                "total_return",
                "annual_return",
                "max_drawdown",
                "sharpe_ratio",
                "turnover_ratio",
                "trade_count",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return rows
