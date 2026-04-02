from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ashare_backtest.data import ParquetDataProvider
from ashare_backtest.engine import BacktestEngine
from ashare_backtest.engine.loader import load_strategy
from ashare_backtest.protocol import BacktestConfig
from ashare_backtest.reporting import export_backtest_result
from ashare_backtest.research.services import ModelBacktestServiceConfig, run_model_backtest_service


def run_backtest(backtest: BacktestConfig, storage_root: str, output_dir: str) -> None:
    provider = ParquetDataProvider(storage_root)
    strategy = load_strategy(backtest.strategy_path)
    provider.preload(
        symbols=backtest.universe,
        start_date=backtest.start_date,
        end_date=backtest.end_date,
        lookback=strategy.metadata.lookback_window,
    )
    engine = BacktestEngine(provider)
    result = engine.run(backtest)
    export_backtest_result(result, output_dir)
    print(
        "RESULT "
        f"total_return={result.total_return:.4f} "
        f"annual_return={result.annual_return:.4f} "
        f"max_drawdown={result.max_drawdown:.4f} "
        f"sharpe={result.sharpe_ratio:.4f} "
        f"trades={len(result.trades)} "
        f"output={output_dir}"
    )


def list_universes(storage_root: str) -> None:
    memberships_path = Path(storage_root) / "parquet" / "universe" / "memberships.parquet"
    frame = pd.read_parquet(
        memberships_path,
        columns=["universe_name", "symbol", "effective_date", "expiry_date"],
    )
    if frame.empty:
        print(f"NO_UNIVERSES storage={storage_root}")
        return

    for column in ("effective_date", "expiry_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")

    summary = (
        frame.groupby("universe_name", dropna=False)
        .agg(
            symbol_count=("symbol", "nunique"),
            effective_from=("effective_date", "min"),
            effective_to=("expiry_date", "max"),
        )
        .reset_index()
        .sort_values("universe_name")
    )
    for _, row in summary.iterrows():
        start = row["effective_from"].date().isoformat() if pd.notna(row["effective_from"]) else "-"
        end = row["effective_to"].date().isoformat() if pd.notna(row["effective_to"]) else "-"
        print(
            "UNIVERSE "
            f"name={row['universe_name']} "
            f"symbols={int(row['symbol_count'])} "
            f"effective_from={start} "
            f"effective_to={end}"
        )


def run_model_backtest(
    scores_path: str,
    storage_root: str,
    start_date: str,
    end_date: str,
    top_k: int,
    rebalance_every: int,
    lookback_window: int,
    min_hold_bars: int,
    keep_buffer: int,
    min_turnover_names: int,
    min_daily_amount: float,
    max_close_price: float,
    max_names_per_industry: int,
    max_position_weight: float,
    exit_policy: str,
    grace_rank_buffer: int,
    grace_momentum_window: int,
    grace_min_return: float,
    trailing_stop_window: int,
    trailing_stop_drawdown: float,
    trailing_stop_min_gain: float,
    score_reversal_confirm_days: int,
    score_reversal_threshold: float,
    hybrid_price_window: int,
    hybrid_price_threshold: float,
    strong_keep_extra_buffer: int,
    strong_keep_momentum_window: int,
    strong_keep_min_return: float,
    strong_trim_slowdown: float,
    strong_trim_momentum_window: int,
    strong_trim_min_return: float,
    initial_cash: float,
    commission_rate: float,
    stamp_tax_rate: float,
    slippage_rate: float,
    max_trade_participation_rate: float,
    max_pending_days: int,
    output_dir: str,
) -> None:
    run_model_backtest_service(
        config=ModelBacktestServiceConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=top_k,
            rebalance_every=rebalance_every,
            lookback_window=lookback_window,
            min_hold_bars=min_hold_bars,
            keep_buffer=keep_buffer,
            min_turnover_names=min_turnover_names,
            min_daily_amount=min_daily_amount,
            max_close_price=max_close_price,
            max_names_per_industry=max_names_per_industry,
            max_position_weight=max_position_weight,
            exit_policy=exit_policy,
            grace_rank_buffer=grace_rank_buffer,
            grace_momentum_window=grace_momentum_window,
            grace_min_return=grace_min_return,
            trailing_stop_window=trailing_stop_window,
            trailing_stop_drawdown=trailing_stop_drawdown,
            trailing_stop_min_gain=trailing_stop_min_gain,
            score_reversal_confirm_days=score_reversal_confirm_days,
            score_reversal_threshold=score_reversal_threshold,
            hybrid_price_window=hybrid_price_window,
            hybrid_price_threshold=hybrid_price_threshold,
            strong_keep_extra_buffer=strong_keep_extra_buffer,
            strong_keep_momentum_window=strong_keep_momentum_window,
            strong_keep_min_return=strong_keep_min_return,
            strong_trim_slowdown=strong_trim_slowdown,
            strong_trim_momentum_window=strong_trim_momentum_window,
            strong_trim_min_return=strong_trim_min_return,
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
            slippage_rate=slippage_rate,
            max_trade_participation_rate=max_trade_participation_rate,
            max_pending_days=max_pending_days,
        ),
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
    )
