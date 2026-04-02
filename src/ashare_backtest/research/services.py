from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

import pandas as pd

from ashare_backtest.engine import BacktestEngine
from ashare_backtest.reporting import export_backtest_result
from ashare_backtest.research.analysis import StrategyStateConfig, generate_strategy_state
from ashare_backtest.research.score_workflow import (
    build_preloaded_score_provider,
    build_score_backtest_config,
    build_score_strategy,
    load_score_symbols,
)
from ashare_backtest.research.trainer import (
    WalkForwardAsOfDateConfig,
    WalkForwardSingleDateConfig,
    train_lightgbm_walk_forward_as_of_date,
    train_lightgbm_walk_forward_single_date,
)
from ashare_backtest.cli.research_config import load_research_config, resolve_dated_output_path
from ashare_backtest.factors import resolve_factor_snapshot_path


@dataclass(frozen=True)
class ModelBacktestServiceConfig:
    scores_path: str
    storage_root: str
    top_k: int
    rebalance_every: int
    lookback_window: int
    min_hold_bars: int
    keep_buffer: int
    min_turnover_names: int
    min_daily_amount: float
    max_close_price: float
    max_names_per_industry: int
    max_position_weight: float
    exit_policy: str
    grace_rank_buffer: int
    grace_momentum_window: int
    grace_min_return: float
    trailing_stop_window: int
    trailing_stop_drawdown: float
    trailing_stop_min_gain: float
    score_reversal_confirm_days: int
    score_reversal_threshold: float
    hybrid_price_window: int
    hybrid_price_threshold: float
    strong_keep_extra_buffer: int
    strong_keep_momentum_window: int
    strong_keep_min_return: float
    strong_trim_slowdown: float
    strong_trim_momentum_window: int
    strong_trim_min_return: float
    initial_cash: float
    commission_rate: float
    stamp_tax_rate: float
    slippage_rate: float
    max_trade_participation_rate: float
    max_pending_days: int


def run_model_backtest_service(
    *,
    config: ModelBacktestServiceConfig,
    start_date: str,
    end_date: str,
    output_dir: str,
) -> None:
    universe = load_score_symbols(config.scores_path)
    strategy = build_score_strategy(config)
    backtest = build_score_backtest_config(
        config,
        universe=universe,
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
    )
    provider = build_preloaded_score_provider(
        storage_root=config.storage_root,
        universe=universe,
        start_date=backtest.start_date,
        end_date=backtest.end_date,
        lookback=strategy.metadata.lookback_window,
    )
    engine = BacktestEngine(provider)
    result = engine.run_with_strategy(backtest, strategy)
    export_backtest_result(result, output_dir)
    print(
        "MODEL_RESULT "
        f"total_return={result.total_return:.4f} "
        f"annual_return={result.annual_return:.4f} "
        f"max_drawdown={result.max_drawdown:.4f} "
        f"sharpe={result.sharpe_ratio:.4f} "
        f"trades={len(result.trades)} "
        f"output={output_dir}"
    )


def infer_as_of_date_from_factor_panel(factor_panel_path: str) -> str:
    if not factor_panel_path:
        raise ValueError("as_of_date is required when factor_panel_path is not provided")

    match = re.search(r"asof_(\d{4}-\d{2}-\d{2})\.parquet$", factor_panel_path)
    if match:
        return match.group(1)

    frame = pd.read_parquet(factor_panel_path, columns=["trade_date"])
    if frame.empty:
        raise ValueError("factor panel is empty; cannot infer as_of_date")
    return pd.to_datetime(frame["trade_date"]).max().date().isoformat()


def train_walk_forward_as_of_date_from_config_service(
    config_path: str,
    as_of_date: str = "",
    factor_panel_path: str = "",
    output_scores_path: str = "",
    output_metrics_path: str = "",
) -> dict[str, float | int | str]:
    config = load_research_config(config_path)
    resolved_as_of_date = as_of_date or infer_as_of_date_from_factor_panel(factor_panel_path)
    resolved_factor_panel_path = factor_panel_path or resolve_factor_snapshot_path(
        factor_spec_id=config.factor_spec_id,
        as_of_date=resolved_as_of_date,
        universe_name=config.factor_universe_name,
        start_date=config.factor_start_date,
    )
    resolved_scores_path = output_scores_path or resolve_dated_output_path(config.score_output_path, resolved_as_of_date)
    resolved_metrics_path = output_metrics_path or resolve_dated_output_path(
        config.metric_output_path, resolved_as_of_date
    )
    return train_lightgbm_walk_forward_as_of_date(
        WalkForwardAsOfDateConfig(
            factor_panel_path=resolved_factor_panel_path,
            output_scores_path=resolved_scores_path,
            output_metrics_path=resolved_metrics_path,
            label_column=config.label_column,
            as_of_date=resolved_as_of_date,
            train_window_months=config.train_window_months,
        )
    )


def train_walk_forward_single_date_from_config_service(
    config_path: str,
    test_month: str,
    as_of_date: str,
    factor_panel_path: str = "",
    output_scores_path: str = "",
    output_metrics_path: str = "",
) -> dict[str, float | int | str]:
    config = load_research_config(config_path)
    resolved_factor_panel_path = factor_panel_path or config.factor_snapshot_path
    resolved_scores_path = output_scores_path or resolve_dated_output_path(config.score_output_path, as_of_date)
    resolved_metrics_path = output_metrics_path or resolve_dated_output_path(config.metric_output_path, as_of_date)
    return train_lightgbm_walk_forward_single_date(
        WalkForwardSingleDateConfig(
            factor_panel_path=resolved_factor_panel_path,
            output_scores_path=resolved_scores_path,
            output_metrics_path=resolved_metrics_path,
            label_column=config.label_column,
            test_month=test_month,
            as_of_date=as_of_date,
            train_window_months=config.train_window_months,
        )
    )


def generate_strategy_state_from_config_service(
    config_path: str,
    trade_date: str,
    output_path: str,
    scores_path: str = "",
    initial_cash: float | None = None,
    mode: str = "historical",
    previous_state_path: str = "",
) -> dict[str, object]:
    config = load_research_config(config_path)
    resolved_scores_path = scores_path or config.score_output_path
    resolved_initial_cash = config.initial_cash if initial_cash is None else initial_cash
    return generate_strategy_state(
        StrategyStateConfig(
            scores_path=resolved_scores_path,
            storage_root=config.storage_root,
            output_path=output_path,
            trade_date=trade_date,
            universe_name=config.factor_universe_name,
            mode=mode,
            previous_state_path=previous_state_path,
            top_k=config.top_k,
            rebalance_every=config.rebalance_every,
            lookback_window=config.lookback_window,
            min_hold_bars=config.min_hold_bars,
            keep_buffer=config.keep_buffer,
            min_turnover_names=config.min_turnover_names,
            min_daily_amount=config.min_daily_amount,
            max_close_price=config.max_close_price,
            max_names_per_industry=config.max_names_per_industry,
            max_position_weight=config.max_position_weight,
            exit_policy=config.exit_policy,
            grace_rank_buffer=config.grace_rank_buffer,
            grace_momentum_window=config.grace_momentum_window,
            grace_min_return=config.grace_min_return,
            trailing_stop_window=config.trailing_stop_window,
            trailing_stop_drawdown=config.trailing_stop_drawdown,
            trailing_stop_min_gain=config.trailing_stop_min_gain,
            score_reversal_confirm_days=config.score_reversal_confirm_days,
            score_reversal_threshold=config.score_reversal_threshold,
            hybrid_price_window=config.hybrid_price_window,
            hybrid_price_threshold=config.hybrid_price_threshold,
            strong_keep_extra_buffer=config.strong_keep_extra_buffer,
            strong_keep_momentum_window=config.strong_keep_momentum_window,
            strong_keep_min_return=config.strong_keep_min_return,
            strong_trim_slowdown=config.strong_trim_slowdown,
            strong_trim_momentum_window=config.strong_trim_momentum_window,
            strong_trim_min_return=config.strong_trim_min_return,
            initial_cash=resolved_initial_cash,
            commission_rate=config.commission_rate,
            stamp_tax_rate=config.stamp_tax_rate,
            slippage_rate=config.slippage_rate,
            max_trade_participation_rate=config.max_trade_participation_rate,
            max_pending_days=config.max_pending_days,
        )
    )
