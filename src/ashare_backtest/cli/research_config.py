from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResearchRunConfig:
    storage_root: str
    factor_output_path: str
    factor_start_date: str
    factor_end_date: str
    label_column: str
    train_window_months: int
    test_start_month: str
    test_end_month: str
    score_output_path: str
    metric_output_path: str
    layer_output_path: str
    model_backtest_output_dir: str
    backtest_start_date: str
    backtest_end_date: str
    top_k: int
    rebalance_every: int
    lookback_window: int
    min_hold_bars: int
    keep_buffer: int
    min_turnover_names: int
    min_daily_amount: float
    max_names_per_industry: int
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


def load_research_config(path: str | Path) -> ResearchRunConfig:
    payload = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    storage = payload["storage"]
    factors = payload["factors"]
    training = payload["training"]
    analysis = payload["analysis"]
    backtest = payload["model_backtest"]
    return ResearchRunConfig(
        storage_root=str(storage.get("root", "storage")),
        factor_output_path=str(factors["output_path"]),
        factor_start_date=str(factors["start_date"]),
        factor_end_date=str(factors["end_date"]),
        label_column=str(training.get("label_column", "excess_fwd_return_5")),
        train_window_months=int(training.get("train_window_months", 12)),
        test_start_month=str(training["test_start_month"]),
        test_end_month=str(training["test_end_month"]),
        score_output_path=str(training["score_output_path"]),
        metric_output_path=str(training["metric_output_path"]),
        layer_output_path=str(analysis["layer_output_path"]),
        model_backtest_output_dir=str(backtest["output_dir"]),
        backtest_start_date=str(backtest["start_date"]),
        backtest_end_date=str(backtest["end_date"]),
        top_k=int(backtest.get("top_k", 5)),
        rebalance_every=int(backtest.get("rebalance_every", 3)),
        lookback_window=int(backtest.get("lookback_window", 20)),
        min_hold_bars=int(backtest.get("min_hold_bars", 5)),
        keep_buffer=int(backtest.get("keep_buffer", 2)),
        min_turnover_names=int(backtest.get("min_turnover_names", 3)),
        min_daily_amount=float(backtest.get("min_daily_amount", 0.0)),
        max_names_per_industry=int(backtest.get("max_names_per_industry", 0)),
        exit_policy=str(backtest.get("exit_policy", "buffered_rank")),
        grace_rank_buffer=int(backtest.get("grace_rank_buffer", 0)),
        grace_momentum_window=int(backtest.get("grace_momentum_window", 3)),
        grace_min_return=float(backtest.get("grace_min_return", 0.0)),
        trailing_stop_window=int(backtest.get("trailing_stop_window", 10)),
        trailing_stop_drawdown=float(backtest.get("trailing_stop_drawdown", 0.12)),
        trailing_stop_min_gain=float(backtest.get("trailing_stop_min_gain", 0.15)),
        score_reversal_confirm_days=int(backtest.get("score_reversal_confirm_days", 3)),
        score_reversal_threshold=float(backtest.get("score_reversal_threshold", 0.0)),
        hybrid_price_window=int(backtest.get("hybrid_price_window", 5)),
        hybrid_price_threshold=float(backtest.get("hybrid_price_threshold", 0.0)),
        strong_keep_extra_buffer=int(backtest.get("strong_keep_extra_buffer", 0)),
        strong_keep_momentum_window=int(backtest.get("strong_keep_momentum_window", 5)),
        strong_keep_min_return=float(backtest.get("strong_keep_min_return", 0.0)),
        strong_trim_slowdown=float(backtest.get("strong_trim_slowdown", 0.0)),
        strong_trim_momentum_window=int(backtest.get("strong_trim_momentum_window", 5)),
        strong_trim_min_return=float(backtest.get("strong_trim_min_return", 0.0)),
        initial_cash=float(backtest.get("initial_cash", 1_000_000.0)),
        commission_rate=float(backtest.get("commission_rate", 0.0003)),
        stamp_tax_rate=float(backtest.get("stamp_tax_rate", 0.001)),
        slippage_rate=float(backtest.get("slippage_rate", 0.0005)),
    )
