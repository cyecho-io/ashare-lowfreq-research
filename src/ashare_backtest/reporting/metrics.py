from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from ashare_backtest.protocol import BacktestResult


@dataclass(frozen=True)
class MetricsSnapshot:
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_loss_ratio: float


def summarize_result(result: BacktestResult) -> MetricsSnapshot:
    return MetricsSnapshot(
        total_return=result.total_return,
        annual_return=result.annual_return,
        max_drawdown=result.max_drawdown,
        sharpe_ratio=result.sharpe_ratio,
        win_rate=result.win_rate,
        profit_loss_ratio=result.profit_loss_ratio,
    )


def calculate_sharpe(daily_returns: list[float], risk_free_rate: float = 0.0) -> float:
    if len(daily_returns) < 2:
        return 0.0
    mean_return = sum(daily_returns) / len(daily_returns)
    variance = sum((item - mean_return) ** 2 for item in daily_returns) / (len(daily_returns) - 1)
    if variance <= 0:
        return 0.0
    return ((mean_return - risk_free_rate / 252) / (variance ** 0.5)) * sqrt(252)
