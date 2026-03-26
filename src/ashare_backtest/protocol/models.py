from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class StrategyMetadata:
    name: str
    description: str
    version: str = "0.1.0"
    author: str = "local"
    lookback_window: int = 60


@dataclass(frozen=True)
class Bar:
    symbol: str
    trade_date: date
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    amount: float = 0.0
    paused: bool = False
    limit_up: bool = False
    limit_down: bool = False


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: int
    cost_basis: float
    last_price: float


@dataclass(frozen=True)
class StrategyContext:
    trade_date: date
    universe: tuple[str, ...]
    bars: dict[str, list[Bar]]
    positions: dict[str, Position]
    cash: float

    def history(self, symbol: str) -> list[Bar]:
        return self.bars.get(symbol, [])


@dataclass(frozen=True)
class RebalanceDecision:
    should_rebalance: bool
    reason: str = ""


@dataclass(frozen=True)
class AllocationDecision:
    target_weights: dict[str, float]
    note: str = ""


@dataclass(frozen=True)
class Trade:
    trade_date: date
    symbol: str
    side: str
    quantity: int
    price: float
    amount: float
    commission: float
    tax: float
    slippage: float
    status: str
    reason: str = ""


@dataclass(frozen=True)
class BacktestConfig:
    strategy_path: str
    start_date: date
    end_date: date
    universe: tuple[str, ...]
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage_rate: float = 0.0005
    rebalance_price: str = "open"
    max_trade_participation_rate: float = 0.0
    max_pending_days: int = 0


@dataclass
class BacktestResult:
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    profit_loss_ratio: float
    turnover_ratio: float = 0.0
    filled_trade_count: int = 0
    rejected_trade_count: int = 0
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[date, float]] = field(default_factory=list)
