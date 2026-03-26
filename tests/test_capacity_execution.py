from __future__ import annotations

from datetime import date

from ashare_backtest.data import InMemoryDataProvider
from ashare_backtest.engine import BacktestEngine
from ashare_backtest.protocol import (
    AllocationDecision,
    BacktestConfig,
    Bar,
    BaseStrategy,
    Position,
    RebalanceDecision,
    StrategyContext,
    StrategyMetadata,
)


class _StaticAllocationStrategy(BaseStrategy):
    metadata = StrategyMetadata(name="static", description="static allocation", lookback_window=1)

    def __init__(
        self,
        target_weights_by_date: dict[date, dict[str, float]],
        rebalance_dates: set[date] | None = None,
    ) -> None:
        self._target_weights_by_date = target_weights_by_date
        self._rebalance_dates = rebalance_dates

    def rebalance(self, context: StrategyContext) -> RebalanceDecision:
        if self._rebalance_dates is not None and context.trade_date not in self._rebalance_dates:
            return RebalanceDecision(False, "test_skip")
        return RebalanceDecision(True, "test")

    def select(self, context: StrategyContext) -> list[str]:
        weights = self._target_weights_by_date.get(context.trade_date, {})
        return list(weights)

    def allocate(self, context: StrategyContext, selected_symbols: list[str]) -> AllocationDecision:
        return AllocationDecision(target_weights=self._target_weights_by_date.get(context.trade_date, {}), note="test")


def test_buy_order_is_partially_filled_when_participation_cap_is_hit() -> None:
    trade_date = date(2025, 1, 6)
    bars = {
        "AAA": [
            Bar(
                symbol="AAA",
                trade_date=trade_date,
                open=10.0,
                high=10.0,
                low=10.0,
                close=10.0,
                amount=3_000.0,
            )
        ]
    }
    engine = BacktestEngine(InMemoryDataProvider(bars))
    strategy = _StaticAllocationStrategy({trade_date: {"AAA": 1.0}})

    result = engine.run_with_strategy(
        BacktestConfig(
            strategy_path="__test__",
            start_date=trade_date,
            end_date=trade_date,
            universe=("AAA",),
            initial_cash=10_000.0,
            max_trade_participation_rate=0.5,
        ),
        strategy,
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.status == "filled"
    assert trade.quantity == 100
    assert trade.reason == "rebalance_entry_or_add_capacity_capped"


def test_exit_order_is_partially_filled_when_participation_cap_is_hit() -> None:
    day_one = date(2025, 1, 6)
    day_two = date(2025, 1, 7)
    bars = {
        "AAA": [
            Bar(symbol="AAA", trade_date=day_one, open=10.0, high=10.0, low=10.0, close=10.0, amount=100_000.0),
            Bar(symbol="AAA", trade_date=day_two, open=10.0, high=10.0, low=10.0, close=10.0, amount=3_000.0),
        ]
    }
    engine = BacktestEngine(InMemoryDataProvider(bars))
    strategy = _StaticAllocationStrategy(
        {
            day_one: {"AAA": 1.0},
            day_two: {},
        }
    )

    result = engine.run_with_strategy(
        BacktestConfig(
            strategy_path="__test__",
            start_date=day_one,
            end_date=day_two,
            universe=("AAA",),
            initial_cash=10_000.0,
            max_trade_participation_rate=0.5,
        ),
        strategy,
    )

    assert len(result.trades) == 2
    exit_trade = result.trades[1]
    assert exit_trade.side == "SELL"
    assert exit_trade.status == "filled"
    assert exit_trade.quantity == 100
    assert exit_trade.reason == "rebalance_exit_capacity_capped"


def test_pending_buy_order_continues_on_next_day() -> None:
    day_one = date(2025, 1, 6)
    day_two = date(2025, 1, 7)
    bars = {
        "AAA": [
            Bar(symbol="AAA", trade_date=day_one, open=10.0, high=10.0, low=10.0, close=10.0, amount=3_000.0),
            Bar(symbol="AAA", trade_date=day_two, open=10.0, high=10.0, low=10.0, close=10.0, amount=20_000.0),
        ]
    }
    engine = BacktestEngine(InMemoryDataProvider(bars))
    strategy = _StaticAllocationStrategy(
        {
            day_one: {"AAA": 1.0},
            day_two: {"AAA": 1.0},
        }
    )

    result = engine.run_with_strategy(
        BacktestConfig(
            strategy_path="__test__",
            start_date=day_one,
            end_date=day_two,
            universe=("AAA",),
            initial_cash=10_000.0,
            max_trade_participation_rate=0.5,
        ),
        strategy,
    )

    assert len(result.trades) >= 2
    assert result.trades[0].quantity == 100
    continued_trade = result.trades[1]
    assert continued_trade.side == "BUY"
    assert continued_trade.status == "filled"
    assert continued_trade.quantity >= 100
    assert continued_trade.reason.startswith("pending_rebalance_continued")


def test_pending_order_expires_after_max_pending_days() -> None:
    day_one = date(2025, 1, 6)
    day_two = date(2025, 1, 7)
    day_three = date(2025, 1, 8)
    bars = {
        "AAA": [
            Bar(symbol="AAA", trade_date=day_one, open=10.0, high=10.0, low=10.0, close=10.0, amount=3_000.0),
            Bar(symbol="AAA", trade_date=day_two, open=10.0, high=10.0, low=10.0, close=10.0, amount=0.0),
            Bar(symbol="AAA", trade_date=day_three, open=10.0, high=10.0, low=10.0, close=10.0, amount=0.0),
        ]
    }
    engine = BacktestEngine(InMemoryDataProvider(bars))
    strategy = _StaticAllocationStrategy({day_one: {"AAA": 1.0}}, rebalance_dates={day_one})

    result = engine.run_with_strategy(
        BacktestConfig(
            strategy_path="__test__",
            start_date=day_one,
            end_date=day_three,
            universe=("AAA",),
            initial_cash=10_000.0,
            max_trade_participation_rate=0.5,
            max_pending_days=1,
        ),
        strategy,
    )

    assert any(trade.reason == "pending_expired" for trade in result.trades)
