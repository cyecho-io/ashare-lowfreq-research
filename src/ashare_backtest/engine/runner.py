from __future__ import annotations

from datetime import date

from ashare_backtest.data import DataProvider
from ashare_backtest.engine.loader import load_strategy
from ashare_backtest.protocol import (
    AllocationDecision,
    Bar,
    BaseStrategy,
    BacktestConfig,
    BacktestResult,
    Position,
    StrategyContext,
    Trade,
)
from ashare_backtest.reporting.metrics import calculate_sharpe


class BacktestEngine:
    """MVP runner skeleton.

    The engine owns scheduling, data access, execution simulation and accounting.
    The current version provides the call chain and extension points, but not the
    full production-grade fill model yet.
    """

    def __init__(self, data_provider: DataProvider) -> None:
        self.data_provider = data_provider

    def run(self, config: BacktestConfig) -> BacktestResult:
        strategy = load_strategy(config.strategy_path)
        return self.run_with_strategy(config, strategy)

    def run_with_strategy(self, config: BacktestConfig, strategy: BaseStrategy) -> BacktestResult:
        trade_dates = self.data_provider.get_trade_dates(config.start_date, config.end_date)
        cash = config.initial_cash
        positions: dict[str, Position] = {}
        trades: list[Trade] = []
        equity_curve: list[tuple[date, float]] = []
        realized_pnls: list[float] = []

        for index, trade_date in enumerate(trade_dates):
            previous_trade_date = trade_dates[index - 1] if index > 0 else None
            bars = self.data_provider.get_history(
                symbols=config.universe,
                end_date=previous_trade_date or trade_date,
                lookback=strategy.metadata.lookback_window,
            )
            current_bars = self.data_provider.get_bars_on_date(config.universe, trade_date)
            positions = self._refresh_positions(positions, current_bars)
            context = StrategyContext(
                trade_date=trade_date,
                universe=config.universe,
                bars=bars,
                positions=positions,
                cash=cash,
            )
            decision = strategy.rebalance(context)
            if decision.should_rebalance:
                selected = strategy.select(context)
                allocation = strategy.allocate(context, selected)
                cash, positions, fill_trades, fill_pnls = self._execute_rebalance(
                    trade_date=trade_date,
                    current_bars=current_bars,
                    cash=cash,
                    positions=positions,
                    allocation=allocation,
                    config=config,
                )
                trades.extend(fill_trades)
                realized_pnls.extend(fill_pnls)
            portfolio_value = cash + self._mark_to_market(positions)
            equity_curve.append((trade_date, portfolio_value))

        total_return = 0.0
        if equity_curve and config.initial_cash > 0:
            total_return = equity_curve[-1][1] / config.initial_cash - 1
        daily_returns = self._daily_returns(equity_curve)
        filled_trade_count = sum(1 for trade in trades if trade.status == "filled")
        rejected_trade_count = sum(1 for trade in trades if trade.status != "filled")
        return BacktestResult(
            total_return=total_return,
            annual_return=self._annual_return(equity_curve, config.initial_cash),
            max_drawdown=self._max_drawdown(equity_curve),
            sharpe_ratio=calculate_sharpe(daily_returns),
            win_rate=self._win_rate(realized_pnls),
            profit_loss_ratio=self._profit_loss_ratio(realized_pnls),
            turnover_ratio=self._turnover_ratio(trades, config.initial_cash),
            filled_trade_count=filled_trade_count,
            rejected_trade_count=rejected_trade_count,
            trades=trades,
            equity_curve=equity_curve,
        )

    @staticmethod
    def _mark_to_market(positions: dict[str, Position]) -> float:
        return sum(position.quantity * position.last_price for position in positions.values())

    @staticmethod
    def _refresh_positions(
        positions: dict[str, Position],
        current_bars: dict[str, Bar],
    ) -> dict[str, Position]:
        refreshed: dict[str, Position] = {}
        for symbol, position in positions.items():
            bar = current_bars.get(symbol)
            last_price = bar.close if bar is not None else position.last_price
            refreshed[symbol] = Position(
                symbol=symbol,
                quantity=position.quantity,
                cost_basis=position.cost_basis,
                last_price=last_price,
            )
        return refreshed

    def _execute_rebalance(
        self,
        trade_date: date,
        current_bars: dict[str, Bar],
        cash: float,
        positions: dict[str, Position],
        allocation: AllocationDecision,
        config: BacktestConfig,
    ) -> tuple[float, dict[str, Position], list[Trade], list[float]]:
        trades: list[Trade] = []
        realized_pnls: list[float] = []
        working_positions = dict(positions)
        portfolio_value = cash + self._mark_to_market(positions)
        target_weights = self._normalize_weights(allocation.target_weights)
        target_symbols = set(target_weights)

        for symbol in sorted(set(working_positions) - target_symbols):
            bar = current_bars.get(symbol)
            position = working_positions.get(symbol)
            if bar is None or position is None:
                continue
            if bar.paused or bar.limit_down:
                trades.append(
                    Trade(
                        trade_date=trade_date,
                        symbol=symbol,
                        side="SELL",
                        quantity=position.quantity,
                        price=bar.open if bar is not None else 0.0,
                        amount=0.0,
                        commission=0.0,
                        tax=0.0,
                        slippage=0.0,
                        status="rejected",
                        reason="paused_or_limit_down",
                    )
                )
                continue
            fill_price = self._execution_price(bar.open, config.slippage_rate, side="SELL")
            amount = position.quantity * fill_price
            commission = amount * config.commission_rate
            tax = amount * config.stamp_tax_rate
            slippage = position.quantity * abs(fill_price - bar.open)
            cash += amount - commission - tax
            realized_pnls.append((fill_price - position.cost_basis) * position.quantity - commission - tax)
            trades.append(
                Trade(
                    trade_date=trade_date,
                    symbol=symbol,
                    side="SELL",
                    quantity=position.quantity,
                    price=fill_price,
                    amount=amount,
                    commission=commission,
                    tax=tax,
                    slippage=slippage,
                    status="filled",
                    reason="rebalance_exit",
                )
            )
            del working_positions[symbol]

        for symbol, target_weight in sorted(target_weights.items()):
            bar = current_bars.get(symbol)
            if bar is None:
                continue
            target_value = portfolio_value * target_weight
            current_quantity = working_positions.get(symbol, Position(symbol, 0, 0.0, bar.close)).quantity
            current_value = current_quantity * bar.open
            delta_value = target_value - current_value
            side = "BUY" if delta_value > 0 else "SELL"
            if abs(delta_value) < bar.open * 100:
                continue
            if side == "BUY" and (bar.paused or bar.limit_up):
                trades.append(
                    Trade(
                        trade_date=trade_date,
                        symbol=symbol,
                        side="BUY",
                        quantity=0,
                        price=bar.open,
                        amount=0.0,
                        commission=0.0,
                        tax=0.0,
                        slippage=0.0,
                        status="rejected",
                        reason="paused_or_limit_up",
                    )
                )
                continue
            if side == "SELL" and (bar.paused or bar.limit_down):
                trades.append(
                    Trade(
                        trade_date=trade_date,
                        symbol=symbol,
                        side="SELL",
                        quantity=0,
                        price=bar.open,
                        amount=0.0,
                        commission=0.0,
                        tax=0.0,
                        slippage=0.0,
                        status="rejected",
                        reason="paused_or_limit_down",
                    )
                )
                continue

            raw_quantity = int(abs(delta_value) / bar.open)
            quantity = (raw_quantity // 100) * 100
            if quantity <= 0:
                continue
            if side == "SELL":
                quantity = min(quantity, current_quantity)
            fill_price = self._execution_price(bar.open, config.slippage_rate, side=side)
            amount = quantity * fill_price
            commission = amount * config.commission_rate
            tax = amount * config.stamp_tax_rate if side == "SELL" else 0.0
            slippage = quantity * abs(fill_price - bar.open)
            if side == "BUY":
                total_cost = amount + commission
                affordable = (cash // (fill_price * 100)) * 100
                quantity = min(quantity, int(affordable))
                if quantity <= 0:
                    continue
                amount = quantity * fill_price
                commission = amount * config.commission_rate
                total_cost = amount + commission
                cash -= total_cost
                previous = working_positions.get(symbol)
                if previous is None:
                    working_positions[symbol] = Position(
                        symbol=symbol,
                        quantity=quantity,
                        cost_basis=fill_price,
                        last_price=bar.close,
                    )
                else:
                    total_quantity = previous.quantity + quantity
                    blended_cost = ((previous.quantity * previous.cost_basis) + amount) / total_quantity
                    working_positions[symbol] = Position(
                        symbol=symbol,
                        quantity=total_quantity,
                        cost_basis=blended_cost,
                        last_price=bar.close,
                    )
                trades.append(
                    Trade(
                        trade_date=trade_date,
                        symbol=symbol,
                        side="BUY",
                        quantity=quantity,
                        price=fill_price,
                        amount=amount,
                        commission=commission,
                        tax=0.0,
                        slippage=slippage,
                        status="filled",
                        reason="rebalance_entry_or_add",
                    )
                )
            else:
                if quantity <= 0:
                    continue
                previous = working_positions.get(symbol)
                if previous is None:
                    continue
                cash += amount - commission - tax
                realized_pnls.append((fill_price - previous.cost_basis) * quantity - commission - tax)
                remaining = previous.quantity - quantity
                if remaining > 0:
                    working_positions[symbol] = Position(
                        symbol=symbol,
                        quantity=remaining,
                        cost_basis=previous.cost_basis,
                        last_price=bar.close,
                    )
                else:
                    del working_positions[symbol]
                trades.append(
                    Trade(
                        trade_date=trade_date,
                        symbol=symbol,
                        side="SELL",
                        quantity=quantity,
                        price=fill_price,
                        amount=amount,
                        commission=commission,
                        tax=tax,
                        slippage=slippage,
                        status="filled",
                        reason="rebalance_trim_or_exit",
                    )
                )

        return cash, working_positions, trades, realized_pnls

    @staticmethod
    def _normalize_weights(target_weights: dict[str, float]) -> dict[str, float]:
        positive = {symbol: max(weight, 0.0) for symbol, weight in target_weights.items() if weight > 0}
        total = sum(positive.values())
        if total <= 0:
            return {}
        return {symbol: weight / total for symbol, weight in positive.items()}

    @staticmethod
    def _execution_price(open_price: float, slippage_rate: float, side: str) -> float:
        if side == "BUY":
            return open_price * (1 + slippage_rate)
        return open_price * (1 - slippage_rate)

    @staticmethod
    def _daily_returns(equity_curve: list[tuple[date, float]]) -> list[float]:
        returns: list[float] = []
        for index in range(1, len(equity_curve)):
            previous = equity_curve[index - 1][1]
            current = equity_curve[index][1]
            if previous > 0:
                returns.append(current / previous - 1)
        return returns

    @staticmethod
    def _annual_return(equity_curve: list[tuple[date, float]], initial_cash: float) -> float:
        if not equity_curve or initial_cash <= 0:
            return 0.0
        periods = len(equity_curve)
        if periods < 2:
            return 0.0
        ending = equity_curve[-1][1]
        return (ending / initial_cash) ** (252 / periods) - 1

    @staticmethod
    def _max_drawdown(equity_curve: list[tuple[date, float]]) -> float:
        peak = 0.0
        max_drawdown = 0.0
        for _, equity in equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                max_drawdown = min(max_drawdown, equity / peak - 1)
        return abs(max_drawdown)

    @staticmethod
    def _win_rate(realized_pnls: list[float]) -> float:
        if not realized_pnls:
            return 0.0
        wins = sum(1 for item in realized_pnls if item > 0)
        return wins / len(realized_pnls)

    @staticmethod
    def _profit_loss_ratio(realized_pnls: list[float]) -> float:
        profits = [item for item in realized_pnls if item > 0]
        losses = [abs(item) for item in realized_pnls if item < 0]
        if not profits or not losses:
            return 0.0
        return (sum(profits) / len(profits)) / (sum(losses) / len(losses))

    @staticmethod
    def _turnover_ratio(trades: list[Trade], initial_cash: float) -> float:
        if initial_cash <= 0:
            return 0.0
        traded_amount = sum(trade.amount for trade in trades if trade.status == "filled")
        return traded_amount / initial_cash
