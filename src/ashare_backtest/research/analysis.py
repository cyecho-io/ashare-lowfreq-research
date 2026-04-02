from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ashare_backtest.data import DataProvider, ParquetDataProvider
from ashare_backtest.engine import BacktestEngine
from ashare_backtest.logging_utils import get_logger
from ashare_backtest.protocol import AllocationDecision, Position, StrategyContext, Trade
from ashare_backtest.research.score_workflow import (
    build_preloaded_score_provider,
    build_score_backtest_config,
    build_score_strategy,
    resolve_score_universe,
)

LOGGER = get_logger("research.analysis")


@dataclass(frozen=True)
class LayeredAnalysisConfig:
    scores_path: str
    output_path: str
    bins: int = 5


@dataclass(frozen=True)
class CapacityAnalysisConfig:
    trades_path: str
    storage_root: str
    output_path: str
    base_capital: float = 1_000_000.0
    scale_capitals: tuple[float, ...] = (100_000.0, 300_000.0, 500_000.0, 1_000_000.0, 3_000_000.0, 5_000_000.0)
    participation_thresholds: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
    top_trade_count: int = 20


@dataclass(frozen=True)
class MonthlyComparisonConfig:
    result_dirs: tuple[str, ...]
    labels: tuple[str, ...]
    output_path: str


@dataclass(frozen=True)
class RiskExposureConfig:
    result_dir: str
    storage_root: str
    output_path: str
    top_industries: int = 5
    volatility_window: int = 20


@dataclass(frozen=True)
class StartDateRobustnessConfig:
    scores_path: str
    storage_root: str
    output_path: str
    analysis_start_date: str
    analysis_end_date: str
    holding_months: int = 8
    cadence: str = "monthly"
    universe_name: str = ""
    top_k: int = 5
    rebalance_every: int = 3
    lookback_window: int = 20
    min_hold_bars: int = 5
    keep_buffer: int = 2
    min_turnover_names: int = 2
    min_daily_amount: float = 0.0
    max_close_price: float = 0.0
    max_names_per_industry: int = 0
    max_position_weight: float = 0.0
    exit_policy: str = "buffered_rank"
    grace_rank_buffer: int = 0
    grace_momentum_window: int = 3
    grace_min_return: float = 0.0
    trailing_stop_window: int = 10
    trailing_stop_drawdown: float = 0.12
    trailing_stop_min_gain: float = 0.15
    score_reversal_confirm_days: int = 3
    score_reversal_threshold: float = 0.0
    hybrid_price_window: int = 5
    hybrid_price_threshold: float = 0.0
    strong_keep_extra_buffer: int = 0
    strong_keep_momentum_window: int = 5
    strong_keep_min_return: float = 0.0
    strong_trim_slowdown: float = 0.0
    strong_trim_momentum_window: int = 5
    strong_trim_min_return: float = 0.0
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_trade_participation_rate: float = 0.0
    max_pending_days: int = 0


@dataclass(frozen=True)
class PremarketReferenceConfig:
    scores_path: str
    storage_root: str
    output_path: str
    trade_date: str
    universe_name: str = ""
    top_k: int = 5
    rebalance_every: int = 3
    lookback_window: int = 20
    min_hold_bars: int = 5
    keep_buffer: int = 2
    min_turnover_names: int = 2
    min_daily_amount: float = 0.0
    max_close_price: float = 0.0
    max_names_per_industry: int = 0
    max_position_weight: float = 0.0
    exit_policy: str = "buffered_rank"
    grace_rank_buffer: int = 0
    grace_momentum_window: int = 3
    grace_min_return: float = 0.0
    trailing_stop_window: int = 10
    trailing_stop_drawdown: float = 0.12
    trailing_stop_min_gain: float = 0.15
    score_reversal_confirm_days: int = 3
    score_reversal_threshold: float = 0.0
    hybrid_price_window: int = 5
    hybrid_price_threshold: float = 0.0
    strong_keep_extra_buffer: int = 0
    strong_keep_momentum_window: int = 5
    strong_keep_min_return: float = 0.0
    strong_trim_slowdown: float = 0.0
    strong_trim_momentum_window: int = 5
    strong_trim_min_return: float = 0.0
    initial_cash: float = 1_000_000.0
    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_trade_participation_rate: float = 0.0
    max_pending_days: int = 0


@dataclass(frozen=True)
class StrategyStateConfig(PremarketReferenceConfig):
    mode: str = "continue"
    previous_state_path: str = ""
    simulate_trade_execution: bool = True


def _serialize_positions(positions: dict[str, Position], portfolio_value: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for symbol in sorted(positions):
        position = positions[symbol]
        market_value = position.quantity * position.last_price
        weight = market_value / portfolio_value if portfolio_value > 0 else 0.0
        rows.append(
            {
                "symbol": symbol,
                "quantity": int(position.quantity),
                "cost_basis": round(float(position.cost_basis), 6),
                "last_price": round(float(position.last_price), 6),
                "market_value": round(float(market_value), 2),
                "weight": round(float(weight), 6),
            }
        )
    return rows


def _serialize_pending_orders(
    pending_orders: dict[tuple[str, str], BacktestEngine._PendingOrder],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in sorted(pending_orders):
        order = pending_orders[key]
        rows.append(
            {
                "symbol": order.symbol,
                "side": order.side,
                "quantity": int(order.quantity),
                "reason": order.reason,
                "age_days": int(order.age_days),
            }
        )
    return rows


def _annotate_action_execution(
    *,
    actions: list[dict[str, object]],
    trade_log: list[Trade],
    trade_date: date,
    portfolio_value: float,
    should_rebalance: bool,
) -> None:
    trade_index: dict[tuple[str, str], list[Trade]] = {}
    for trade in trade_log:
        if trade.trade_date != trade_date:
            continue
        trade_index.setdefault((trade.symbol, trade.side), []).append(trade)

    for action in actions:
        action_name = str(action.get("action") or "").upper()
        current_quantity = int(action.get("current_quantity") or 0)
        target_weight = float(action.get("target_weight") or 0.0)
        next_open = float(action.get("next_open") or 0.0)

        if not should_rebalance:
            action["planned_quantity"] = 0
            action["executed_quantity"] = 0
            action["would_trade"] = False
            action["execution_status"] = "not_rebalanced"
            action["execution_reason"] = "rebalance_not_triggered"
            continue

        if action_name not in {"BUY", "ADD", "SELL", "TRIM"}:
            action["planned_quantity"] = 0
            action["executed_quantity"] = 0
            action["would_trade"] = False
            action["execution_status"] = "already_at_target"
            action["execution_reason"] = "no_order_needed"
            continue

        side = "BUY" if action_name in {"BUY", "ADD"} else "SELL"
        planned_quantity = 0
        if next_open > 0:
            target_value = portfolio_value * target_weight
            current_value = current_quantity * next_open
            delta_value = target_value - current_value
            if abs(delta_value) >= next_open * 100:
                raw_quantity = int(abs(delta_value) / next_open)
                planned_quantity = (raw_quantity // 100) * 100
                if side == "SELL":
                    planned_quantity = min(planned_quantity, current_quantity)

        related_trades = trade_index.get((str(action.get("symbol") or ""), side), [])
        executed_quantity = sum(int(trade.quantity) for trade in related_trades if trade.status == "filled")
        rejected_reasons = [str(trade.reason) for trade in related_trades if trade.status != "filled"]

        action["planned_quantity"] = planned_quantity
        action["executed_quantity"] = executed_quantity
        action["would_trade"] = executed_quantity > 0

        if executed_quantity > 0 and planned_quantity > executed_quantity:
            action["execution_status"] = "partially_executed"
            action["execution_reason"] = "partially_filled"
        elif executed_quantity > 0:
            action["execution_status"] = "executed"
            action["execution_reason"] = "filled"
        elif planned_quantity <= 0:
            action["execution_status"] = "below_round_lot"
            action["execution_reason"] = "below_100_share_lot"
        elif rejected_reasons:
            action["execution_status"] = "rejected"
            action["execution_reason"] = ",".join(sorted(set(rejected_reasons)))
        else:
            action["execution_status"] = "not_executed"
            action["execution_reason"] = "no_fill"


def _write_trade_log(path: Path, trades: list[Trade]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trade_date",
                "symbol",
                "side",
                "quantity",
                "price",
                "amount",
                "commission",
                "tax",
                "slippage",
                "status",
                "reason",
            ],
        )
        writer.writeheader()
        for trade in trades:
            writer.writerow(
                {
                    "trade_date": trade.trade_date.isoformat(),
                    "symbol": trade.symbol,
                    "side": trade.side,
                    "quantity": int(trade.quantity),
                    "price": float(trade.price),
                    "amount": float(trade.amount),
                    "commission": float(trade.commission),
                    "tax": float(trade.tax),
                    "slippage": float(trade.slippage),
                    "status": trade.status,
                    "reason": trade.reason,
                }
            )


def _write_decision_log(path: Path, decisions: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "trade_date",
                "signal_date",
                "decision_reason",
                "should_rebalance",
                "selected_symbols",
                "current_position_count",
                "target_position_count",
                "cash_pre_decision",
            ],
        )
        writer.writeheader()
        for item in decisions:
            writer.writerow(item)


def _load_strategy_state(path: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    next_state = payload.get("next_state")
    if not isinstance(next_state, dict):
        raise ValueError(f"strategy state missing next_state payload: {path}")
    as_of_trade_date_text = str(next_state.get("as_of_trade_date", "")).strip()
    if not as_of_trade_date_text:
        raise ValueError(f"strategy state missing next_state.as_of_trade_date: {path}")

    positions: dict[str, Position] = {}
    for item in next_state.get("positions", []):
        symbol = str(item.get("symbol", "")).strip()
        if not symbol:
            continue
        positions[symbol] = Position(
            symbol=symbol,
            quantity=int(item.get("quantity", 0)),
            cost_basis=float(item.get("cost_basis", 0.0)),
            last_price=float(item.get("last_price", 0.0)),
        )

    pending_orders: dict[tuple[str, str], BacktestEngine._PendingOrder] = {}
    for item in next_state.get("pending_orders", []):
        symbol = str(item.get("symbol", "")).strip()
        side = str(item.get("side", "")).strip().upper()
        if not symbol or not side:
            continue
        pending_orders[(symbol, side)] = BacktestEngine._PendingOrder(
            symbol=symbol,
            side=side,
            quantity=int(item.get("quantity", 0)),
            reason=str(item.get("reason", "")),
            age_days=int(item.get("age_days", 0)),
        )

    hold_days = {
        str(symbol): int(days)
        for symbol, days in dict(next_state.get("hold_days", {})).items()
        if str(symbol).strip()
    }
    raw_last_rebalance = next_state.get("last_rebalance_date")
    last_rebalance_text = "" if raw_last_rebalance is None else str(raw_last_rebalance).strip()
    if last_rebalance_text.lower() in {"", "none", "null"}:
        last_rebalance_date = None
    else:
        last_rebalance_date = date.fromisoformat(last_rebalance_text)
    planned_target_weights = {
        str(symbol): float(weight)
        for symbol, weight in dict(next_state.get("planned_target_weights", {})).items()
        if str(symbol).strip()
    }
    return {
        "as_of_trade_date": date.fromisoformat(as_of_trade_date_text),
        "cash": float(next_state.get("cash", 0.0)),
        "positions": positions,
        "pending_orders": pending_orders,
        "hold_days": hold_days,
        "last_rebalance_date": last_rebalance_date,
        "execution_pending": bool(next_state.get("execution_pending", False)),
        "planned_target_weights": planned_target_weights,
    }


def generate_premarket_reference(
    config: PremarketReferenceConfig,
    provider: DataProvider | None = None,
) -> dict[str, object]:
    target_trade_date = date.fromisoformat(config.trade_date)
    scores = pd.read_parquet(config.scores_path).sort_values(["trade_date", "symbol"])
    if scores.empty:
        raise ValueError("scores parquet is empty")

    scores["trade_date"] = pd.to_datetime(scores["trade_date"])
    provider = provider or ParquetDataProvider(config.storage_root)
    strategy = build_score_strategy(config)

    earliest_score_date = scores["trade_date"].min().date()
    trade_dates = provider.get_trade_dates(earliest_score_date, target_trade_date)
    if target_trade_date not in trade_dates:
        raise ValueError(f"trade date is not an open trading day: {config.trade_date}")
    target_index = trade_dates.index(target_trade_date)
    if target_index == 0:
        raise ValueError("cannot build premarket reference for the first available trade date")
    signal_date = trade_dates[target_index - 1]
    universe = resolve_score_universe(config, scores, as_of_date=signal_date)
    provider = build_preloaded_score_provider(
        storage_root=config.storage_root,
        universe=universe,
        start_date=earliest_score_date,
        end_date=target_trade_date,
        lookback=strategy.metadata.lookback_window,
        provider=provider,
    )

    engine = BacktestEngine(provider)
    cash = config.initial_cash
    positions: dict[str, Position] = {}
    pending_orders: dict[tuple[str, str], BacktestEngine._PendingOrder] = {}
    last_allocation = AllocationDecision(target_weights={}, note="no_prior_allocation")
    last_selected: list[str] = []
    last_decision_reason = "not_rebalanced"
    model_positions: dict[str, Position] = {}
    model_cash = cash
    model_context: StrategyContext | None = None
    should_rebalance_on_target = False

    for index, trade_date_item in enumerate(trade_dates[: target_index + 1]):
        previous_trade_date = trade_dates[index - 1] if index > 0 else None
        bars = provider.get_history(
            symbols=universe,
            end_date=previous_trade_date or trade_date_item,
            lookback=strategy.metadata.lookback_window,
        )
        current_bars = provider.get_bars_on_date(universe, trade_date_item)
        positions = engine._refresh_positions(positions, current_bars)
        cash, positions, _, _, pending_orders = engine._execute_pending_orders(
            trade_date=trade_date_item,
            current_bars=current_bars,
            cash=cash,
            positions=positions,
            pending_orders=pending_orders,
            config=build_score_backtest_config(config, universe, trade_date_item, trade_date_item),
        )
        context = StrategyContext(
            trade_date=trade_date_item,
            universe=universe,
            bars=bars,
            positions=positions,
            cash=cash,
        )
        decision = strategy.rebalance(context)
        if trade_date_item == target_trade_date:
            should_rebalance_on_target = decision.should_rebalance
            if decision.should_rebalance:
                last_selected = strategy.select(context)
                last_allocation = strategy.allocate(context, last_selected)
            else:
                last_selected = sorted(positions)
            last_decision_reason = decision.reason
            model_positions = positions
            model_cash = cash
            model_context = context
            break

        if decision.should_rebalance:
            selected = strategy.select(context)
            allocation = strategy.allocate(context, selected)
            cash, positions, _, _ = engine._execute_rebalance(
                trade_date=trade_date_item,
                current_bars=current_bars,
                cash=cash,
                positions=positions,
                allocation=allocation,
                config=build_score_backtest_config(config, universe, trade_date_item, trade_date_item),
            )
            pending_orders = engine._build_pending_orders(
                cash=cash,
                positions=positions,
                target_weights=allocation.target_weights,
                current_bars=current_bars,
            )

    portfolio_value = model_cash + sum(
        position.quantity * position.last_price for position in model_positions.values()
    )
    current_symbols = set(model_positions)
    if should_rebalance_on_target:
        target_weights = last_allocation.target_weights
    else:
        target_weights = {
            symbol: ((position.quantity * position.last_price) / portfolio_value if portfolio_value > 0 else 0.0)
            for symbol, position in model_positions.items()
        }
    target_symbols = set(target_weights)

    current_weight_by_symbol = {
        symbol: ((position.quantity * position.last_price) / portfolio_value if portfolio_value > 0 else 0.0)
        for symbol, position in model_positions.items()
    }
    actions: list[dict[str, object]] = []
    risk_flags: list[dict[str, object]] = []
    current_bars = provider.get_bars_on_date(universe, target_trade_date)

    for symbol in sorted(current_symbols | target_symbols):
        current_weight = current_weight_by_symbol.get(symbol, 0.0)
        target_weight = float(target_weights.get(symbol, 0.0))
        delta_weight = target_weight - current_weight
        if symbol in current_symbols and symbol not in target_symbols:
            action = "SELL"
        elif symbol not in current_symbols and symbol in target_symbols:
            action = "BUY"
        elif delta_weight > 1e-6:
            action = "ADD"
        elif delta_weight < -1e-6:
            action = "TRIM"
        else:
            action = "HOLD"

        position = model_positions.get(symbol)
        bar = current_bars.get(symbol)
        history = model_context.history(symbol) if model_context is not None else []
        actions.append(
            {
                "symbol": symbol,
                "action": action,
                "current_quantity": int(position.quantity) if position is not None else 0,
                "current_weight": round(current_weight, 6),
                "target_weight": round(target_weight, 6),
                "delta_weight": round(delta_weight, 6),
                "signal_close": round(
                    position.last_price
                    if position is not None
                    else (history[-1].close if history else (bar.close if bar else 0.0)),
                    6,
                ),
                "next_open": None if bar is None else round(float(bar.open), 6),
            }
        )

        if history:
            latest_bar = history[-1]
            if latest_bar.amount <= max(config.min_daily_amount, 0.0) and latest_bar.amount > 0:
                risk_flags.append(
                    {
                        "symbol": symbol,
                        "flag": "low_liquidity",
                        "detail": f"previous_day_amount={latest_bar.amount:.2f}",
                    }
                )
            if len(history) >= 4:
                start_close = history[-4].close
                if start_close > 0:
                    recent_return = latest_bar.close / start_close - 1.0
                    if recent_return >= 0.12:
                        risk_flags.append(
                            {
                                "symbol": symbol,
                                "flag": "strong_recent_runup",
                                "detail": f"3d_return={recent_return:.4f}",
                            }
                        )

    industry_counts: dict[str, int] = {}
    for symbol in last_selected:
        industry = strategy._industry_by_symbol.get(symbol, "")
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
    for industry, count in sorted(industry_counts.items()):
        if count >= max(2, config.top_k // 2 + (config.top_k % 2 > 0)):
            risk_flags.append(
                {
                    "symbol": "",
                    "flag": "industry_concentration",
                    "detail": f"{industry or 'unknown'}={count}",
                }
            )

    summary = {
        "signal_date": signal_date.isoformat(),
        "execution_date": target_trade_date.isoformat(),
        "decision_reason": last_decision_reason,
        "portfolio_value_pre_open": round(portfolio_value, 2),
        "model_cash_pre_open": round(model_cash, 2),
        "current_position_count": len(model_positions),
        "target_position_count": len(target_symbols),
    }
    payload = {
        "summary": summary,
        "selected_symbols": last_selected,
        "target_weights": {symbol: round(float(weight), 6) for symbol, weight in sorted(target_weights.items())},
        "actions": actions,
        "risk_flags": risk_flags,
    }

    target = Path(config.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def generate_strategy_state(
    config: StrategyStateConfig,
    provider: DataProvider | None = None,
) -> dict[str, object]:
    LOGGER.info(
        "generate strategy state start output=%s mode=%s trade_date=%s scores_path=%s universe_name=%s previous_state=%s",
        config.output_path,
        config.mode,
        config.trade_date,
        config.scores_path,
        config.universe_name or "-",
        config.previous_state_path or "-",
    )
    target_trade_date = date.fromisoformat(config.trade_date)
    scores = pd.read_parquet(config.scores_path).sort_values(["trade_date", "symbol"])
    if scores.empty:
        raise ValueError("scores parquet is empty")

    scores["trade_date"] = pd.to_datetime(scores["trade_date"])
    provider = provider or ParquetDataProvider(config.storage_root)
    strategy = build_score_strategy(config)

    earliest_score_date = scores["trade_date"].min().date()
    previous_state: dict[str, object] | None = None
    replay_start_date = earliest_score_date
    if config.mode in {"continue", "historical"} and config.previous_state_path:
        previous_state = _load_strategy_state(config.previous_state_path)
        replay_start_date = min(replay_start_date, previous_state["as_of_trade_date"])
    trade_dates = provider.get_trade_dates(replay_start_date, target_trade_date)
    if target_trade_date not in trade_dates:
        raise ValueError(f"trade date is not an open trading day: {config.trade_date}")
    target_index = trade_dates.index(target_trade_date)
    if target_index == 0:
        raise ValueError("cannot build strategy state for the first available trade date")
    signal_date = trade_dates[target_index - 1]
    universe = resolve_score_universe(config, scores, as_of_date=signal_date)
    LOGGER.info(
        "generate strategy state resolved signal_date=%s universe_size=%s target_trade_date=%s",
        signal_date.isoformat(),
        len(universe),
        target_trade_date.isoformat(),
    )
    provider = build_preloaded_score_provider(
        storage_root=config.storage_root,
        universe=universe,
        start_date=replay_start_date,
        end_date=target_trade_date,
        lookback=strategy.metadata.lookback_window,
        provider=provider,
    )

    engine = BacktestEngine(provider)
    last_allocation = AllocationDecision(target_weights={}, note="no_prior_allocation")
    last_selected: list[str] = []
    last_decision_reason = "not_rebalanced"
    model_positions: dict[str, Position] = {}
    model_cash = config.initial_cash
    model_context: StrategyContext | None = None
    current_pending_orders: dict[tuple[str, str], BacktestEngine._PendingOrder] = {}
    should_rebalance_on_target = False
    target_rebalance_already_executed = False
    trade_log: list[Trade] = []
    decision_log: list[dict[str, object]] = []

    if config.mode == "initial_entry":
        current_bars = provider.get_bars_on_date(universe, target_trade_date)
        context = StrategyContext(
            trade_date=target_trade_date,
            universe=universe,
            bars=provider.get_history(
                symbols=universe,
                end_date=signal_date,
                lookback=strategy.metadata.lookback_window,
            ),
            positions={},
            cash=config.initial_cash,
        )
        last_selected = strategy.select(context)
        last_allocation = strategy.allocate(context, last_selected)
        last_decision_reason = "initial_entry"
        model_positions = {}
        model_cash = config.initial_cash
        model_context = context
        current_pending_orders = {}
        should_rebalance_on_target = True
        decision_log.append(
            {
                "trade_date": target_trade_date.isoformat(),
                "signal_date": signal_date.isoformat(),
                "decision_reason": last_decision_reason,
                "should_rebalance": True,
                "selected_symbols": ",".join(last_selected),
                "current_position_count": 0,
                "target_position_count": len(last_allocation.target_weights),
                "cash_pre_decision": round(float(config.initial_cash), 2),
            }
        )
    else:
        if config.mode not in {"continue", "historical"}:
            raise ValueError(f"unsupported strategy state mode: {config.mode}")

        if previous_state is not None:
            state = previous_state
            start_index = trade_dates.index(state["as_of_trade_date"]) + (0 if state["execution_pending"] else 1)
            if start_index > target_index:
                raise ValueError("target trade date must be after previous strategy state date")
            cash = float(state["cash"])
            positions = dict(state["positions"])
            pending_orders = dict(state["pending_orders"])
            strategy._hold_days = dict(state["hold_days"])
            strategy._last_rebalance_date = state["last_rebalance_date"]
            bootstrap_allocation = (
                AllocationDecision(target_weights=state["planned_target_weights"], note="carry_forward_planned_target")
                if state["execution_pending"] and state["planned_target_weights"]
                else None
            )
            replay_dates = trade_dates[start_index : target_index + 1]
        else:
            cash = config.initial_cash
            positions = {}
            pending_orders = {}
            bootstrap_allocation = None
            replay_dates = trade_dates[: target_index + 1]

        for trade_date_item in replay_dates:
            trade_date_index = trade_dates.index(trade_date_item)
            previous_trade_date = trade_dates[trade_date_index - 1] if trade_date_index > 0 else None
            bars = provider.get_history(
                symbols=universe,
                end_date=previous_trade_date or trade_date_item,
                lookback=strategy.metadata.lookback_window,
            )
            current_bars = provider.get_bars_on_date(universe, trade_date_item)
            positions = engine._refresh_positions(positions, current_bars)
            cash, positions, pending_trades, _, pending_orders = engine._execute_pending_orders(
                trade_date=trade_date_item,
                current_bars=current_bars,
                cash=cash,
                positions=positions,
                pending_orders=pending_orders,
                config=build_score_backtest_config(config, universe, trade_date_item, trade_date_item),
            )
            trade_log.extend(pending_trades)
            if bootstrap_allocation is not None and trade_date_item == state["as_of_trade_date"]:
                cash, positions, fill_trades, _ = engine._execute_rebalance(
                    trade_date=trade_date_item,
                    current_bars=current_bars,
                    cash=cash,
                    positions=positions,
                    allocation=bootstrap_allocation,
                    config=build_score_backtest_config(config, universe, trade_date_item, trade_date_item),
                )
                trade_log.extend(fill_trades)
                pending_orders = engine._build_pending_orders(
                    cash=cash,
                    positions=positions,
                    target_weights=bootstrap_allocation.target_weights,
                    current_bars=current_bars,
                )
                strategy._last_rebalance_date = trade_date_item
                target_rebalance_already_executed = trade_date_item == target_trade_date
                bootstrap_allocation = None
                if trade_date_item != target_trade_date:
                    continue
            context = StrategyContext(
                trade_date=trade_date_item,
                universe=universe,
                bars=bars,
                positions=positions,
                cash=cash,
            )
            decision = strategy.rebalance(context)
            selected_for_log = strategy.select(context) if decision.should_rebalance else sorted(positions)
            allocation_for_log = strategy.allocate(context, selected_for_log) if decision.should_rebalance else None
            decision_log.append(
                {
                    "trade_date": trade_date_item.isoformat(),
                    "signal_date": (previous_trade_date or signal_date).isoformat(),
                    "decision_reason": decision.reason,
                    "should_rebalance": bool(decision.should_rebalance),
                    "selected_symbols": ",".join(selected_for_log),
                    "current_position_count": len(positions),
                    "target_position_count": (
                        len(allocation_for_log.target_weights) if allocation_for_log is not None else len(positions)
                    ),
                    "cash_pre_decision": round(float(cash), 2),
                }
            )
            if trade_date_item == target_trade_date:
                should_rebalance_on_target = decision.should_rebalance
                if decision.should_rebalance:
                    last_selected = selected_for_log
                    last_allocation = allocation_for_log or strategy.allocate(context, last_selected)
                else:
                    last_selected = sorted(positions)
                last_decision_reason = decision.reason
                model_positions = positions
                model_cash = cash
                model_context = context
                current_pending_orders = pending_orders
                break

            if decision.should_rebalance:
                selected = selected_for_log
                allocation = allocation_for_log or strategy.allocate(context, selected)
                cash, positions, fill_trades, _ = engine._execute_rebalance(
                    trade_date=trade_date_item,
                    current_bars=current_bars,
                    cash=cash,
                    positions=positions,
                    allocation=allocation,
                    config=build_score_backtest_config(config, universe, trade_date_item, trade_date_item),
                )
                trade_log.extend(fill_trades)
                pending_orders = engine._build_pending_orders(
                    cash=cash,
                    positions=positions,
                    target_weights=allocation.target_weights,
                    current_bars=current_bars,
                )

    portfolio_value = model_cash + sum(
        position.quantity * position.last_price for position in model_positions.values()
    )
    current_symbols = set(model_positions)
    if should_rebalance_on_target:
        target_weights = last_allocation.target_weights
    else:
        target_weights = {
            symbol: ((position.quantity * position.last_price) / portfolio_value if portfolio_value > 0 else 0.0)
            for symbol, position in model_positions.items()
        }
    target_symbols = set(target_weights)
    current_weight_by_symbol = {
        symbol: ((position.quantity * position.last_price) / portfolio_value if portfolio_value > 0 else 0.0)
        for symbol, position in model_positions.items()
    }
    actions: list[dict[str, object]] = []
    risk_flags: list[dict[str, object]] = []
    current_bars = provider.get_bars_on_date(universe, target_trade_date)

    for symbol in sorted(current_symbols | target_symbols):
        current_weight = current_weight_by_symbol.get(symbol, 0.0)
        target_weight = float(target_weights.get(symbol, 0.0))
        delta_weight = target_weight - current_weight
        if symbol in current_symbols and symbol not in target_symbols:
            action = "SELL"
        elif symbol not in current_symbols and symbol in target_symbols:
            action = "BUY"
        elif delta_weight > 1e-6:
            action = "ADD"
        elif delta_weight < -1e-6:
            action = "TRIM"
        else:
            action = "HOLD"

        position = model_positions.get(symbol)
        bar = current_bars.get(symbol)
        actions.append(
            {
                "symbol": symbol,
                "action": action,
                "current_quantity": int(position.quantity) if position is not None else 0,
                "current_weight": round(current_weight, 6),
                "target_weight": round(target_weight, 6),
                "delta_weight": round(delta_weight, 6),
                "signal_close": round(position.last_price if position is not None else (bar.close if bar else 0.0), 6),
                "next_open": None if bar is None else round(float(bar.open), 6),
            }
        )

        history = model_context.history(symbol) if model_context is not None else []
        if history:
            latest_bar = history[-1]
            if latest_bar.amount <= max(config.min_daily_amount, 0.0) and latest_bar.amount > 0:
                risk_flags.append(
                    {
                        "symbol": symbol,
                        "flag": "low_liquidity",
                        "detail": f"previous_day_amount={latest_bar.amount:.2f}",
                    }
                )
            if len(history) >= 4:
                start_close = history[-4].close
                if start_close > 0:
                    recent_return = latest_bar.close / start_close - 1.0
                    if recent_return >= 0.12:
                        risk_flags.append(
                            {
                                "symbol": symbol,
                                "flag": "strong_recent_runup",
                                "detail": f"3d_return={recent_return:.4f}",
                            }
                        )

    industry_counts: dict[str, int] = {}
    for symbol in last_selected:
        industry = strategy._industry_by_symbol.get(symbol, "")
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
    for industry, count in sorted(industry_counts.items()):
        if count >= max(2, config.top_k // 2 + (config.top_k % 2 > 0)):
            risk_flags.append(
                {
                    "symbol": "",
                    "flag": "industry_concentration",
                    "detail": f"{industry or 'unknown'}={count}",
                }
            )

    if should_rebalance_on_target and config.simulate_trade_execution and not target_rebalance_already_executed:
        has_trade_day_bars = any(bar is not None for bar in current_bars.values())
        if has_trade_day_bars:
            next_cash, next_positions, next_fill_trades, _ = engine._execute_rebalance(
                trade_date=target_trade_date,
                current_bars=current_bars,
                cash=model_cash,
                positions=model_positions,
                allocation=last_allocation,
                config=build_score_backtest_config(config, universe, target_trade_date, target_trade_date),
            )
            trade_log.extend(next_fill_trades)
            next_pending_orders = engine._build_pending_orders(
                cash=next_cash,
                positions=next_positions,
                target_weights=last_allocation.target_weights,
                current_bars=current_bars,
            )
            execution_pending = False
            next_last_rebalance_date = strategy._last_rebalance_date
            planned_target_weights: dict[str, float] = {}
        else:
            next_cash = model_cash
            next_positions = dict(model_positions)
            next_pending_orders = dict(current_pending_orders)
            execution_pending = True
            next_last_rebalance_date = None
            planned_target_weights = {
                symbol: round(float(weight), 6) for symbol, weight in sorted(last_allocation.target_weights.items())
            }
    elif should_rebalance_on_target and target_rebalance_already_executed:
        next_cash = model_cash
        next_positions = dict(model_positions)
        next_pending_orders = dict(current_pending_orders)
        execution_pending = bool(next_pending_orders)
        next_last_rebalance_date = strategy._last_rebalance_date
        planned_target_weights = (
            {
                symbol: round(float(weight), 6)
                for symbol, weight in sorted(last_allocation.target_weights.items())
            }
            if execution_pending
            else {}
        )
    elif should_rebalance_on_target:
        next_cash = model_cash
        next_positions = dict(model_positions)
        next_pending_orders = dict(current_pending_orders)
        execution_pending = True
        next_last_rebalance_date = None
        planned_target_weights = {
            symbol: round(float(weight), 6) for symbol, weight in sorted(last_allocation.target_weights.items())
        }
    else:
        next_cash = model_cash
        next_positions = dict(model_positions)
        next_pending_orders = dict(current_pending_orders)
        execution_pending = False
        next_last_rebalance_date = strategy._last_rebalance_date
        planned_target_weights = {}

    _annotate_action_execution(
        actions=actions,
        trade_log=trade_log,
        trade_date=target_trade_date,
        portfolio_value=portfolio_value,
        should_rebalance=should_rebalance_on_target,
    )

    next_portfolio_value = next_cash + sum(position.quantity * position.last_price for position in next_positions.values())
    summary = {
        "signal_date": signal_date.isoformat(),
        "execution_date": target_trade_date.isoformat(),
        "decision_reason": last_decision_reason,
        "state_mode": config.mode,
        "portfolio_value_pre_open": round(portfolio_value, 2),
        "model_cash_pre_open": round(model_cash, 2),
        "current_position_count": len(model_positions),
        "target_position_count": len(target_symbols),
    }
    payload = {
        "state_version": "v1.1",
        "summary": summary,
        "strategy_config": {
            "scores_path": config.scores_path,
            "storage_root": config.storage_root,
            "top_k": config.top_k,
            "rebalance_every": config.rebalance_every,
            "lookback_window": config.lookback_window,
            "min_hold_bars": config.min_hold_bars,
            "keep_buffer": config.keep_buffer,
            "min_turnover_names": config.min_turnover_names,
            "max_names_per_industry": config.max_names_per_industry,
            "initial_cash": config.initial_cash,
        },
        "pre_open": {
            "cash": round(model_cash, 2),
            "positions": _serialize_positions(model_positions, portfolio_value),
            "pending_orders": _serialize_pending_orders(current_pending_orders),
        },
        "plan": {
            "selected_symbols": last_selected,
            "target_weights": {symbol: round(float(weight), 6) for symbol, weight in sorted(target_weights.items())},
            "actions": actions,
            "risk_flags": risk_flags,
        },
        "next_state": {
            "as_of_trade_date": target_trade_date.isoformat(),
            "cash": round(next_cash, 2),
            "portfolio_value": round(next_portfolio_value, 2),
            "positions": _serialize_positions(next_positions, next_portfolio_value),
            "pending_orders": _serialize_pending_orders(next_pending_orders),
            "hold_days": {symbol: int(days) for symbol, days in sorted(strategy._hold_days.items())},
            "last_rebalance_date": (
                None if next_last_rebalance_date is None else next_last_rebalance_date.isoformat()
            ),
            "execution_pending": execution_pending,
            "planned_target_weights": planned_target_weights,
        },
    }

    target = Path(config.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_trade_log(target.with_name("trades.csv"), trade_log)
    _write_decision_log(target.with_name("decision_log.csv"), decision_log)
    LOGGER.info(
        "generate strategy state complete output=%s mode=%s decision_reason=%s selected=%s current_positions=%s target_positions=%s",
        target.as_posix(),
        config.mode,
        summary["decision_reason"],
        len(payload["plan"]["selected_symbols"]),
        summary["current_position_count"],
        summary["target_position_count"],
    )
    return payload


def analyze_score_layers(config: LayeredAnalysisConfig) -> dict[str, object]:
    frame = pd.read_parquet(config.scores_path).sort_values(["trade_date", "prediction"], ascending=[True, False])
    grouped_records: list[dict[str, object]] = []

    for trade_date, day_frame in frame.groupby("trade_date"):
        if len(day_frame) < config.bins:
            continue
        working = day_frame.copy()
        working["layer"] = pd.qcut(
            working["prediction"].rank(method="first"),
            q=config.bins,
            labels=False,
        )
        working["layer"] = config.bins - 1 - working["layer"].astype(int)
        layer_mean = working.groupby("layer")["label"].mean().to_dict()
        grouped_records.append(
            {
                "trade_date": pd.Timestamp(trade_date).date().isoformat(),
                "top_layer_return": float(layer_mean.get(0, 0.0)),
                "bottom_layer_return": float(layer_mean.get(config.bins - 1, 0.0)),
                "top_bottom_spread": float(layer_mean.get(0, 0.0) - layer_mean.get(config.bins - 1, 0.0)),
            }
        )

    analysis_frame = pd.DataFrame(grouped_records)
    if analysis_frame.empty:
        raise ValueError("no valid layered analysis rows were generated")

    summary = {
        "rows": int(len(analysis_frame)),
        "mean_top_layer_return": float(analysis_frame["top_layer_return"].mean()),
        "mean_bottom_layer_return": float(analysis_frame["bottom_layer_return"].mean()),
        "mean_top_bottom_spread": float(analysis_frame["top_bottom_spread"].mean()),
        "positive_spread_ratio": float((analysis_frame["top_bottom_spread"] > 0).mean()),
    }
    payload = {"summary": summary, "by_date": grouped_records}

    target = Path(config.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def compare_backtest_monthly_returns(config: MonthlyComparisonConfig) -> dict[str, object]:
    if not config.result_dirs:
        raise ValueError("result_dirs must not be empty")
    if len(config.result_dirs) != len(config.labels):
        raise ValueError("result_dirs and labels must have the same length")

    merged: pd.DataFrame | None = None
    summary_rows: list[dict[str, object]] = []
    per_label_monthly: dict[str, pd.Series] = {}

    for label, result_dir in zip(config.labels, config.result_dirs):
        equity_path = Path(result_dir) / "equity_curve.csv"
        equity = pd.read_csv(equity_path)
        if equity.empty:
            raise ValueError(f"equity curve is empty: {equity_path}")
        equity["trade_date"] = pd.to_datetime(equity["trade_date"])
        equity["equity"] = pd.to_numeric(equity["equity"], errors="coerce")
        equity = equity.dropna(subset=["equity"]).sort_values("trade_date")
        if equity.empty:
            raise ValueError(f"equity curve has no valid values: {equity_path}")

        monthly = (
            equity.assign(trade_month=equity["trade_date"].dt.strftime("%Y-%m"))
            .groupby("trade_month")["equity"]
            .agg(["first", "last"])
        )
        monthly_return = monthly["last"] / monthly["first"] - 1.0
        per_label_monthly[label] = monthly_return

        row = monthly_return.rename(label).reset_index()
        row.columns = ["trade_month", label]
        merged = row if merged is None else merged.merge(row, on="trade_month", how="outer")

        best_month = monthly_return.idxmax()
        worst_month = monthly_return.idxmin()
        summary_rows.append(
            {
                "label": label,
                "best_month": str(best_month),
                "best_month_return": float(monthly_return.loc[best_month]),
                "worst_month": str(worst_month),
                "worst_month_return": float(monthly_return.loc[worst_month]),
            }
        )

    assert merged is not None
    merged = merged.sort_values("trade_month").reset_index(drop=True)

    baseline_label = config.labels[0]
    baseline_monthly = per_label_monthly[baseline_label]
    if baseline_label not in merged.columns:
        raise ValueError("baseline label missing from merged monthly comparison")

    for label in config.labels[1:]:
        diff_column = f"{label}_minus_{baseline_label}"
        current_monthly = per_label_monthly[label]
        diff = current_monthly.subtract(baseline_monthly, fill_value=0.0)
        diff_frame = diff.rename(diff_column).reset_index()
        diff_frame.columns = ["trade_month", diff_column]
        merged = merged.merge(diff_frame, on="trade_month", how="left")

    output_rows = []
    for _, row in merged.iterrows():
        record = {"trade_month": str(row["trade_month"])}
        for column in merged.columns:
            if column == "trade_month":
                continue
            value = row[column]
            record[column] = None if pd.isna(value) else float(value)
        output_rows.append(record)

    payload = {
        "summary": {
            "baseline_label": baseline_label,
            "labels": list(config.labels),
            "by_label": summary_rows,
        },
        "by_month": output_rows,
    }

    target = Path(config.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def analyze_start_date_robustness(
    config: StartDateRobustnessConfig,
    provider: DataProvider | None = None,
) -> dict[str, object]:
    scores = pd.read_parquet(config.scores_path).sort_values(["trade_date", "symbol"])
    if scores.empty:
        raise ValueError("scores parquet is empty")

    analysis_start_date = date.fromisoformat(config.analysis_start_date)
    analysis_end_date = date.fromisoformat(config.analysis_end_date)
    if analysis_end_date < analysis_start_date:
        raise ValueError("analysis_end_date must be on or after analysis_start_date")
    if config.holding_months <= 0:
        raise ValueError("holding_months must be positive")
    if config.cadence not in {"daily", "monthly"}:
        raise ValueError("cadence must be one of: daily, monthly")

    scores["trade_date"] = pd.to_datetime(scores["trade_date"])
    provider = provider or ParquetDataProvider(config.storage_root)
    strategy = build_score_strategy(config)

    earliest_score_date = scores["trade_date"].min().date()
    latest_score_date = scores["trade_date"].max().date()
    candidate_end = min(analysis_end_date, latest_score_date)
    if candidate_end < analysis_start_date:
        raise ValueError("analysis window falls outside score coverage")

    trade_dates = provider.get_trade_dates(earliest_score_date, candidate_end)
    candidate_trade_dates = [item for item in trade_dates if analysis_start_date <= item <= candidate_end]
    if not candidate_trade_dates:
        raise ValueError("no open trade dates within analysis window")

    sample_anchor_date = candidate_trade_dates[0]
    universe = resolve_score_universe(config, scores, as_of_date=sample_anchor_date)
    provider = build_preloaded_score_provider(
        storage_root=config.storage_root,
        universe=universe,
        start_date=earliest_score_date,
        end_date=candidate_end,
        lookback=strategy.metadata.lookback_window,
        provider=provider,
    )

    sampled_starts = _sample_robustness_start_dates(candidate_trade_dates, config.cadence)
    result_rows: list[dict[str, object]] = []
    engine = BacktestEngine(provider)

    for start_trade_date in sampled_starts:
        end_trade_date = _resolve_robustness_end_date(candidate_trade_dates, start_trade_date, config.holding_months)
        if end_trade_date is None or end_trade_date <= start_trade_date:
            continue

        run_strategy = build_score_strategy(config)
        backtest = build_score_backtest_config(config, universe, start_trade_date, end_trade_date)
        result = engine.run_with_strategy(backtest, run_strategy)
        result_rows.append(
            {
                "start_date": start_trade_date.isoformat(),
                "end_date": end_trade_date.isoformat(),
                "trade_days": len(result.equity_curve),
                "total_return": float(result.total_return),
                "annual_return": float(result.annual_return),
                "max_drawdown": float(result.max_drawdown),
                "sharpe_ratio": float(result.sharpe_ratio),
                "turnover_ratio": float(result.turnover_ratio),
                "filled_trade_count": int(result.filled_trade_count),
                "rejected_trade_count": int(result.rejected_trade_count),
            }
        )

    if not result_rows:
        raise ValueError("no valid robustness windows were generated")

    frame = pd.DataFrame(result_rows)
    best = frame.loc[frame["total_return"].idxmax()]
    worst = frame.loc[frame["total_return"].idxmin()]
    payload = {
        "summary": {
            "analysis_start_date": analysis_start_date.isoformat(),
            "analysis_end_date": candidate_end.isoformat(),
            "holding_months": int(config.holding_months),
            "cadence": config.cadence,
            "sample_count": int(len(frame)),
            "mean_total_return": float(frame["total_return"].mean()),
            "median_total_return": float(frame["total_return"].median()),
            "min_total_return": float(frame["total_return"].min()),
            "max_total_return": float(frame["total_return"].max()),
            "win_rate": float((frame["total_return"] > 0).mean()),
            "mean_max_drawdown": float(frame["max_drawdown"].mean()),
            "best_start_date": str(best["start_date"]),
            "best_total_return": float(best["total_return"]),
            "worst_start_date": str(worst["start_date"]),
            "worst_total_return": float(worst["total_return"]),
        },
        "by_start_date": result_rows,
    }

    target = Path(config.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _sample_robustness_start_dates(trade_dates: list[date], cadence: str) -> list[date]:
    if cadence == "daily":
        return list(trade_dates)
    sampled: list[date] = []
    seen_months: set[str] = set()
    for trade_date_item in trade_dates:
        month_key = trade_date_item.strftime("%Y-%m")
        if month_key in seen_months:
            continue
        sampled.append(trade_date_item)
        seen_months.add(month_key)
    return sampled


def _resolve_robustness_end_date(
    trade_dates: list[date],
    start_trade_date: date,
    holding_months: int,
) -> date | None:
    anchor = (pd.Timestamp(start_trade_date) + pd.DateOffset(months=holding_months)).date()
    eligible = [item for item in trade_dates if item >= anchor]
    if not eligible:
        return None
    end_trade_date = eligible[0]
    if end_trade_date <= start_trade_date:
        return None
    return end_trade_date


def analyze_monthly_risk_exposures(config: RiskExposureConfig) -> dict[str, object]:
    result_root = Path(config.result_dir)
    trades = pd.read_csv(result_root / "trades.csv")
    equity = pd.read_csv(result_root / "equity_curve.csv")
    bars = pd.read_parquet(
        Path(config.storage_root) / "parquet" / "bars" / "daily.parquet",
        columns=["trade_date", "symbol", "close", "close_adj", "amount", "turnover_rate"],
    )
    instruments = pd.read_parquet(
        Path(config.storage_root) / "parquet" / "instruments" / "ashare_instruments.parquet",
        columns=["symbol", "industry_level_1"],
    )

    if equity.empty:
        raise ValueError("equity curve is empty")

    trades["trade_date"] = pd.to_datetime(trades["trade_date"])
    equity["trade_date"] = pd.to_datetime(equity["trade_date"])
    filled = trades.loc[trades["status"] == "filled"].copy()
    filled["signed_quantity"] = filled.apply(
        lambda row: int(row["quantity"]) if row["side"] == "BUY" else -int(row["quantity"]),
        axis=1,
    )

    bars["trade_date"] = pd.to_datetime(bars["trade_date"])
    price_column = "close_adj" if "close_adj" in bars.columns else "close"
    bars["price_used"] = pd.to_numeric(bars[price_column], errors="coerce").fillna(pd.to_numeric(bars["close"], errors="coerce"))
    bars["daily_return"] = bars.groupby("symbol")["price_used"].pct_change()
    bars["rolling_volatility"] = (
        bars.groupby("symbol")["daily_return"]
        .rolling(max(2, config.volatility_window), min_periods=2)
        .std()
        .reset_index(level=0, drop=True)
    )

    bars = bars.merge(instruments, how="left", on="symbol")
    bars["industry_level_1"] = bars["industry_level_1"].fillna("")

    quantity_changes = (
        filled.groupby(["trade_date", "symbol"], as_index=False)["signed_quantity"].sum().sort_values(["trade_date", "symbol"])
    )
    exposure_rows: list[dict[str, object]] = []
    positions: dict[str, int] = {}
    changes_by_date = {
        pd.Timestamp(trade_date): day_frame
        for trade_date, day_frame in quantity_changes.groupby("trade_date")
    }

    for trade_date in equity["trade_date"].sort_values():
        day_changes = changes_by_date.get(pd.Timestamp(trade_date))
        if day_changes is not None:
            for _, row in day_changes.iterrows():
                symbol = str(row["symbol"])
                next_quantity = positions.get(symbol, 0) + int(row["signed_quantity"])
                if next_quantity > 0:
                    positions[symbol] = next_quantity
                else:
                    positions.pop(symbol, None)

        if not positions:
            exposure_rows.append(
                {
                    "trade_date": trade_date.strftime("%Y-%m-%d"),
                    "trade_month": trade_date.strftime("%Y-%m"),
                    "name_count": 0,
                    "top_position_weight": 0.0,
                    "position_hhi": 0.0,
                    "top_industry_weight": 0.0,
                    "weighted_avg_amount": 0.0,
                    "weighted_avg_turnover_rate": 0.0,
                    "weighted_avg_volatility": 0.0,
                    "top_industries": [],
                }
            )
            continue

        day_bars = bars.loc[(bars["trade_date"] == trade_date) & (bars["symbol"].isin(positions))]
        if day_bars.empty:
            continue
        day_bars = day_bars.copy()
        day_bars["quantity"] = day_bars["symbol"].map(positions).fillna(0).astype(int)
        day_bars["market_value"] = day_bars["quantity"] * day_bars["price_used"]
        day_bars = day_bars.loc[day_bars["market_value"] > 0]
        if day_bars.empty:
            continue
        total_value = float(day_bars["market_value"].sum())
        day_bars["weight"] = day_bars["market_value"] / total_value
        industry_weights = (
            day_bars.groupby("industry_level_1", as_index=False)["weight"].sum().sort_values("weight", ascending=False)
        )
        exposure_rows.append(
            {
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "trade_month": trade_date.strftime("%Y-%m"),
                "name_count": int(len(day_bars)),
                "top_position_weight": float(day_bars["weight"].max()),
                "position_hhi": float((day_bars["weight"] ** 2).sum()),
                "top_industry_weight": float(industry_weights["weight"].max()) if not industry_weights.empty else 0.0,
                "weighted_avg_amount": float((day_bars["weight"] * day_bars["amount"].fillna(0.0)).sum()),
                "weighted_avg_turnover_rate": float(
                    (day_bars["weight"] * day_bars["turnover_rate"].fillna(0.0)).sum()
                ),
                "weighted_avg_volatility": float(
                    (day_bars["weight"] * day_bars["rolling_volatility"].fillna(0.0)).sum()
                ),
                "top_industries": [
                    {"industry": str(row["industry_level_1"]), "weight": float(row["weight"])}
                    for _, row in industry_weights.head(config.top_industries).iterrows()
                ],
            }
        )

    exposure_frame = pd.DataFrame(exposure_rows)
    if exposure_frame.empty:
        raise ValueError("no daily exposures were generated")

    monthly_rows: list[dict[str, object]] = []
    for trade_month, month_frame in exposure_frame.groupby("trade_month"):
        industry_totals: dict[str, float] = {}
        for industries in month_frame["top_industries"]:
            for item in industries:
                industry = str(item["industry"])
                industry_totals[industry] = industry_totals.get(industry, 0.0) + float(item["weight"])
        top_industries = sorted(industry_totals.items(), key=lambda item: item[1], reverse=True)[: config.top_industries]
        monthly_rows.append(
            {
                "trade_month": str(trade_month),
                "avg_name_count": float(month_frame["name_count"].mean()),
                "avg_top_position_weight": float(month_frame["top_position_weight"].mean()),
                "avg_position_hhi": float(month_frame["position_hhi"].mean()),
                "avg_top_industry_weight": float(month_frame["top_industry_weight"].mean()),
                "avg_weighted_amount": float(month_frame["weighted_avg_amount"].mean()),
                "avg_weighted_turnover_rate": float(month_frame["weighted_avg_turnover_rate"].mean()),
                "avg_weighted_volatility": float(month_frame["weighted_avg_volatility"].mean()),
                "top_industries": [
                    {"industry": industry, "avg_weight_proxy": float(weight / len(month_frame))}
                    for industry, weight in top_industries
                ],
            }
        )

    payload = {
        "summary": {
            "result_dir": config.result_dir,
            "monthly_count": int(len(monthly_rows)),
            "volatility_window": int(config.volatility_window),
        },
        "by_month": monthly_rows,
    }

    target = Path(config.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def analyze_trade_capacity(config: CapacityAnalysisConfig) -> dict[str, object]:
    if config.base_capital <= 0:
        raise ValueError("base_capital must be positive")
    if not config.scale_capitals:
        raise ValueError("scale_capitals must not be empty")

    trades = pd.read_csv(config.trades_path)
    if trades.empty:
        raise ValueError("trades file is empty")

    filled = trades.loc[trades["status"] == "filled"].copy()
    if filled.empty:
        raise ValueError("no filled trades were found in trades file")

    filled["trade_date"] = pd.to_datetime(filled["trade_date"])
    filled["amount"] = pd.to_numeric(filled["amount"], errors="coerce")
    filled = filled.loc[filled["amount"].notna() & (filled["amount"] > 0)]
    if filled.empty:
        raise ValueError("no positive filled trade amounts were found in trades file")

    bars_path = Path(config.storage_root) / "parquet" / "bars" / "daily.parquet"
    bars = pd.read_parquet(bars_path, columns=["trade_date", "symbol", "amount"])
    bars["trade_date"] = pd.to_datetime(bars["trade_date"])
    bars["amount"] = pd.to_numeric(bars["amount"], errors="coerce").fillna(0.0)

    merged = filled.merge(bars, how="left", on=["trade_date", "symbol"], suffixes=("", "_market"))
    merged = merged.rename(columns={"amount_market": "market_amount", "amount": "trade_amount"})
    merged["market_amount"] = pd.to_numeric(merged["market_amount"], errors="coerce").fillna(0.0)
    merged["base_participation"] = 0.0
    positive_mask = merged["market_amount"] > 0
    merged.loc[positive_mask, "base_participation"] = (
        merged.loc[positive_mask, "trade_amount"] / merged.loc[positive_mask, "market_amount"]
    )

    threshold_values = tuple(sorted(set(float(value) for value in config.participation_thresholds if value > 0)))
    scale_capitals = tuple(float(value) for value in config.scale_capitals if value > 0)
    if not scale_capitals:
        raise ValueError("scale_capitals must contain at least one positive value")

    scale_summaries: list[dict[str, object]] = []
    for capital in scale_capitals:
        scale_ratio = capital / config.base_capital
        participation = merged["base_participation"] * scale_ratio
        valid_participation = participation.loc[merged["market_amount"] > 0]
        if valid_participation.empty:
            raise ValueError("no trades could be matched to positive market amounts")

        threshold_summary = {
            f"{threshold:.2%}": float((valid_participation > threshold).mean())
            for threshold in threshold_values
        }
        stressed = merged.assign(
            scaled_trade_amount=merged["trade_amount"] * scale_ratio,
            participation=participation,
            capital=capital,
        ).sort_values("participation", ascending=False)
        stressed["trade_month"] = stressed["trade_date"].dt.strftime("%Y-%m")

        symbol_summary = (
            stressed.groupby("symbol", as_index=False)
            .agg(
                trade_count=("symbol", "size"),
                total_scaled_trade_amount=("scaled_trade_amount", "sum"),
                mean_participation=("participation", "mean"),
                max_participation=("participation", "max"),
            )
            .sort_values(["max_participation", "mean_participation"], ascending=False)
        )
        month_summary = (
            stressed.groupby("trade_month", as_index=False)
            .agg(
                trade_count=("trade_month", "size"),
                total_scaled_trade_amount=("scaled_trade_amount", "sum"),
                mean_participation=("participation", "mean"),
                max_participation=("participation", "max"),
            )
            .sort_values(["max_participation", "mean_participation"], ascending=False)
        )

        scale_summaries.append(
            {
                "capital": capital,
                "scale_ratio": scale_ratio,
                "filled_trade_count": int(len(valid_participation)),
                "participation_mean": float(valid_participation.mean()),
                "participation_median": float(valid_participation.median()),
                "participation_p90": float(valid_participation.quantile(0.90)),
                "participation_p95": float(valid_participation.quantile(0.95)),
                "participation_max": float(valid_participation.max()),
                "threshold_breach_ratio": threshold_summary,
                "top_stressed_trades": [
                    {
                        "trade_date": row["trade_date"].date().isoformat(),
                        "symbol": str(row["symbol"]),
                        "side": str(row["side"]),
                        "trade_amount": float(row["trade_amount"]),
                        "scaled_trade_amount": float(row["scaled_trade_amount"]),
                        "market_amount": float(row["market_amount"]),
                        "participation": float(row["participation"]),
                    }
                    for _, row in stressed.head(config.top_trade_count).iterrows()
                ],
                "top_stressed_symbols": [
                    {
                        "symbol": str(row["symbol"]),
                        "trade_count": int(row["trade_count"]),
                        "total_scaled_trade_amount": float(row["total_scaled_trade_amount"]),
                        "mean_participation": float(row["mean_participation"]),
                        "max_participation": float(row["max_participation"]),
                    }
                    for _, row in symbol_summary.head(config.top_trade_count).iterrows()
                ],
                "top_stressed_months": [
                    {
                        "trade_month": str(row["trade_month"]),
                        "trade_count": int(row["trade_count"]),
                        "total_scaled_trade_amount": float(row["total_scaled_trade_amount"]),
                        "mean_participation": float(row["mean_participation"]),
                        "max_participation": float(row["max_participation"]),
                    }
                    for _, row in month_summary.head(config.top_trade_count).iterrows()
                ],
            }
        )

    payload = {
        "summary": {
            "base_capital": float(config.base_capital),
            "scale_capitals": [float(value) for value in scale_capitals],
            "participation_thresholds": [float(value) for value in threshold_values],
            "filled_trade_count": int(len(merged)),
            "matched_positive_market_amount_count": int((merged["market_amount"] > 0).sum()),
        },
        "by_scale": scale_summaries,
    }

    target = Path(config.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload
