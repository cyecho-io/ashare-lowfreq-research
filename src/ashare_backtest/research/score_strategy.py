from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from ashare_backtest.protocol import (
    AllocationDecision,
    BaseStrategy,
    RebalanceDecision,
    StrategyContext,
    StrategyMetadata,
)


@dataclass(frozen=True)
class ScoreStrategyConfig:
    scores_path: str
    storage_root: str = "storage"
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


class ScoreTopKStrategy(BaseStrategy):
    def __init__(self, config: ScoreStrategyConfig) -> None:
        self.config = config
        self.metadata = StrategyMetadata(
            name="score_top_k",
            description="Use model scores to hold top-k names on a fixed rebalance schedule",
            lookback_window=config.lookback_window,
        )
        scores = pd.read_parquet(config.scores_path).sort_values(["trade_date", "symbol"])
        self._scores_by_date: dict[str, list[tuple[float, str]]] = {}
        self._score_value_by_date: dict[str, dict[str, float]] = {}
        for trade_date, day_frame in scores.groupby("trade_date"):
            ordered = day_frame.sort_values("prediction", ascending=False)
            score_key = pd.Timestamp(trade_date).date().isoformat()
            self._scores_by_date[score_key] = [
                (float(row["prediction"]), str(row["symbol"])) for _, row in ordered.iterrows()
            ]
            self._score_value_by_date[score_key] = {
                str(row["symbol"]): float(row["prediction"]) for _, row in day_frame.iterrows()
            }
        instruments_path = Path(config.storage_root) / "parquet" / "instruments" / "ashare_instruments.parquet"
        instruments = pd.read_parquet(instruments_path)
        if "is_st" not in instruments.columns:
            instruments["is_st"] = False
        else:
            instruments["is_st"] = instruments["is_st"].fillna(False).astype(bool)
        self._industry_by_symbol = {
            str(row["symbol"]): str(row["industry_level_1"]) if pd.notna(row["industry_level_1"]) else ""
            for _, row in instruments.iterrows()
        }
        self._st_symbols = {
            str(row["symbol"])
            for _, row in instruments.loc[instruments["is_st"], ["symbol"]].iterrows()
        }
        self._hold_days: dict[str, int] = {}
        self._last_rebalance_date: date | None = None

    def rebalance(self, context: StrategyContext) -> RebalanceDecision:
        eligible_universe = self._eligible_universe(context)
        if not eligible_universe:
            return RebalanceDecision(False, "empty_universe")
        sample_history = context.history(eligible_universe[0])
        if len(sample_history) < self.metadata.lookback_window:
            return RebalanceDecision(False, "insufficient_history")
        if self._last_rebalance_date is not None:
            elapsed = self._trading_day_distance(sample_history, self._last_rebalance_date)
            if elapsed < self.config.rebalance_every:
                return RebalanceDecision(False, "rebalance_schedule")
        score_date = self._score_date(context)
        if score_date is None or score_date.isoformat() not in self._scores_by_date:
            return RebalanceDecision(False, "missing_scores")
        return RebalanceDecision(True, "model_score_schedule")

    def select(self, context: StrategyContext) -> list[str]:
        score_date = self._score_date(context)
        if score_date is None:
            return []
        ranked = self._scores_by_date.get(score_date.isoformat(), [])
        allowed = set(self._eligible_universe(context))
        rank_map = {symbol: index for index, (_, symbol) in enumerate(ranked) if symbol in allowed}
        liquidity_ok = self._liquidity_ok_symbols(context)
        self._advance_holding_days(context)

        keep_limit = self.config.top_k + self.config.keep_buffer - 1
        retained: list[str] = []
        for symbol in context.positions:
            rank = rank_map.get(symbol)
            if self._should_retain_position(
                symbol=symbol,
                rank=rank,
                keep_limit=keep_limit,
                liquidity_ok=liquidity_ok,
                context=context,
                score_date=score_date.isoformat(),
            ):
                retained.append(symbol)

        selected = list(dict.fromkeys(retained))
        industry_counts = self._industry_counts(selected)
        for _, symbol in ranked:
            if symbol not in allowed or symbol in selected:
                continue
            if symbol not in liquidity_ok:
                continue
            if not self._can_add_industry(symbol, industry_counts):
                continue
            selected.append(symbol)
            industry = self._industry_by_symbol.get(symbol, "")
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
            if len(selected) >= self.config.top_k:
                break
        return selected[: self.config.top_k]

    def allocate(
        self,
        context: StrategyContext,
        selected_symbols: list[str],
    ) -> AllocationDecision:
        if not selected_symbols:
            return AllocationDecision(target_weights={}, note="no_model_selection")
        self._last_rebalance_date = context.trade_date
        current_symbols = set(context.positions)
        target_symbols = set(selected_symbols)
        score_date = self._score_date(context)
        changed_names = len(current_symbols - target_symbols) + len(target_symbols - current_symbols)
        stale_symbols = {
            symbol
            for symbol in current_symbols
            if not self._has_fresh_history(symbol, context, score_date)
        }
        if current_symbols and not stale_symbols and changed_names < self.config.min_turnover_names:
            retained = sorted(current_symbols)
            if retained:
                weight = round(1 / len(retained), 6)
                return AllocationDecision(
                    target_weights=self._apply_position_weight_cap({symbol: weight for symbol in retained}),
                    note="turnover_below_threshold_keep_current",
                )
        target_weights = self._build_target_weights(context, selected_symbols)
        return AllocationDecision(
            target_weights=target_weights,
            note="model_top_k_with_trim_control",
        )

    @staticmethod
    def _score_date(context: StrategyContext):
        if not context.universe:
            return None
        sample_history = context.history(context.universe[0])
        if not sample_history:
            return None
        return sample_history[-1].trade_date

    def _eligible_universe(self, context: StrategyContext) -> tuple[str, ...]:
        return tuple(symbol for symbol in context.universe if symbol not in self._st_symbols)

    def _advance_holding_days(self, context: StrategyContext) -> None:
        active_symbols = set(context.positions)
        next_hold_days: dict[str, int] = {}
        for symbol in active_symbols:
            next_hold_days[symbol] = self._hold_days.get(symbol, 0) + 1
        self._hold_days = next_hold_days

    @staticmethod
    def _has_fresh_history(symbol: str, context: StrategyContext, score_date: date | None) -> bool:
        history = context.history(symbol)
        if not history or score_date is None:
            return False
        return history[-1].trade_date == score_date

    def _liquidity_ok_symbols(self, context: StrategyContext) -> set[str]:
        passed: set[str] = set()
        for symbol in context.universe:
            history = context.history(symbol)
            if not history:
                continue
            latest_bar = history[-1]
            if self.config.min_daily_amount > 0 and latest_bar.amount < self.config.min_daily_amount:
                continue
            if self.config.max_close_price > 0 and latest_bar.close > self.config.max_close_price:
                continue
            passed.add(symbol)
        return passed

    def _industry_counts(self, symbols: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for symbol in symbols:
            industry = self._industry_by_symbol.get(symbol, "")
            counts[industry] = counts.get(industry, 0) + 1
        return counts

    def _can_add_industry(self, symbol: str, industry_counts: dict[str, int]) -> bool:
        if self.config.max_names_per_industry <= 0:
            return True
        industry = self._industry_by_symbol.get(symbol, "")
        return industry_counts.get(industry, 0) < self.config.max_names_per_industry

    def _should_retain_position(
        self,
        symbol: str,
        rank: int | None,
        keep_limit: int,
        liquidity_ok: set[str],
        context: StrategyContext,
        score_date: str,
    ) -> bool:
        if rank is None or symbol not in liquidity_ok:
            return False

        hold_days = self._hold_days.get(symbol, 0)
        if hold_days < self.config.min_hold_bars:
            return True
        if self.config.exit_policy == "trailing_drawdown":
            return not self._hit_trailing_stop(symbol, context)
        if self.config.exit_policy == "score_reversal":
            if rank <= keep_limit:
                return True
            return not self._hit_score_reversal(symbol, score_date)
        if self.config.exit_policy == "score_price_hybrid":
            if rank <= keep_limit:
                return True
            return not self._hit_score_price_hybrid(symbol, context, score_date)
        if self.config.exit_policy != "rank_momentum_grace":
            if rank <= keep_limit:
                return True
            return self._should_extend_strong_position(rank, keep_limit, symbol, context)

        core_keep_limit = self.config.top_k - 1
        if rank <= core_keep_limit:
            return True

        grace_limit = keep_limit + self.config.grace_rank_buffer
        if rank > grace_limit:
            return False

        recent_return = self._recent_return(symbol, context, self.config.grace_momentum_window)
        if recent_return is None:
            return False
        return recent_return >= self.config.grace_min_return

    @staticmethod
    def _recent_return(symbol: str, context: StrategyContext, window: int) -> float | None:
        if window <= 0:
            return 0.0
        history = context.history(symbol)
        if len(history) <= window:
            return None
        start_close = history[-(window + 1)].close
        end_close = history[-1].close
        if start_close <= 0:
            return None
        return end_close / start_close - 1.0

    def _hit_trailing_stop(self, symbol: str, context: StrategyContext) -> bool:
        position = context.positions.get(symbol)
        if position is None or position.cost_basis <= 0:
            return False
        unrealized_gain = position.last_price / position.cost_basis - 1.0
        if unrealized_gain < self.config.trailing_stop_min_gain:
            return False

        history = context.history(symbol)
        if not history:
            return False
        window = max(1, self.config.trailing_stop_window)
        recent_closes = [bar.close for bar in history[-window:]]
        peak_close = max(recent_closes)
        if peak_close <= 0:
            return False
        drawdown = position.last_price / peak_close - 1.0
        return drawdown <= -abs(self.config.trailing_stop_drawdown)

    def _hit_score_reversal(self, symbol: str, score_date: str) -> bool:
        confirm_days = max(1, self.config.score_reversal_confirm_days)
        score_dates = sorted(self._scores_by_date)
        if score_date not in score_dates:
            return False
        end_index = score_dates.index(score_date)
        if end_index + 1 < confirm_days:
            return False
        recent_dates = score_dates[end_index + 1 - confirm_days : end_index + 1]
        for candidate_date in recent_dates:
            score_value = self._score_value_by_date.get(candidate_date, {}).get(symbol)
            if score_value is None or score_value > self.config.score_reversal_threshold:
                return False
        return True

    def _hit_score_price_hybrid(self, symbol: str, context: StrategyContext, score_date: str) -> bool:
        if not self._hit_score_reversal(symbol, score_date):
            return False
        recent_return = self._recent_return(symbol, context, self.config.hybrid_price_window)
        if recent_return is None:
            return False
        return recent_return <= self.config.hybrid_price_threshold

    def _should_extend_strong_position(
        self,
        rank: int,
        keep_limit: int,
        symbol: str,
        context: StrategyContext,
    ) -> bool:
        if self.config.strong_keep_extra_buffer <= 0:
            return False
        extension_limit = keep_limit + self.config.strong_keep_extra_buffer
        if rank > extension_limit:
            return False
        recent_return = self._recent_return(symbol, context, self.config.strong_keep_momentum_window)
        if recent_return is None:
            return False
        return recent_return >= self.config.strong_keep_min_return

    def _build_target_weights(
        self,
        context: StrategyContext,
        selected_symbols: list[str],
    ) -> dict[str, float]:
        if not selected_symbols:
            return {}
        base_weight = 1 / len(selected_symbols)
        portfolio_value = context.cash + sum(
            position.quantity * position.last_price for position in context.positions.values()
        )
        if portfolio_value <= 0 or self.config.strong_trim_slowdown <= 0:
            return {symbol: round(base_weight, 6) for symbol in selected_symbols}

        slowdown = min(max(self.config.strong_trim_slowdown, 0.0), 1.0)
        raw_weights: dict[str, float] = {}
        for symbol in selected_symbols:
            target_weight = base_weight
            position = context.positions.get(symbol)
            if position is not None and position.last_price > 0:
                current_weight = (position.quantity * position.last_price) / portfolio_value
                recent_return = self._recent_return(symbol, context, self.config.strong_trim_momentum_window)
                if (
                    recent_return is not None
                    and recent_return >= self.config.strong_trim_min_return
                    and current_weight > target_weight
                ):
                    target_weight = target_weight + slowdown * (current_weight - target_weight)
            raw_weights[symbol] = target_weight

        total_weight = sum(raw_weights.values())
        if total_weight <= 0:
            return self._apply_position_weight_cap({symbol: round(base_weight, 6) for symbol in selected_symbols})
        normalized = {
            symbol: round(weight / total_weight, 6)
            for symbol, weight in raw_weights.items()
        }
        return self._apply_position_weight_cap(normalized)

    def _apply_position_weight_cap(self, target_weights: dict[str, float]) -> dict[str, float]:
        positive = {symbol: float(weight) for symbol, weight in target_weights.items() if weight > 0}
        if not positive:
            return {}
        if self.config.max_position_weight <= 0:
            return positive

        cap = min(max(self.config.max_position_weight, 0.0), 1.0)
        working = dict(positive)
        for _ in range(len(working) * 2):
            over = {symbol: weight for symbol, weight in working.items() if weight > cap}
            if not over:
                break
            excess = sum(weight - cap for weight in over.values())
            for symbol in over:
                working[symbol] = cap
            under = {symbol: weight for symbol, weight in working.items() if weight < cap}
            if excess <= 0 or not under:
                break
            under_total = sum(under.values())
            if under_total <= 0:
                break
            for symbol, weight in under.items():
                working[symbol] = weight + excess * (weight / under_total)

        total = sum(working.values())
        if total <= 0:
            return {}
        return {
            symbol: round(weight / total, 6)
            for symbol, weight in working.items()
            if weight > 0
        }

    @staticmethod
    def _trading_day_distance(history: list, last_rebalance_date: date) -> int:
        trade_dates = [bar.trade_date for bar in history]
        if last_rebalance_date not in trade_dates:
            return len(trade_dates)
        return trade_dates.index(trade_dates[-1]) - trade_dates.index(last_rebalance_date)
