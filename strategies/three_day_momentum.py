from ashare_backtest.protocol import (
    AllocationDecision,
    BaseStrategy,
    RebalanceDecision,
    StrategyContext,
    StrategyMetadata,
)


class ThreeDayMomentumStrategy(BaseStrategy):
    metadata = StrategyMetadata(
        name="three_day_momentum",
        description="Rebalance every 3 trading days and hold the top 2 names by 20-day momentum",
        lookback_window=20,
    )

    def rebalance(self, context: StrategyContext) -> RebalanceDecision:
        if not context.universe:
            return RebalanceDecision(False, "empty_universe")
        sample_history = context.history(context.universe[0])
        if len(sample_history) < self.metadata.lookback_window:
            return RebalanceDecision(False, "insufficient_history")
        offset = len(sample_history) - self.metadata.lookback_window
        if offset % 3 != 0:
            return RebalanceDecision(False, "three_day_schedule")
        return RebalanceDecision(True, "scheduled_rebalance")

    def select(self, context: StrategyContext) -> list[str]:
        scored: list[tuple[float, str]] = []
        for symbol in context.universe:
            history = context.history(symbol)
            if len(history) < self.metadata.lookback_window:
                continue
            start_close = history[0].close
            end_close = history[-1].close
            if start_close <= 0:
                continue
            momentum = end_close / start_close - 1
            scored.append((momentum, symbol))
        scored.sort(reverse=True)
        return [symbol for _, symbol in scored[:2]]

    def allocate(
        self,
        context: StrategyContext,
        selected_symbols: list[str],
    ) -> AllocationDecision:
        if not selected_symbols:
            return AllocationDecision(target_weights={}, note="no_valid_selection")
        weight = round(1 / len(selected_symbols), 4)
        return AllocationDecision(
            target_weights={symbol: weight for symbol in selected_symbols},
            note="equal_weight_top_momentum",
        )
