from ashare_backtest.protocol import (
    AllocationDecision,
    BaseStrategy,
    RebalanceDecision,
    StrategyContext,
    StrategyMetadata,
)


class BuyAndHoldStrategy(BaseStrategy):
    metadata = StrategyMetadata(
        name="buy_and_hold",
        description="Minimal placeholder strategy for protocol validation",
        lookback_window=5,
    )

    def rebalance(self, context: StrategyContext) -> RebalanceDecision:
        has_position = bool(context.positions)
        if has_position:
            return RebalanceDecision(False, "already_allocated")
        enough_history = all(len(context.history(symbol)) >= self.metadata.lookback_window for symbol in context.universe)
        if not enough_history:
            return RebalanceDecision(False, "insufficient_history")
        return RebalanceDecision(True, "first_entry")

    def select(self, context: StrategyContext) -> list[str]:
        return list(context.universe[:1])

    def allocate(
        self,
        context: StrategyContext,
        selected_symbols: list[str],
    ) -> AllocationDecision:
        if not selected_symbols:
            return AllocationDecision(target_weights={}, note="empty_universe")
        return AllocationDecision(target_weights={selected_symbols[0]: 1.0}, note="single_name_full_weight")
