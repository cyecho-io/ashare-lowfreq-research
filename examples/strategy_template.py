from ashare_backtest.protocol import (
    AllocationDecision,
    BaseStrategy,
    RebalanceDecision,
    StrategyContext,
    StrategyMetadata,
)


class Strategy(BaseStrategy):
    metadata = StrategyMetadata(
        name="template_strategy",
        description="Example restricted strategy template",
        lookback_window=20,
    )

    def rebalance(self, context: StrategyContext) -> RebalanceDecision:
        if len(context.history(context.universe[0])) < self.metadata.lookback_window:
            return RebalanceDecision(False, "insufficient_history")
        return RebalanceDecision(True, "fixed_schedule")

    def select(self, context: StrategyContext) -> list[str]:
        return list(context.universe[:5])

    def allocate(
        self,
        context: StrategyContext,
        selected_symbols: list[str],
    ) -> AllocationDecision:
        if not selected_symbols:
            return AllocationDecision(target_weights={}, note="no_selection")
        weight = round(1 / len(selected_symbols), 4)
        return AllocationDecision(
            target_weights={symbol: weight for symbol in selected_symbols},
            note="equal_weight",
        )
