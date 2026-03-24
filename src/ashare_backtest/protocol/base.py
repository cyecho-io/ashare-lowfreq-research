from __future__ import annotations

from abc import ABC, abstractmethod

from .models import AllocationDecision, RebalanceDecision, StrategyContext, StrategyMetadata


class BaseStrategy(ABC):
    """Restricted strategy interface owned by the backtest engine."""

    metadata: StrategyMetadata

    @abstractmethod
    def rebalance(self, context: StrategyContext) -> RebalanceDecision:
        """Return whether the engine should rebalance on the current trade date."""

    @abstractmethod
    def select(self, context: StrategyContext) -> list[str]:
        """Return the candidate symbols for the current decision point."""

    @abstractmethod
    def allocate(
        self,
        context: StrategyContext,
        selected_symbols: list[str],
    ) -> AllocationDecision:
        """Return normalized target weights. The engine owns the execution details."""
