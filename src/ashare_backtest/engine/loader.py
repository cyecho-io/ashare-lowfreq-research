from __future__ import annotations

import importlib.util
from pathlib import Path

from ashare_backtest.protocol import BaseStrategy
from ashare_backtest.sandbox import StrategyValidator


def load_strategy(path: str | Path) -> BaseStrategy:
    validator = StrategyValidator()
    report = validator.validate_file(path)
    target = Path(path)
    spec = importlib.util.spec_from_file_location(target.stem, target)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load strategy: {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategy_class = getattr(module, report.class_name)
    strategy = strategy_class()
    if not isinstance(strategy, BaseStrategy):
        raise TypeError("loaded class is not a BaseStrategy instance")
    return strategy
