from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from ashare_backtest.protocol import BacktestConfig


@dataclass(frozen=True)
class RunConfig:
    backtest: BacktestConfig
    storage_root: str
    output_dir: str


def load_run_config(path: str | Path) -> RunConfig:
    target = Path(path)
    payload = tomllib.loads(target.read_text(encoding="utf-8"))
    backtest_section = payload["backtest"]
    return RunConfig(
        backtest=BacktestConfig(
            strategy_path=backtest_section["strategy_path"],
            start_date=date.fromisoformat(backtest_section["start_date"]),
            end_date=date.fromisoformat(backtest_section["end_date"]),
            universe=tuple(backtest_section["universe"]),
            initial_cash=float(backtest_section.get("initial_cash", 1_000_000.0)),
            commission_rate=float(backtest_section.get("commission_rate", 0.0003)),
            stamp_tax_rate=float(backtest_section.get("stamp_tax_rate", 0.001)),
            slippage_rate=float(backtest_section.get("slippage_rate", 0.0005)),
            rebalance_price=str(backtest_section.get("rebalance_price", "open")),
            max_trade_participation_rate=float(backtest_section.get("max_trade_participation_rate", 0.0)),
            max_pending_days=int(backtest_section.get("max_pending_days", 0)),
        ),
        storage_root=str(payload.get("storage", {}).get("root", "storage")),
        output_dir=str(payload.get("output", {}).get("dir", "results/latest")),
    )
