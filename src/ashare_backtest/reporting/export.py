from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from ashare_backtest.protocol import BacktestResult


def export_backtest_result(result: BacktestResult, output_dir: str | Path) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    summary = {
        "total_return": result.total_return,
        "annual_return": result.annual_return,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "win_rate": result.win_rate,
        "profit_loss_ratio": result.profit_loss_ratio,
        "turnover_ratio": result.turnover_ratio,
        "trade_count": len(result.trades),
        "filled_trade_count": result.filled_trade_count,
        "rejected_trade_count": result.rejected_trade_count,
        "equity_points": len(result.equity_curve),
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (root / "equity_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trade_date", "equity"])
        for trade_date, equity in result.equity_curve:
            writer.writerow([trade_date.isoformat(), f"{equity:.6f}"])

    with (root / "trades.csv").open("w", encoding="utf-8", newline="") as handle:
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
        for trade in result.trades:
            row = asdict(trade)
            row["trade_date"] = trade.trade_date.isoformat()
            writer.writerow(row)
