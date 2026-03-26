from __future__ import annotations

import csv
import json

import pandas as pd

from ashare_backtest.research.analysis import CapacityAnalysisConfig, analyze_trade_capacity


def test_analyze_trade_capacity_scales_participation_and_breaches(tmp_path) -> None:
    trades_path = tmp_path / "trades.csv"
    with trades_path.open("w", encoding="utf-8", newline="") as handle:
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
        writer.writerow(
            {
                "trade_date": "2025-01-06",
                "symbol": "AAA",
                "side": "buy",
                "quantity": 100,
                "price": 10.0,
                "amount": 1_000.0,
                "commission": 0.0,
                "tax": 0.0,
                "slippage": 0.0,
                "status": "filled",
                "reason": "",
            }
        )
        writer.writerow(
            {
                "trade_date": "2025-01-06",
                "symbol": "BBB",
                "side": "buy",
                "quantity": 100,
                "price": 5.0,
                "amount": 500.0,
                "commission": 0.0,
                "tax": 0.0,
                "slippage": 0.0,
                "status": "rejected",
                "reason": "paused",
            }
        )

    bars_path = tmp_path / "storage" / "parquet" / "bars" / "daily.parquet"
    bars_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"trade_date": "2025-01-06", "symbol": "AAA", "amount": 10_000.0},
            {"trade_date": "2025-01-06", "symbol": "BBB", "amount": 8_000.0},
        ]
    ).to_parquet(bars_path, index=False)

    output_path = tmp_path / "capacity.json"
    payload = analyze_trade_capacity(
        CapacityAnalysisConfig(
            trades_path=trades_path.as_posix(),
            storage_root=(tmp_path / "storage").as_posix(),
            output_path=output_path.as_posix(),
            base_capital=100_000.0,
            scale_capitals=(100_000.0, 500_000.0),
            participation_thresholds=(0.05, 0.20),
            top_trade_count=5,
        )
    )

    assert payload["summary"]["filled_trade_count"] == 1
    assert payload["summary"]["matched_positive_market_amount_count"] == 1

    base_scale = payload["by_scale"][0]
    assert base_scale["capital"] == 100_000.0
    assert base_scale["participation_max"] == 0.1
    assert base_scale["threshold_breach_ratio"]["5.00%"] == 1.0
    assert base_scale["threshold_breach_ratio"]["20.00%"] == 0.0

    larger_scale = payload["by_scale"][1]
    assert larger_scale["capital"] == 500_000.0
    assert larger_scale["participation_max"] == 0.5
    assert larger_scale["threshold_breach_ratio"]["20.00%"] == 1.0
    assert larger_scale["top_stressed_trades"][0]["symbol"] == "AAA"
    assert larger_scale["top_stressed_symbols"][0]["symbol"] == "AAA"
    assert larger_scale["top_stressed_months"][0]["trade_month"] == "2025-01"

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["by_scale"][1]["participation_max"] == 0.5
