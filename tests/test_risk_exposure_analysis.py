from __future__ import annotations

import csv
import json

import pandas as pd

from ashare_backtest.research.analysis import RiskExposureConfig, analyze_monthly_risk_exposures


def test_analyze_monthly_risk_exposures_outputs_monthly_metrics(tmp_path) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    with (result_dir / "trades.csv").open("w", encoding="utf-8", newline="") as handle:
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
                "trade_date": "2025-01-02",
                "symbol": "AAA",
                "side": "BUY",
                "quantity": 100,
                "price": 10.0,
                "amount": 1000.0,
                "commission": 0.0,
                "tax": 0.0,
                "slippage": 0.0,
                "status": "filled",
                "reason": "test",
            }
        )
        writer.writerow(
            {
                "trade_date": "2025-01-02",
                "symbol": "BBB",
                "side": "BUY",
                "quantity": 100,
                "price": 20.0,
                "amount": 2000.0,
                "commission": 0.0,
                "tax": 0.0,
                "slippage": 0.0,
                "status": "filled",
                "reason": "test",
            }
        )
    (result_dir / "equity_curve.csv").write_text(
        "\n".join(
            [
                "trade_date,equity",
                "2025-01-02,3000",
                "2025-01-03,3300",
            ]
        ),
        encoding="utf-8",
    )

    storage_root = tmp_path / "storage"
    bars_path = storage_root / "parquet" / "bars" / "daily.parquet"
    bars_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"trade_date": "2025-01-02", "symbol": "AAA", "close": 10.0, "close_adj": 10.0, "amount": 10000.0, "turnover_rate": 1.0},
            {"trade_date": "2025-01-02", "symbol": "BBB", "close": 20.0, "close_adj": 20.0, "amount": 20000.0, "turnover_rate": 2.0},
            {"trade_date": "2025-01-03", "symbol": "AAA", "close": 11.0, "close_adj": 11.0, "amount": 11000.0, "turnover_rate": 1.1},
            {"trade_date": "2025-01-03", "symbol": "BBB", "close": 22.0, "close_adj": 22.0, "amount": 21000.0, "turnover_rate": 2.1},
        ]
    ).to_parquet(bars_path, index=False)

    instruments_path = storage_root / "parquet" / "instruments" / "ashare_instruments.parquet"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "industry_level_1": "银行"},
            {"symbol": "BBB", "industry_level_1": "软件服务"},
        ]
    ).to_parquet(instruments_path, index=False)

    output_path = tmp_path / "risk.json"
    payload = analyze_monthly_risk_exposures(
        RiskExposureConfig(
            result_dir=result_dir.as_posix(),
            storage_root=storage_root.as_posix(),
            output_path=output_path.as_posix(),
            top_industries=2,
            volatility_window=2,
        )
    )

    assert payload["summary"]["monthly_count"] == 1
    month = payload["by_month"][0]
    assert month["trade_month"] == "2025-01"
    assert month["avg_name_count"] == 2.0
    assert month["avg_top_position_weight"] > 0.5
    assert month["top_industries"][0]["industry"] == "软件服务"

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["by_month"][0]["avg_weighted_amount"] > 0
