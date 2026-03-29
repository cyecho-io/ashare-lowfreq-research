from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from ashare_backtest.data import InMemoryDataProvider
from ashare_backtest.protocol import Bar
from ashare_backtest.research.analysis import StartDateRobustnessConfig, analyze_start_date_robustness


def test_analyze_start_date_robustness_outputs_ranked_windows(tmp_path) -> None:
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-02", "symbol": "AAA", "prediction": 1.0},
            {"trade_date": "2025-02-03", "symbol": "AAA", "prediction": 1.0},
            {"trade_date": "2025-03-03", "symbol": "AAA", "prediction": 1.0},
            {"trade_date": "2025-04-01", "symbol": "AAA", "prediction": 1.0},
            {"trade_date": "2025-05-01", "symbol": "AAA", "prediction": 1.0},
        ]
    ).to_parquet(scores_path, index=False)

    instruments_path = tmp_path / "storage" / "parquet" / "instruments" / "ashare_instruments.parquet"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "industry_level_1": "银行", "is_st": False},
        ]
    ).to_parquet(instruments_path, index=False)

    provider = InMemoryDataProvider(
        {
            "AAA": [
                Bar("AAA", date(2025, 1, 2), 10.0, 10.0, 10.0, 10.0, amount=1_000_000.0),
                Bar("AAA", date(2025, 2, 3), 12.0, 12.0, 12.0, 12.0, amount=1_000_000.0),
                Bar("AAA", date(2025, 3, 3), 14.0, 14.0, 14.0, 14.0, amount=1_000_000.0),
                Bar("AAA", date(2025, 4, 1), 9.0, 9.0, 9.0, 9.0, amount=1_000_000.0),
                Bar("AAA", date(2025, 5, 1), 8.0, 8.0, 8.0, 8.0, amount=1_000_000.0),
            ]
        }
    )

    output_path = tmp_path / "robustness.json"
    payload = analyze_start_date_robustness(
            StartDateRobustnessConfig(
                scores_path=scores_path.as_posix(),
                storage_root=(tmp_path / "storage").as_posix(),
                output_path=output_path.as_posix(),
                analysis_start_date="2025-01-02",
                analysis_end_date="2025-05-01",
                holding_months=1,
                cadence="monthly",
                top_k=1,
                rebalance_every=1,
                lookback_window=1,
            min_hold_bars=1,
                keep_buffer=0,
                min_turnover_names=1,
                initial_cash=100_000.0,
                commission_rate=0.0,
                stamp_tax_rate=0.0,
                slippage_rate=0.0,
            ),
            provider=provider,
        )

    assert payload["summary"]["sample_count"] == 4
    assert payload["summary"]["best_start_date"] == "2025-01-02"
    assert payload["summary"]["worst_start_date"] == "2025-03-03"
    assert payload["summary"]["best_total_return"] > 0
    assert payload["summary"]["worst_total_return"] < 0
    assert payload["by_start_date"][0]["start_date"] == "2025-01-02"
    assert payload["by_start_date"][0]["end_date"] == "2025-02-03"
    assert payload["by_start_date"][0]["total_return"] == pytest.approx(0.2, rel=1e-4)
    assert payload["by_start_date"][1]["total_return"] == pytest.approx(0.166, rel=1e-4)
    assert payload["by_start_date"][2]["total_return"] == pytest.approx(-0.426, rel=1e-4)
    assert payload["by_start_date"][3]["total_return"] == pytest.approx(-0.111, rel=1e-4)

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["summary"]["median_total_return"] == pytest.approx(0.0275, rel=1e-4)
