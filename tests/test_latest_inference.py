from __future__ import annotations

import json

import pandas as pd

from ashare_backtest.research.trainer import WalkForwardAsOfDateConfig, train_lightgbm_walk_forward_as_of_date


def test_train_lightgbm_walk_forward_as_of_date_scores_single_date(tmp_path) -> None:
    factor_panel_path = tmp_path / "factor_panel.parquet"
    output_scores_path = tmp_path / "walk_forward_scores_2025-01-06.parquet"
    output_metrics_path = tmp_path / "walk_forward_metrics_2025-01-06.json"

    rows = [
        {
            "trade_date": "2025-01-02",
            "symbol": "AAA",
            "mom_5": 0.1,
            "mom_10": 0.1,
            "mom_20": 0.1,
            "mom_60": 0.1,
            "ma_gap_5": 0.1,
            "ma_gap_10": 0.1,
            "ma_gap_20": 0.1,
            "ma_gap_60": 0.1,
            "volatility_10": 0.1,
            "volatility_20": 0.1,
            "volatility_60": 0.1,
            "range_ratio_5": 0.1,
            "volume_ratio_5_20": 0.1,
            "amount_ratio_5_20": 0.1,
            "amount_mom_10": 0.1,
            "price_pos_20": 0.1,
            "volatility_ratio_10_60": 0.1,
            "trend_strength_20": 0.1,
            "cross_rank_mom_20": 0.1,
            "cross_rank_amount_ratio_5_20": 0.1,
            "cross_rank_volatility_20": 0.1,
            "fwd_return_5": 0.01,
        },
        {
            "trade_date": "2025-01-02",
            "symbol": "BBB",
            "mom_5": 0.2,
            "mom_10": 0.2,
            "mom_20": 0.2,
            "mom_60": 0.2,
            "ma_gap_5": 0.2,
            "ma_gap_10": 0.2,
            "ma_gap_20": 0.2,
            "ma_gap_60": 0.2,
            "volatility_10": 0.2,
            "volatility_20": 0.2,
            "volatility_60": 0.2,
            "range_ratio_5": 0.2,
            "volume_ratio_5_20": 0.2,
            "amount_ratio_5_20": 0.2,
            "amount_mom_10": 0.2,
            "price_pos_20": 0.2,
            "volatility_ratio_10_60": 0.2,
            "trend_strength_20": 0.2,
            "cross_rank_mom_20": 0.2,
            "cross_rank_amount_ratio_5_20": 0.2,
            "cross_rank_volatility_20": 0.2,
            "fwd_return_5": 0.02,
        },
        {
            "trade_date": "2025-01-03",
            "symbol": "AAA",
            "mom_5": 0.15,
            "mom_10": 0.15,
            "mom_20": 0.15,
            "mom_60": 0.15,
            "ma_gap_5": 0.15,
            "ma_gap_10": 0.15,
            "ma_gap_20": 0.15,
            "ma_gap_60": 0.15,
            "volatility_10": 0.15,
            "volatility_20": 0.15,
            "volatility_60": 0.15,
            "range_ratio_5": 0.15,
            "volume_ratio_5_20": 0.15,
            "amount_ratio_5_20": 0.15,
            "amount_mom_10": 0.15,
            "price_pos_20": 0.15,
            "volatility_ratio_10_60": 0.15,
            "trend_strength_20": 0.15,
            "cross_rank_mom_20": 0.15,
            "cross_rank_amount_ratio_5_20": 0.15,
            "cross_rank_volatility_20": 0.15,
            "fwd_return_5": 0.015,
        },
        {
            "trade_date": "2025-01-03",
            "symbol": "BBB",
            "mom_5": 0.25,
            "mom_10": 0.25,
            "mom_20": 0.25,
            "mom_60": 0.25,
            "ma_gap_5": 0.25,
            "ma_gap_10": 0.25,
            "ma_gap_20": 0.25,
            "ma_gap_60": 0.25,
            "volatility_10": 0.25,
            "volatility_20": 0.25,
            "volatility_60": 0.25,
            "range_ratio_5": 0.25,
            "volume_ratio_5_20": 0.25,
            "amount_ratio_5_20": 0.25,
            "amount_mom_10": 0.25,
            "price_pos_20": 0.25,
            "volatility_ratio_10_60": 0.25,
            "trend_strength_20": 0.25,
            "cross_rank_mom_20": 0.25,
            "cross_rank_amount_ratio_5_20": 0.25,
            "cross_rank_volatility_20": 0.25,
            "fwd_return_5": 0.025,
        },
        {
            "trade_date": "2025-01-06",
            "symbol": "AAA",
            "mom_5": 0.18,
            "mom_10": 0.18,
            "mom_20": 0.18,
            "mom_60": 0.18,
            "ma_gap_5": 0.18,
            "ma_gap_10": 0.18,
            "ma_gap_20": 0.18,
            "ma_gap_60": 0.18,
            "volatility_10": 0.18,
            "volatility_20": 0.18,
            "volatility_60": 0.18,
            "range_ratio_5": 0.18,
            "volume_ratio_5_20": 0.18,
            "amount_ratio_5_20": 0.18,
            "amount_mom_10": 0.18,
            "price_pos_20": 0.18,
            "volatility_ratio_10_60": 0.18,
            "trend_strength_20": 0.18,
            "cross_rank_mom_20": 0.18,
            "cross_rank_amount_ratio_5_20": 0.18,
            "cross_rank_volatility_20": 0.18,
            "fwd_return_5": None,
        },
        {
            "trade_date": "2025-01-06",
            "symbol": "BBB",
            "mom_5": 0.28,
            "mom_10": 0.28,
            "mom_20": 0.28,
            "mom_60": 0.28,
            "ma_gap_5": 0.28,
            "ma_gap_10": 0.28,
            "ma_gap_20": 0.28,
            "ma_gap_60": 0.28,
            "volatility_10": 0.28,
            "volatility_20": 0.28,
            "volatility_60": 0.28,
            "range_ratio_5": 0.28,
            "volume_ratio_5_20": 0.28,
            "amount_ratio_5_20": 0.28,
            "amount_mom_10": 0.28,
            "price_pos_20": 0.28,
            "volatility_ratio_10_60": 0.28,
            "trend_strength_20": 0.28,
            "cross_rank_mom_20": 0.28,
            "cross_rank_amount_ratio_5_20": 0.28,
            "cross_rank_volatility_20": 0.28,
            "fwd_return_5": None,
        },
    ]
    pd.DataFrame(rows).to_parquet(factor_panel_path, index=False)

    metrics = train_lightgbm_walk_forward_as_of_date(
        WalkForwardAsOfDateConfig(
            factor_panel_path=factor_panel_path.as_posix(),
            output_scores_path=output_scores_path.as_posix(),
            output_metrics_path=output_metrics_path.as_posix(),
            as_of_date="2025-01-06",
            train_window_months=12,
            n_estimators=20,
        )
    )

    scored = pd.read_parquet(output_scores_path)
    assert metrics["as_of_date"] == "2025-01-06"
    assert metrics["scored_rows"] == 2
    assert set(scored["symbol"]) == {"AAA", "BBB"}
    assert set(pd.to_datetime(scored["trade_date"]).dt.date.astype(str)) == {"2025-01-06"}
    assert scored["prediction"].notna().all()

    persisted_metrics = json.loads(output_metrics_path.read_text(encoding="utf-8"))
    assert persisted_metrics["train_rows"] == 4
