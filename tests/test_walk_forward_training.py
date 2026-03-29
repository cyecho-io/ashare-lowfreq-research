from __future__ import annotations

import json

import pandas as pd

from ashare_backtest.research.trainer import (
    WalkForwardConfig,
    WalkForwardSingleDateConfig,
    train_lightgbm_walk_forward,
    train_lightgbm_walk_forward_single_date,
)


def test_train_lightgbm_walk_forward_keeps_unlabeled_latest_test_rows(tmp_path) -> None:
    factor_panel_path = tmp_path / "factor_panel.parquet"
    output_scores_path = tmp_path / "walk_forward_scores.parquet"
    output_metrics_path = tmp_path / "walk_forward_metrics.json"

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
            "trade_date": "2025-02-03",
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
            "fwd_return_5": None,
        },
        {
            "trade_date": "2025-02-03",
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
            "fwd_return_5": None,
        },
    ]
    pd.DataFrame(rows).to_parquet(factor_panel_path, index=False)

    metrics = train_lightgbm_walk_forward(
        WalkForwardConfig(
            factor_panel_path=factor_panel_path.as_posix(),
            output_scores_path=output_scores_path.as_posix(),
            output_metrics_path=output_metrics_path.as_posix(),
            test_start_month="2025-02",
            test_end_month="2025-02",
            train_window_months=12,
            n_estimators=20,
        )
    )

    scored = pd.read_parquet(output_scores_path)
    persisted_metrics = json.loads(output_metrics_path.read_text(encoding="utf-8"))

    assert set(pd.to_datetime(scored["trade_date"]).dt.date.astype(str)) == {"2025-02-03"}
    assert scored["prediction"].notna().all()
    assert scored["label"].isna().all()
    assert metrics["total_scored_rows"] == 2
    assert persisted_metrics["windows"][0]["test_rows"] == 2
    assert persisted_metrics["windows"][0]["eval_rows"] == 0
    assert persisted_metrics["windows"][0]["mae"] == "n/a"


def test_train_lightgbm_walk_forward_single_date_matches_monthly_walk_forward_slice(tmp_path) -> None:
    factor_panel_path = tmp_path / "factor_panel.parquet"
    output_scores_path = tmp_path / "walk_forward_scores.parquet"
    output_metrics_path = tmp_path / "walk_forward_metrics.json"
    single_scores_path = tmp_path / "walk_forward_scores_2025-02-03.parquet"
    single_metrics_path = tmp_path / "walk_forward_metrics_2025-02-03.json"

    rows = [
        {
            "trade_date": trade_date,
            "symbol": symbol,
            "mom_5": base,
            "mom_10": base,
            "mom_20": base,
            "mom_60": base,
            "ma_gap_5": base,
            "ma_gap_10": base,
            "ma_gap_20": base,
            "ma_gap_60": base,
            "volatility_10": base,
            "volatility_20": base,
            "volatility_60": base,
            "range_ratio_5": base,
            "volume_ratio_5_20": base,
            "amount_ratio_5_20": base,
            "amount_mom_10": base,
            "price_pos_20": base,
            "volatility_ratio_10_60": base,
            "trend_strength_20": base,
            "cross_rank_mom_20": base,
            "cross_rank_amount_ratio_5_20": base,
            "cross_rank_volatility_20": base,
            "fwd_return_5": label,
        }
        for trade_date, symbol, base, label in [
            ("2025-01-02", "AAA", 0.1, 0.01),
            ("2025-01-02", "BBB", 0.2, 0.02),
            ("2025-01-03", "AAA", 0.15, 0.015),
            ("2025-01-03", "BBB", 0.25, 0.025),
            ("2025-02-03", "AAA", 0.18, None),
            ("2025-02-03", "BBB", 0.28, None),
        ]
    ]
    pd.DataFrame(rows).to_parquet(factor_panel_path, index=False)

    train_lightgbm_walk_forward(
        WalkForwardConfig(
            factor_panel_path=factor_panel_path.as_posix(),
            output_scores_path=output_scores_path.as_posix(),
            output_metrics_path=output_metrics_path.as_posix(),
            test_start_month="2025-02",
            test_end_month="2025-02",
            train_window_months=12,
            n_estimators=20,
        )
    )
    single_metrics = train_lightgbm_walk_forward_single_date(
        WalkForwardSingleDateConfig(
            factor_panel_path=factor_panel_path.as_posix(),
            output_scores_path=single_scores_path.as_posix(),
            output_metrics_path=single_metrics_path.as_posix(),
            test_month="2025-02",
            as_of_date="2025-02-03",
            train_window_months=12,
            n_estimators=20,
        )
    )

    bulk = pd.read_parquet(output_scores_path)
    single = pd.read_parquet(single_scores_path)
    bulk["trade_date"] = pd.to_datetime(bulk["trade_date"], errors="coerce")
    bulk = bulk.loc[bulk["trade_date"] == pd.Timestamp("2025-02-03"), ["symbol", "prediction"]].sort_values("symbol")
    single = single.loc[:, ["symbol", "prediction"]].sort_values("symbol")

    assert bulk["symbol"].tolist() == single["symbol"].tolist()
    assert bulk["prediction"].tolist() == single["prediction"].tolist()
    assert single_metrics["test_month"] == "2025-02"
    assert single_metrics["as_of_date"] == "2025-02-03"
