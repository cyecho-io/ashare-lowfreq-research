from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    "mom_5",
    "mom_10",
    "mom_20",
    "mom_60",
    "ma_gap_5",
    "ma_gap_10",
    "ma_gap_20",
    "ma_gap_60",
    "volatility_10",
    "volatility_20",
    "volatility_60",
    "range_ratio_5",
    "volume_ratio_5_20",
    "amount_ratio_5_20",
    "amount_mom_10",
    "price_pos_20",
    "volatility_ratio_10_60",
    "trend_strength_20",
    "cross_rank_mom_20",
    "cross_rank_amount_ratio_5_20",
    "cross_rank_volatility_20",
]


@dataclass(frozen=True)
class ModelTrainConfig:
    factor_panel_path: str
    output_scores_path: str
    output_metrics_path: str
    label_column: str = "fwd_return_5"
    train_end_date: str = "2024-09-30"
    test_start_date: str = "2024-10-01"
    test_end_date: str = "2024-12-31"
    feature_columns: tuple[str, ...] = tuple(DEFAULT_FEATURE_COLUMNS)
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_data_in_leaf: int = 20


@dataclass(frozen=True)
class WalkForwardConfig:
    factor_panel_path: str
    output_scores_path: str
    output_metrics_path: str
    label_column: str = "fwd_return_5"
    feature_columns: tuple[str, ...] = tuple(DEFAULT_FEATURE_COLUMNS)
    train_window_months: int = 12
    test_start_month: str = "2025-07"
    test_end_month: str = "2026-02"
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_data_in_leaf: int = 20


def train_lightgbm_model(config: ModelTrainConfig) -> dict[str, float | int | str]:
    import lightgbm as lgb

    frame = pd.read_parquet(config.factor_panel_path).sort_values(["trade_date", "symbol"])
    frame = frame.dropna(subset=list(config.feature_columns) + [config.label_column]).copy()

    train_mask = frame["trade_date"] <= pd.Timestamp(config.train_end_date)
    test_mask = (
        (frame["trade_date"] >= pd.Timestamp(config.test_start_date))
        & (frame["trade_date"] <= pd.Timestamp(config.test_end_date))
    )

    train_frame = frame.loc[train_mask].copy()
    test_frame = frame.loc[test_mask].copy()
    if train_frame.empty or test_frame.empty:
        raise ValueError("train/test split produced an empty dataset")

    x_train = train_frame.loc[:, list(config.feature_columns)]
    y_train = train_frame[config.label_column]
    x_test = test_frame.loc[:, list(config.feature_columns)]
    y_test = test_frame[config.label_column]

    model = lgb.LGBMRegressor(
        objective="regression",
        num_leaves=config.num_leaves,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        min_data_in_leaf=config.min_data_in_leaf,
        random_state=42,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    scored = test_frame.loc[:, ["trade_date", "symbol", config.label_column]].copy()
    scored["prediction"] = predictions
    scored = scored.rename(columns={config.label_column: "label"})

    scores_path = Path(config.output_scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(scores_path, index=False)

    mae = float((scored["prediction"] - scored["label"]).abs().mean())
    rmse = float((((scored["prediction"] - scored["label"]) ** 2).mean()) ** 0.5)
    ic = float(scored[["prediction", "label"]].corr(method="spearman").iloc[0, 1])
    metrics = {
        "label_column": config.label_column,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "feature_count": int(len(config.feature_columns)),
        "mae": mae,
        "rmse": rmse,
        "spearman_ic": ic,
        "train_end_date": config.train_end_date,
        "test_start_date": config.test_start_date,
        "test_end_date": config.test_end_date,
    }

    metrics_path = Path(config.output_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def train_lightgbm_walk_forward(config: WalkForwardConfig) -> dict[str, float | int | str]:
    import lightgbm as lgb

    frame = pd.read_parquet(config.factor_panel_path).sort_values(["trade_date", "symbol"])
    frame = frame.dropna(subset=list(config.feature_columns) + [config.label_column]).copy()
    frame["month"] = frame["trade_date"].dt.to_period("M")

    all_months = sorted(frame["month"].unique().tolist())
    start_period = pd.Period(config.test_start_month, freq="M")
    end_period = pd.Period(config.test_end_month, freq="M")
    test_months = [month for month in all_months if start_period <= month <= end_period]
    if not test_months:
        raise ValueError("walk-forward test month range produced no periods")

    scored_parts: list[pd.DataFrame] = []
    window_metrics: list[dict[str, float | int | str]] = []

    for test_month in test_months:
        month_index = all_months.index(test_month)
        train_end_index = month_index - 1
        train_start_index = max(0, train_end_index - config.train_window_months + 1)
        if train_end_index < 0:
            continue
        train_months = all_months[train_start_index : train_end_index + 1]
        train_frame = frame.loc[frame["month"].isin(train_months)].copy()
        test_frame = frame.loc[frame["month"] == test_month].copy()
        if train_frame.empty or test_frame.empty:
            continue

        model = lgb.LGBMRegressor(
            objective="regression",
            num_leaves=config.num_leaves,
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            min_data_in_leaf=config.min_data_in_leaf,
            random_state=42,
            verbose=-1,
        )
        model.fit(train_frame.loc[:, list(config.feature_columns)], train_frame[config.label_column])
        predictions = model.predict(test_frame.loc[:, list(config.feature_columns)])

        scored = test_frame.loc[:, ["trade_date", "symbol", config.label_column]].copy()
        scored["prediction"] = predictions
        scored["train_end_month"] = str(train_months[-1])
        scored["test_month"] = str(test_month)
        scored = scored.rename(columns={config.label_column: "label"})
        scored_parts.append(scored)

        mae = float((scored["prediction"] - scored["label"]).abs().mean())
        rmse = float((((scored["prediction"] - scored["label"]) ** 2).mean()) ** 0.5)
        ic = float(scored[["prediction", "label"]].corr(method="spearman").iloc[0, 1])
        window_metrics.append(
            {
                "test_month": str(test_month),
                "train_start_month": str(train_months[0]),
                "train_end_month": str(train_months[-1]),
                "train_rows": int(len(train_frame)),
                "test_rows": int(len(test_frame)),
                "mae": mae,
                "rmse": rmse,
                "spearman_ic": ic,
            }
        )

    if not scored_parts:
        raise ValueError("walk-forward training produced no scored windows")

    all_scored = pd.concat(scored_parts, ignore_index=True).sort_values(["trade_date", "symbol"])
    scores_path = Path(config.output_scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    all_scored.to_parquet(scores_path, index=False)

    mean_mae = float(pd.Series([item["mae"] for item in window_metrics]).mean())
    mean_rmse = float(pd.Series([item["rmse"] for item in window_metrics]).mean())
    mean_ic = float(pd.Series([item["spearman_ic"] for item in window_metrics]).mean())
    metrics = {
        "label_column": config.label_column,
        "feature_count": int(len(config.feature_columns)),
        "train_window_months": config.train_window_months,
        "test_start_month": config.test_start_month,
        "test_end_month": config.test_end_month,
        "window_count": len(window_metrics),
        "total_scored_rows": int(len(all_scored)),
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "mean_spearman_ic": mean_ic,
        "windows": window_metrics,
    }

    metrics_path = Path(config.output_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics
