from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ashare_backtest.logging_utils import get_logger


LOGGER = get_logger("research.trainer")


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


@dataclass(frozen=True)
class LatestInferenceConfig:
    factor_panel_path: str
    output_scores_path: str
    output_metrics_path: str
    label_column: str = "fwd_return_5"
    inference_date: str | None = None
    train_window_months: int = 12
    feature_columns: tuple[str, ...] = tuple(DEFAULT_FEATURE_COLUMNS)
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_data_in_leaf: int = 20


@dataclass(frozen=True)
class WalkForwardAsOfDateConfig:
    factor_panel_path: str
    output_scores_path: str
    output_metrics_path: str
    label_column: str = "fwd_return_5"
    as_of_date: str | None = None
    train_window_months: int = 12
    feature_columns: tuple[str, ...] = tuple(DEFAULT_FEATURE_COLUMNS)
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_data_in_leaf: int = 20


@dataclass(frozen=True)
class WalkForwardSingleDateConfig:
    factor_panel_path: str
    output_scores_path: str
    output_metrics_path: str
    test_month: str
    as_of_date: str
    label_column: str = "fwd_return_5"
    feature_columns: tuple[str, ...] = tuple(DEFAULT_FEATURE_COLUMNS)
    train_window_months: int = 12
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


def train_lightgbm_walk_forward_as_of_date(config: WalkForwardAsOfDateConfig) -> dict[str, float | int | str]:
    import lightgbm as lgb

    frame = pd.read_parquet(config.factor_panel_path).sort_values(["trade_date", "symbol"]).copy()
    if frame.empty:
        raise ValueError("factor panel is empty")

    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    if config.as_of_date is None:
        as_of_date = frame["trade_date"].max()
    else:
        as_of_date = pd.Timestamp(config.as_of_date)

    scoring_frame = frame.loc[frame["trade_date"] == as_of_date].copy()
    if scoring_frame.empty:
        raise ValueError("as-of date produced an empty dataset")
    scoring_frame = scoring_frame.dropna(subset=list(config.feature_columns)).copy()
    if scoring_frame.empty:
        raise ValueError("as-of-date rows have no valid feature values")

    train_frame = frame.loc[frame["trade_date"] < as_of_date].copy()
    train_frame["month"] = train_frame["trade_date"].dt.to_period("M")
    if config.train_window_months > 0:
        train_months = sorted(train_frame["month"].dropna().unique().tolist())
        train_months = train_months[-config.train_window_months :]
        train_frame = train_frame.loc[train_frame["month"].isin(train_months)].copy()
    train_frame = train_frame.dropna(subset=list(config.feature_columns) + [config.label_column]).copy()
    if train_frame.empty:
        raise ValueError("latest inference training set is empty")

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
    predictions = model.predict(scoring_frame.loc[:, list(config.feature_columns)])

    scored = scoring_frame.loc[:, ["trade_date", "symbol"]].copy()
    if config.label_column in scoring_frame.columns:
        scored["label"] = scoring_frame[config.label_column]
    scored["prediction"] = predictions
    scored["train_end_date"] = train_frame["trade_date"].max().date().isoformat()
    scored["as_of_date"] = as_of_date.date().isoformat()

    scores_path = Path(config.output_scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(scores_path, index=False)

    metrics = {
        "label_column": config.label_column,
        "feature_count": int(len(config.feature_columns)),
        "train_window_months": config.train_window_months,
        "train_rows": int(len(train_frame)),
        "train_start_date": train_frame["trade_date"].min().date().isoformat(),
        "train_end_date": train_frame["trade_date"].max().date().isoformat(),
        "as_of_date": as_of_date.date().isoformat(),
        "scored_rows": int(len(scored)),
        "scored_symbol_count": int(scored["symbol"].nunique()),
    }

    metrics_path = Path(config.output_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def train_lightgbm_walk_forward_single_date(config: WalkForwardSingleDateConfig) -> dict[str, float | int | str]:
    import lightgbm as lgb

    LOGGER.info(
        "single-date score start factor_panel=%s as_of_date=%s test_month=%s output_scores=%s",
        config.factor_panel_path,
        config.as_of_date,
        config.test_month,
        config.output_scores_path,
    )
    frame = pd.read_parquet(config.factor_panel_path).sort_values(["trade_date", "symbol"]).copy()
    if frame.empty:
        raise ValueError("factor panel is empty")

    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).copy()
    frame = frame.dropna(subset=list(config.feature_columns)).copy()
    frame["month"] = frame["trade_date"].dt.to_period("M")

    as_of_date = pd.Timestamp(config.as_of_date)
    test_month = pd.Period(config.test_month, freq="M")
    if as_of_date.to_period("M") != test_month:
        raise ValueError("as_of_date must fall within test_month")

    all_months = sorted(frame["month"].dropna().unique().tolist())
    if test_month not in all_months:
        raise ValueError("test_month produced no data in factor panel")
    month_index = all_months.index(test_month)
    train_end_index = month_index - 1
    if train_end_index < 0:
        raise ValueError("walk-forward single-date training has no prior months to train on")
    train_start_index = max(0, train_end_index - config.train_window_months + 1)
    train_months = all_months[train_start_index : train_end_index + 1]

    train_frame = frame.loc[frame["month"].isin(train_months)].copy()
    train_frame = train_frame.dropna(subset=list(config.feature_columns) + [config.label_column]).copy()
    scoring_frame = frame.loc[frame["trade_date"] == as_of_date].copy()
    if train_frame.empty:
        raise ValueError("walk-forward single-date training set is empty")
    if scoring_frame.empty:
        raise ValueError("as-of date produced an empty dataset")

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
    predictions = model.predict(scoring_frame.loc[:, list(config.feature_columns)])

    scored = scoring_frame.loc[:, ["trade_date", "symbol"]].copy()
    if config.label_column in scoring_frame.columns:
        scored["label"] = scoring_frame[config.label_column]
    scored["prediction"] = predictions
    scored["train_end_month"] = str(train_months[-1])
    scored["test_month"] = str(test_month)

    scores_path = Path(config.output_scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(scores_path, index=False)

    labeled_eval = scored.dropna(subset=["label"]).copy() if "label" in scored.columns else pd.DataFrame()
    metrics: dict[str, float | int | str] = {
        "label_column": config.label_column,
        "feature_count": int(len(config.feature_columns)),
        "train_window_months": config.train_window_months,
        "train_rows": int(len(train_frame)),
        "train_start_month": str(train_months[0]),
        "train_end_month": str(train_months[-1]),
        "test_month": str(test_month),
        "as_of_date": as_of_date.date().isoformat(),
        "scored_rows": int(len(scored)),
        "scored_symbol_count": int(scored["symbol"].nunique()),
        "eval_rows": int(len(labeled_eval)),
    }
    if not labeled_eval.empty:
        metrics["mae"] = float((labeled_eval["prediction"] - labeled_eval["label"]).abs().mean())
        metrics["rmse"] = float((((labeled_eval["prediction"] - labeled_eval["label"]) ** 2).mean()) ** 0.5)
        metrics["spearman_ic"] = float(labeled_eval[["prediction", "label"]].corr(method="spearman").iloc[0, 1])
    else:
        metrics["mae"] = "n/a"
        metrics["rmse"] = "n/a"
        metrics["spearman_ic"] = "n/a"

    metrics_path = Path(config.output_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info(
        "single-date score complete as_of_date=%s scored_rows=%s scored_symbols=%s train_rows=%s output_scores=%s",
        metrics["as_of_date"],
        metrics["scored_rows"],
        metrics["scored_symbol_count"],
        metrics["train_rows"],
        scores_path.as_posix(),
    )
    return metrics


def train_lightgbm_latest_inference(config: LatestInferenceConfig) -> dict[str, float | int | str]:
    metrics = train_lightgbm_walk_forward_as_of_date(
        WalkForwardAsOfDateConfig(
            factor_panel_path=config.factor_panel_path,
            output_scores_path=config.output_scores_path,
            output_metrics_path=config.output_metrics_path,
            label_column=config.label_column,
            as_of_date=config.inference_date,
            train_window_months=config.train_window_months,
            feature_columns=config.feature_columns,
            num_leaves=config.num_leaves,
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            min_data_in_leaf=config.min_data_in_leaf,
        )
    )
    metrics["inference_date"] = str(metrics["as_of_date"])
    return metrics


def train_lightgbm_walk_forward(config: WalkForwardConfig) -> dict[str, float | int | str]:
    import lightgbm as lgb

    frame = pd.read_parquet(config.factor_panel_path).sort_values(["trade_date", "symbol"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=list(config.feature_columns)).copy()
    frame["month"] = frame["trade_date"].dt.to_period("M")

    all_months = sorted(frame["month"].unique().tolist())
    start_period = pd.Period(config.test_start_month, freq="M")
    end_period = pd.Period(config.test_end_month, freq="M")
    test_months = [month for month in all_months if start_period <= month <= end_period]
    if not test_months:
        raise ValueError("walk-forward test month range produced no periods")

    scored_parts: list[pd.DataFrame] = []
    window_metrics: list[dict[str, float | int | str]] = []
    eval_metrics: list[dict[str, float]] = []

    for test_month in test_months:
        month_index = all_months.index(test_month)
        train_end_index = month_index - 1
        train_start_index = max(0, train_end_index - config.train_window_months + 1)
        if train_end_index < 0:
            continue
        train_months = all_months[train_start_index : train_end_index + 1]
        train_frame = frame.loc[frame["month"].isin(train_months)].copy()
        test_frame = frame.loc[frame["month"] == test_month].copy()
        train_frame = train_frame.dropna(subset=list(config.feature_columns) + [config.label_column]).copy()
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

        scored = test_frame.loc[:, ["trade_date", "symbol"]].copy()
        if config.label_column in test_frame.columns:
            scored["label"] = test_frame[config.label_column]
        scored["prediction"] = predictions
        scored["train_end_month"] = str(train_months[-1])
        scored["test_month"] = str(test_month)
        scored_parts.append(scored)

        labeled_eval = scored.dropna(subset=["label"]).copy() if "label" in scored.columns else pd.DataFrame()
        metric_row: dict[str, float | int | str] = {
            "test_month": str(test_month),
            "train_start_month": str(train_months[0]),
            "train_end_month": str(train_months[-1]),
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "eval_rows": int(len(labeled_eval)),
        }
        if not labeled_eval.empty:
            mae = float((labeled_eval["prediction"] - labeled_eval["label"]).abs().mean())
            rmse = float((((labeled_eval["prediction"] - labeled_eval["label"]) ** 2).mean()) ** 0.5)
            ic = float(labeled_eval[["prediction", "label"]].corr(method="spearman").iloc[0, 1])
            metric_row.update(
                {
                    "mae": mae,
                    "rmse": rmse,
                    "spearman_ic": ic,
                }
            )
            eval_metrics.append({"mae": mae, "rmse": rmse, "spearman_ic": ic})
        else:
            metric_row.update(
                {
                    "mae": "n/a",
                    "rmse": "n/a",
                    "spearman_ic": "n/a",
                }
            )
        window_metrics.append(metric_row)

    if not scored_parts:
        raise ValueError("walk-forward training produced no scored windows")

    all_scored = pd.concat(scored_parts, ignore_index=True).sort_values(["trade_date", "symbol"])
    scores_path = Path(config.output_scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    all_scored.to_parquet(scores_path, index=False)

    mean_mae = float(pd.Series([item["mae"] for item in eval_metrics]).mean()) if eval_metrics else 0.0
    mean_rmse = float(pd.Series([item["rmse"] for item in eval_metrics]).mean()) if eval_metrics else 0.0
    mean_ic = float(pd.Series([item["spearman_ic"] for item in eval_metrics]).mean()) if eval_metrics else 0.0
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
