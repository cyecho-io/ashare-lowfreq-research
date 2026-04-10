from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ashare_backtest.logging_utils import get_logger


LOGGER = get_logger("research.trainer")
NATIVE_BACKEND = "native"
NATIVE_MODEL = "lgbm"


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


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


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
    validation_window_months: int = 1
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
    validation_window_months: int = 1
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
    validation_window_months: int = 1
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
    validation_window_months: int = 1
    num_leaves: int = 31
    learning_rate: float = 0.05
    n_estimators: int = 200
    min_data_in_leaf: int = 20


def _build_lgbm_regressor(config) -> object:
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="regression",
        num_leaves=config.num_leaves,
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        min_data_in_leaf=config.min_data_in_leaf,
        random_state=42,
        verbose=-1,
    )


def _compute_eval_metrics(scored: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    labeled_eval = scored.dropna(subset=["label"]).copy() if "label" in scored.columns else pd.DataFrame()
    if labeled_eval.empty:
        return labeled_eval, {"eval_rows": 0, "mae": "n/a", "rmse": "n/a", "spearman_ic": "n/a"}

    mae = float((labeled_eval["prediction"] - labeled_eval["label"]).abs().mean())
    rmse = float((((labeled_eval["prediction"] - labeled_eval["label"]) ** 2).mean()) ** 0.5)
    ic = float(labeled_eval[["prediction", "label"]].corr(method="spearman").iloc[0, 1])
    return labeled_eval, {"eval_rows": int(len(labeled_eval)), "mae": mae, "rmse": rmse, "spearman_ic": ic}


def _resolve_walk_forward_windows(
    *,
    all_months: list[pd.Period],
    anchor_month: pd.Period,
    train_window_months: int,
    validation_window_months: int,
) -> tuple[list[pd.Period], list[pd.Period]]:
    month_index = all_months.index(anchor_month)
    validation_end_index = month_index - 1
    if validation_end_index < 0:
        raise ValueError("walk-forward split has no prior months available for validation")

    validation_start_index = max(0, validation_end_index - validation_window_months + 1)
    validation_months = all_months[validation_start_index : validation_end_index + 1]
    if validation_window_months > 0 and len(validation_months) < validation_window_months:
        raise ValueError("walk-forward split does not have enough prior months for validation")

    train_end_index = validation_start_index - 1
    if train_end_index < 0:
        raise ValueError("walk-forward split has no prior months available for training")

    train_start_index = max(0, train_end_index - train_window_months + 1)
    train_months = all_months[train_start_index : train_end_index + 1]
    if not train_months:
        raise ValueError("walk-forward split produced an empty training window")
    return train_months, validation_months


def _score_frame(model, frame: pd.DataFrame, *, feature_columns: tuple[str, ...], label_column: str) -> pd.DataFrame:
    predictions = model.predict(frame.loc[:, list(feature_columns)])
    scored = frame.loc[:, ["trade_date", "symbol"]].copy()
    if label_column in frame.columns:
        scored["label"] = frame[label_column]
    scored["prediction"] = predictions
    return scored


def _attach_native_output_metadata(scored: pd.DataFrame) -> pd.DataFrame:
    enriched = scored.copy()
    enriched["backend"] = NATIVE_BACKEND
    enriched["model"] = NATIVE_MODEL
    return enriched


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
    scored = _attach_native_output_metadata(scored)

    scores_path = Path(config.output_scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(scores_path, index=False)

    mae = float((scored["prediction"] - scored["label"]).abs().mean())
    rmse = float((((scored["prediction"] - scored["label"]) ** 2).mean()) ** 0.5)
    ic = float(scored[["prediction", "label"]].corr(method="spearman").iloc[0, 1])
    metrics = {
        "backend": NATIVE_BACKEND,
        "model": NATIVE_MODEL,
        "label_column": config.label_column,
        "feature_columns": list(config.feature_columns),
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
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def train_lightgbm_walk_forward_as_of_date(config: WalkForwardAsOfDateConfig) -> dict[str, float | int | str]:
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

    historical_frame = frame.loc[frame["trade_date"] < as_of_date].copy()
    historical_frame["month"] = historical_frame["trade_date"].dt.to_period("M")
    all_months = sorted(historical_frame["month"].dropna().unique().tolist())
    if not all_months:
        raise ValueError("latest inference has no historical months available")
    validation_months = all_months[-config.validation_window_months :]
    if config.validation_window_months > 0 and len(validation_months) < config.validation_window_months:
        raise ValueError("latest inference does not have enough prior months for validation")
    train_months = all_months[: -config.validation_window_months] if config.validation_window_months > 0 else all_months
    if config.train_window_months > 0:
        train_months = train_months[-config.train_window_months :]
    if not train_months:
        raise ValueError("latest inference has no prior months available for training")

    train_frame = historical_frame.loc[historical_frame["month"].isin(train_months)].copy()
    validation_frame = historical_frame.loc[historical_frame["month"].isin(validation_months)].copy()
    train_frame = train_frame.dropna(subset=list(config.feature_columns) + [config.label_column]).copy()
    validation_frame = validation_frame.dropna(subset=list(config.feature_columns) + [config.label_column]).copy()
    if train_frame.empty:
        raise ValueError("latest inference training set is empty")
    if validation_frame.empty:
        raise ValueError("latest inference validation set is empty")

    validation_model = _build_lgbm_regressor(config)
    validation_model.fit(train_frame.loc[:, list(config.feature_columns)], train_frame[config.label_column])
    validation_scored = _score_frame(
        validation_model,
        validation_frame,
        feature_columns=config.feature_columns,
        label_column=config.label_column,
    )
    _, validation_metrics = _compute_eval_metrics(validation_scored)

    final_train_frame = pd.concat([train_frame, validation_frame], ignore_index=True)
    scoring_model = _build_lgbm_regressor(config)
    scoring_model.fit(final_train_frame.loc[:, list(config.feature_columns)], final_train_frame[config.label_column])
    scored = _score_frame(
        scoring_model,
        scoring_frame,
        feature_columns=config.feature_columns,
        label_column=config.label_column,
    )
    scored["train_end_date"] = train_frame["trade_date"].max().date().isoformat()
    scored["validation_end_date"] = validation_frame["trade_date"].max().date().isoformat()
    scored["as_of_date"] = as_of_date.date().isoformat()
    scored = _attach_native_output_metadata(scored)

    scores_path = Path(config.output_scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(scores_path, index=False)

    metrics = {
        "backend": NATIVE_BACKEND,
        "model": NATIVE_MODEL,
        "label_column": config.label_column,
        "feature_columns": list(config.feature_columns),
        "feature_count": int(len(config.feature_columns)),
        "train_window_months": config.train_window_months,
        "validation_window_months": config.validation_window_months,
        "train_rows": int(len(train_frame)),
        "train_start_date": train_frame["trade_date"].min().date().isoformat(),
        "train_end_date": train_frame["trade_date"].max().date().isoformat(),
        "validation_rows": int(len(validation_frame)),
        "validation_start_date": validation_frame["trade_date"].min().date().isoformat(),
        "validation_end_date": validation_frame["trade_date"].max().date().isoformat(),
        "validation_eval_rows": validation_metrics["eval_rows"],
        "validation_mae": validation_metrics["mae"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_spearman_ic": validation_metrics["spearman_ic"],
        "final_train_rows": int(len(final_train_frame)),
        "as_of_date": as_of_date.date().isoformat(),
        "scored_rows": int(len(scored)),
        "scored_symbol_count": int(scored["symbol"].nunique()),
    }

    metrics_path = Path(config.output_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def train_lightgbm_walk_forward_single_date(config: WalkForwardSingleDateConfig) -> dict[str, float | int | str]:
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
    train_months, validation_months = _resolve_walk_forward_windows(
        all_months=all_months,
        anchor_month=test_month,
        train_window_months=config.train_window_months,
        validation_window_months=config.validation_window_months,
    )

    train_frame = frame.loc[frame["month"].isin(train_months)].copy()
    validation_frame = frame.loc[frame["month"].isin(validation_months)].copy()
    train_frame = train_frame.dropna(subset=list(config.feature_columns) + [config.label_column]).copy()
    validation_frame = validation_frame.dropna(subset=list(config.feature_columns) + [config.label_column]).copy()
    scoring_frame = frame.loc[frame["trade_date"] == as_of_date].copy()
    if train_frame.empty:
        raise ValueError("walk-forward single-date training set is empty")
    if validation_frame.empty:
        raise ValueError("walk-forward single-date validation set is empty")
    if scoring_frame.empty:
        raise ValueError("as-of date produced an empty dataset")

    validation_model = _build_lgbm_regressor(config)
    validation_model.fit(train_frame.loc[:, list(config.feature_columns)], train_frame[config.label_column])
    validation_scored = _score_frame(
        validation_model,
        validation_frame,
        feature_columns=config.feature_columns,
        label_column=config.label_column,
    )
    _, validation_metrics = _compute_eval_metrics(validation_scored)

    final_train_frame = pd.concat([train_frame, validation_frame], ignore_index=True)
    scoring_model = _build_lgbm_regressor(config)
    scoring_model.fit(final_train_frame.loc[:, list(config.feature_columns)], final_train_frame[config.label_column])
    scored = _score_frame(
        scoring_model,
        scoring_frame,
        feature_columns=config.feature_columns,
        label_column=config.label_column,
    )
    scored["train_end_month"] = str(train_months[-1])
    scored["validation_end_month"] = str(validation_months[-1])
    scored["test_month"] = str(test_month)
    scored = _attach_native_output_metadata(scored)

    scores_path = Path(config.output_scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(scores_path, index=False)

    metrics: dict[str, float | int | str] = {
        "backend": NATIVE_BACKEND,
        "model": NATIVE_MODEL,
        "label_column": config.label_column,
        "feature_columns": list(config.feature_columns),
        "feature_count": int(len(config.feature_columns)),
        "train_window_months": config.train_window_months,
        "validation_window_months": config.validation_window_months,
        "train_rows": int(len(train_frame)),
        "train_start_month": str(train_months[0]),
        "train_end_month": str(train_months[-1]),
        "validation_rows": int(len(validation_frame)),
        "validation_start_month": str(validation_months[0]),
        "validation_end_month": str(validation_months[-1]),
        "validation_eval_rows": validation_metrics["eval_rows"],
        "validation_mae": validation_metrics["mae"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_spearman_ic": validation_metrics["spearman_ic"],
        "final_train_rows": int(len(final_train_frame)),
        "test_month": str(test_month),
        "as_of_date": as_of_date.date().isoformat(),
        "scored_rows": int(len(scored)),
        "scored_symbol_count": int(scored["symbol"].nunique()),
    }
    _, scored_metrics = _compute_eval_metrics(scored)
    metrics["eval_rows"] = scored_metrics["eval_rows"]
    metrics["mae"] = scored_metrics["mae"]
    metrics["rmse"] = scored_metrics["rmse"]
    metrics["spearman_ic"] = scored_metrics["spearman_ic"]

    metrics_path = Path(config.output_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
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
            validation_window_months=config.validation_window_months,
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
    validation_eval_metrics: list[dict[str, float]] = []

    for test_month in test_months:
        try:
            train_months, validation_months = _resolve_walk_forward_windows(
                all_months=all_months,
                anchor_month=test_month,
                train_window_months=config.train_window_months,
                validation_window_months=config.validation_window_months,
            )
        except ValueError:
            continue

        train_frame = frame.loc[frame["month"].isin(train_months)].copy()
        validation_frame = frame.loc[frame["month"].isin(validation_months)].copy()
        test_frame = frame.loc[frame["month"] == test_month].copy()
        train_frame = train_frame.dropna(subset=list(config.feature_columns) + [config.label_column]).copy()
        validation_frame = validation_frame.dropna(subset=list(config.feature_columns) + [config.label_column]).copy()
        if train_frame.empty or validation_frame.empty or test_frame.empty:
            continue

        validation_model = _build_lgbm_regressor(config)
        validation_model.fit(train_frame.loc[:, list(config.feature_columns)], train_frame[config.label_column])
        validation_scored = _score_frame(
            validation_model,
            validation_frame,
            feature_columns=config.feature_columns,
            label_column=config.label_column,
        )
        _, validation_metrics = _compute_eval_metrics(validation_scored)

        final_train_frame = pd.concat([train_frame, validation_frame], ignore_index=True)
        scoring_model = _build_lgbm_regressor(config)
        scoring_model.fit(final_train_frame.loc[:, list(config.feature_columns)], final_train_frame[config.label_column])
        scored = _score_frame(
            scoring_model,
            test_frame,
            feature_columns=config.feature_columns,
            label_column=config.label_column,
        )
        scored["train_end_month"] = str(train_months[-1])
        scored["validation_end_month"] = str(validation_months[-1])
        scored["test_month"] = str(test_month)
        scored_parts.append(scored)

        metric_row: dict[str, float | int | str] = {
            "test_month": str(test_month),
            "train_start_month": str(train_months[0]),
            "train_end_month": str(train_months[-1]),
            "validation_start_month": str(validation_months[0]),
            "validation_end_month": str(validation_months[-1]),
            "train_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
            "final_train_rows": int(len(final_train_frame)),
            "test_rows": int(len(test_frame)),
            "validation_eval_rows": validation_metrics["eval_rows"],
            "validation_mae": validation_metrics["mae"],
            "validation_rmse": validation_metrics["rmse"],
            "validation_spearman_ic": validation_metrics["spearman_ic"],
        }
        _, scored_metrics = _compute_eval_metrics(scored)
        metric_row["eval_rows"] = scored_metrics["eval_rows"]
        if scored_metrics["eval_rows"] != 0:
            mae = float(scored_metrics["mae"])
            rmse = float(scored_metrics["rmse"])
            ic = float(scored_metrics["spearman_ic"])
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
        if validation_metrics["eval_rows"] != 0:
            validation_eval_metrics.append(
                {
                    "mae": float(validation_metrics["mae"]),
                    "rmse": float(validation_metrics["rmse"]),
                    "spearman_ic": float(validation_metrics["spearman_ic"]),
                }
            )
        window_metrics.append(metric_row)

    if not scored_parts:
        raise ValueError("walk-forward training produced no scored windows")

    all_scored = pd.concat(scored_parts, ignore_index=True).sort_values(["trade_date", "symbol"])
    all_scored = _attach_native_output_metadata(all_scored)
    scores_path = Path(config.output_scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    all_scored.to_parquet(scores_path, index=False)

    mean_mae = float(pd.Series([item["mae"] for item in eval_metrics]).mean()) if eval_metrics else 0.0
    mean_rmse = float(pd.Series([item["rmse"] for item in eval_metrics]).mean()) if eval_metrics else 0.0
    mean_ic = float(pd.Series([item["spearman_ic"] for item in eval_metrics]).mean()) if eval_metrics else 0.0
    mean_validation_mae = (
        float(pd.Series([item["mae"] for item in validation_eval_metrics]).mean()) if validation_eval_metrics else 0.0
    )
    mean_validation_rmse = (
        float(pd.Series([item["rmse"] for item in validation_eval_metrics]).mean()) if validation_eval_metrics else 0.0
    )
    mean_validation_ic = (
        float(pd.Series([item["spearman_ic"] for item in validation_eval_metrics]).mean())
        if validation_eval_metrics
        else 0.0
    )
    metrics = {
        "backend": NATIVE_BACKEND,
        "model": NATIVE_MODEL,
        "label_column": config.label_column,
        "feature_columns": list(config.feature_columns),
        "feature_count": int(len(config.feature_columns)),
        "train_window_months": config.train_window_months,
        "validation_window_months": config.validation_window_months,
        "test_start_month": config.test_start_month,
        "test_end_month": config.test_end_month,
        "window_count": len(window_metrics),
        "total_scored_rows": int(len(all_scored)),
        "mean_validation_mae": mean_validation_mae,
        "mean_validation_rmse": mean_validation_rmse,
        "mean_validation_spearman_ic": mean_validation_ic,
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "mean_spearman_ic": mean_ic,
        "windows": window_metrics,
    }

    metrics_path = Path(config.output_metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics
