from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from ashare_backtest.qlib_integration.config import QlibAsOfDateConfig, QlibSingleDateConfig, QlibWalkForwardConfig
from ashare_backtest.qlib_integration.dataset import load_qlib_market_frame
from ashare_backtest.qlib_integration.export import export_score_frame
from ashare_backtest.research.trainer import _compute_eval_metrics, _json_safe, _resolve_walk_forward_windows, _score_frame


def _build_lgbm_regressor() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression",
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=200,
        min_data_in_leaf=20,
        random_state=42,
        verbose=-1,
    )


def _feature_columns(config: QlibWalkForwardConfig | QlibAsOfDateConfig | QlibSingleDateConfig) -> tuple[str, ...]:
    return tuple(spec.name for spec in config.feature_specs)


def _serialize_metrics(output_path: str, metrics: dict[str, object]) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_json_safe(metrics), indent=2, ensure_ascii=False), encoding="utf-8")


def train_qlib_walk_forward(config: QlibWalkForwardConfig) -> dict[str, object]:
    end_date = pd.Period(config.test_end_month, freq="M").end_time.date().isoformat()
    start_anchor = pd.Period(config.test_start_month, freq="M") - (config.train_window_months + config.validation_window_months)
    start_date = start_anchor.start_time.date().isoformat()
    frame = load_qlib_market_frame(config, start_date=start_date, end_date=end_date)
    feature_columns = _feature_columns(config)
    frame["month"] = pd.PeriodIndex(frame["trade_date"], freq="M")

    all_months = sorted(frame["month"].dropna().unique().tolist())
    target_months = [
        month
        for month in all_months
        if pd.Period(config.test_start_month, freq="M") <= month <= pd.Period(config.test_end_month, freq="M")
    ]
    if not target_months:
        raise ValueError("no qlib rows were available inside the requested walk-forward months")

    all_scores: list[pd.DataFrame] = []
    window_metrics: list[dict[str, object]] = []
    aggregated_eval_metrics: list[dict[str, float]] = []
    validation_eval_metrics: list[dict[str, float]] = []
    for test_month in target_months:
        train_months, validation_months = _resolve_walk_forward_windows(
            all_months=all_months,
            anchor_month=test_month,
            train_window_months=config.train_window_months,
            validation_window_months=config.validation_window_months,
        )
        train_frame = frame.loc[frame["month"].isin(train_months)].dropna(subset=[*feature_columns, "label"]).copy()
        validation_frame = frame.loc[frame["month"].isin(validation_months)].dropna(subset=list(feature_columns)).copy()
        test_frame = frame.loc[frame["month"] == test_month].dropna(subset=list(feature_columns)).copy()
        if train_frame.empty or validation_frame.empty or test_frame.empty:
            raise ValueError(f"qlib walk-forward window {test_month} produced an empty split")

        model = _build_lgbm_regressor()
        model.fit(train_frame.loc[:, list(feature_columns)], train_frame["label"])
        scored = _score_frame(model, test_frame, feature_columns=feature_columns, label_column="label")
        scored["train_end_date"] = train_months[-1].end_time.date().isoformat()
        scored["validation_end_date"] = validation_months[-1].end_time.date().isoformat()
        all_scores.append(scored)

        _, eval_metrics = _compute_eval_metrics(scored)
        validation_scored = _score_frame(model, validation_frame, feature_columns=feature_columns, label_column="label")
        _, validation_metrics = _compute_eval_metrics(validation_scored)
        if eval_metrics["eval_rows"] != 0:
            eval_metrics_row = {
                "mae": float(eval_metrics["mae"]),
                "rmse": float(eval_metrics["rmse"]),
                "spearman_ic": float(eval_metrics["spearman_ic"]),
            }
            aggregated_eval_metrics.append(eval_metrics_row)
        if validation_metrics["eval_rows"] != 0:
            validation_eval_metrics.append(
                {
                    "mae": float(validation_metrics["mae"]),
                    "rmse": float(validation_metrics["rmse"]),
                    "spearman_ic": float(validation_metrics["spearman_ic"]),
                }
            )
        window_metrics.append(
            {
                "test_month": str(test_month),
                "train_rows": int(len(train_frame)),
                "validation_rows": int(len(validation_frame)),
                "test_rows": int(len(test_frame)),
                **eval_metrics,
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_spearman_ic": validation_metrics["spearman_ic"],
                "train_end_date": scored["train_end_date"].iloc[0],
                "validation_end_date": scored["validation_end_date"].iloc[0],
            }
        )

    exported = export_score_frame(
        pd.concat(all_scores, ignore_index=True),
        config.output_scores_path,
        backend="qlib",
        model=config.model_name,
        config_id=config.config_id,
    )
    mean_mae = float(pd.Series([item["mae"] for item in aggregated_eval_metrics]).mean()) if aggregated_eval_metrics else "n/a"
    mean_rmse = float(pd.Series([item["rmse"] for item in aggregated_eval_metrics]).mean()) if aggregated_eval_metrics else "n/a"
    mean_ic = (
        float(pd.Series([item["spearman_ic"] for item in aggregated_eval_metrics]).mean())
        if aggregated_eval_metrics
        else "n/a"
    )
    mean_validation_mae = (
        float(pd.Series([item["mae"] for item in validation_eval_metrics]).mean()) if validation_eval_metrics else "n/a"
    )
    mean_validation_rmse = (
        float(pd.Series([item["rmse"] for item in validation_eval_metrics]).mean())
        if validation_eval_metrics
        else "n/a"
    )
    mean_validation_ic = (
        float(pd.Series([item["spearman_ic"] for item in validation_eval_metrics]).mean())
        if validation_eval_metrics
        else "n/a"
    )
    metrics = {
        "backend": "qlib",
        "model": config.model_name,
        "config_id": config.config_id,
        "provider_uri": config.provider_uri,
        "market": config.market,
        "feature_columns": list(feature_columns),
        "label_mode": config.label_mode,
        "label_expression": config.label_expression,
        "window_count": len(window_metrics),
        "total_scored_rows": int(len(exported)),
        "mean_validation_mae": mean_validation_mae,
        "mean_validation_rmse": mean_validation_rmse,
        "mean_validation_spearman_ic": mean_validation_ic,
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "mean_spearman_ic": mean_ic,
        "windows": window_metrics,
    }
    _serialize_metrics(config.output_metrics_path, metrics)
    return metrics


def train_qlib_as_of_date(config: QlibAsOfDateConfig) -> dict[str, object]:
    if not config.as_of_date:
        raise ValueError("as_of_date is required for qlib as-of-date training")
    anchor_month = pd.Period(config.as_of_date[:7], freq="M")
    start_anchor = anchor_month - (config.train_window_months + config.validation_window_months)
    frame = load_qlib_market_frame(
        config,
        start_date=start_anchor.start_time.date().isoformat(),
        end_date=config.as_of_date,
    )
    frame["month"] = pd.PeriodIndex(frame["trade_date"], freq="M")
    all_months = sorted(frame["month"].dropna().unique().tolist())
    train_months, validation_months = _resolve_walk_forward_windows(
        all_months=all_months,
        anchor_month=anchor_month,
        train_window_months=config.train_window_months,
        validation_window_months=config.validation_window_months,
    )
    feature_columns = _feature_columns(config)
    train_frame = frame.loc[frame["month"].isin(train_months)].dropna(subset=[*feature_columns, "label"]).copy()
    test_frame = frame.loc[frame["trade_date"] == pd.Timestamp(config.as_of_date)].dropna(subset=list(feature_columns)).copy()
    if train_frame.empty or test_frame.empty:
        raise ValueError("qlib as-of-date split produced an empty train or test frame")

    model = _build_lgbm_regressor()
    model.fit(train_frame.loc[:, list(feature_columns)], train_frame["label"])
    scored = _score_frame(model, test_frame, feature_columns=feature_columns, label_column="label")
    scored["train_end_date"] = train_months[-1].end_time.date().isoformat()
    scored["validation_end_date"] = validation_months[-1].end_time.date().isoformat()
    exported = export_score_frame(
        scored,
        config.output_scores_path,
        backend="qlib",
        model=config.model_name,
        config_id=config.config_id,
    )
    _, eval_metrics = _compute_eval_metrics(scored)
    metrics = {
        "backend": "qlib",
        "model": config.model_name,
        "config_id": config.config_id,
        "as_of_date": config.as_of_date,
        "feature_columns": list(feature_columns),
        "label_mode": config.label_mode,
        "label_expression": config.label_expression,
        "train_rows": int(len(train_frame)),
        "scored_rows": int(len(exported)),
        **eval_metrics,
        "train_end_date": scored["train_end_date"].iloc[0],
        "validation_end_date": scored["validation_end_date"].iloc[0],
    }
    _serialize_metrics(config.output_metrics_path, metrics)
    return metrics


def train_qlib_single_date(config: QlibSingleDateConfig) -> dict[str, object]:
    anchor_month = pd.Period(config.test_month, freq="M")
    start_anchor = anchor_month - (config.train_window_months + config.validation_window_months)
    frame = load_qlib_market_frame(
        config,
        start_date=start_anchor.start_time.date().isoformat(),
        end_date=config.as_of_date,
    )
    frame["month"] = pd.PeriodIndex(frame["trade_date"], freq="M")
    all_months = sorted(frame["month"].dropna().unique().tolist())
    train_months, validation_months = _resolve_walk_forward_windows(
        all_months=all_months,
        anchor_month=anchor_month,
        train_window_months=config.train_window_months,
        validation_window_months=config.validation_window_months,
    )
    feature_columns = _feature_columns(config)
    train_frame = frame.loc[frame["month"].isin(train_months)].dropna(subset=[*feature_columns, "label"]).copy()
    test_frame = frame.loc[frame["trade_date"] == pd.Timestamp(config.as_of_date)].dropna(subset=list(feature_columns)).copy()
    if train_frame.empty or test_frame.empty:
        raise ValueError("qlib single-date split produced an empty train or test frame")

    model = _build_lgbm_regressor()
    model.fit(train_frame.loc[:, list(feature_columns)], train_frame["label"])
    scored = _score_frame(model, test_frame, feature_columns=feature_columns, label_column="label")
    scored["train_end_date"] = train_months[-1].end_time.date().isoformat()
    scored["validation_end_date"] = validation_months[-1].end_time.date().isoformat()
    exported = export_score_frame(
        scored,
        config.output_scores_path,
        backend="qlib",
        model=config.model_name,
        config_id=config.config_id,
    )
    _, eval_metrics = _compute_eval_metrics(scored)
    metrics = {
        "backend": "qlib",
        "model": config.model_name,
        "config_id": config.config_id,
        "test_month": config.test_month,
        "as_of_date": config.as_of_date,
        "feature_columns": list(feature_columns),
        "label_mode": config.label_mode,
        "label_expression": config.label_expression,
        "train_rows": int(len(train_frame)),
        "scored_rows": int(len(exported)),
        **eval_metrics,
        "train_end_date": scored["train_end_date"].iloc[0],
        "validation_end_date": scored["validation_end_date"].iloc[0],
    }
    _serialize_metrics(config.output_metrics_path, metrics)
    return metrics
