from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QlibFeatureSpec:
    name: str
    expression: str


DEFAULT_QLIB_FEATURES: tuple[QlibFeatureSpec, ...] = (
    QlibFeatureSpec("mom_5", "$close / Ref($close, 5) - 1"),
    QlibFeatureSpec("mom_10", "$close / Ref($close, 10) - 1"),
    QlibFeatureSpec("mom_20", "$close / Ref($close, 20) - 1"),
    QlibFeatureSpec("ma_gap_5", "$close / Mean($close, 5) - 1"),
    QlibFeatureSpec("ma_gap_10", "$close / Mean($close, 10) - 1"),
    QlibFeatureSpec("ma_gap_20", "$close / Mean($close, 20) - 1"),
    QlibFeatureSpec("volatility_10", "Std($close / Ref($close, 1) - 1, 10)"),
    QlibFeatureSpec("volatility_20", "Std($close / Ref($close, 1) - 1, 20)"),
)

DEFAULT_QLIB_LABEL_EXPRESSION = "Ref($close, -5) / Ref($close, -1) - 1"
DEFAULT_QLIB_LABEL_MODE = "raw_fwd_return_5"


def select_qlib_feature_specs(feature_columns: tuple[str, ...] | list[str] | None) -> tuple[QlibFeatureSpec, ...]:
    if not feature_columns:
        return DEFAULT_QLIB_FEATURES

    feature_name_map = {spec.name: spec for spec in DEFAULT_QLIB_FEATURES}
    selected_specs: list[QlibFeatureSpec] = []
    unknown_names: list[str] = []
    for name in feature_columns:
        normalized = str(name).strip()
        if not normalized:
            continue
        spec = feature_name_map.get(normalized)
        if spec is None:
            unknown_names.append(normalized)
            continue
        selected_specs.append(spec)

    if unknown_names:
        raise ValueError(
            "qlib.feature_columns contains unknown feature names: " + ", ".join(sorted(set(unknown_names)))
        )
    if not selected_specs:
        raise ValueError("qlib.feature_columns did not select any valid default Qlib features")
    return tuple(selected_specs)


def parse_qlib_feature_specs(
    *,
    feature_specs: Any = None,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[QlibFeatureSpec, ...]:
    if feature_specs is not None:
        if not isinstance(feature_specs, list):
            raise ValueError("qlib.feature_specs must be an array of tables when provided")

        parsed_specs: list[QlibFeatureSpec] = []
        for index, item in enumerate(feature_specs):
            if not isinstance(item, dict):
                raise ValueError(f"qlib.feature_specs[{index}] must be a table with name and expression")
            name = str(item.get("name") or "").strip()
            expression = str(item.get("expression") or "").strip()
            if not name or not expression:
                raise ValueError(f"qlib.feature_specs[{index}] must include non-empty name and expression")
            parsed_specs.append(QlibFeatureSpec(name=name, expression=expression))

        if not parsed_specs:
            raise ValueError("qlib.feature_specs did not define any valid feature specs")
        return tuple(parsed_specs)

    return select_qlib_feature_specs(feature_columns)


@dataclass(frozen=True)
class QlibBaseConfig:
    storage_root: str = "storage"
    universe_name: str = ""
    provider_uri: str = "~/.qlib/qlib_data/cn_data"
    region: str = "cn"
    market: str = "csi300"
    benchmark: str = "SH000300"
    freq: str = "day"
    config_id: str = "qlib_default"
    model_name: str = "lgbm"
    label_mode: str = DEFAULT_QLIB_LABEL_MODE
    label_expression: str = DEFAULT_QLIB_LABEL_EXPRESSION
    feature_specs: tuple[QlibFeatureSpec, ...] = DEFAULT_QLIB_FEATURES


@dataclass(frozen=True)
class QlibWalkForwardConfig(QlibBaseConfig):
    output_scores_path: str = "research/models/walk_forward_scores_qlib.parquet"
    output_metrics_path: str = "research/models/walk_forward_metrics_qlib.json"
    train_window_months: int = 12
    validation_window_months: int = 1
    test_start_month: str = "2025-07"
    test_end_month: str = "2026-02"
    data_start_date: str | None = None
    data_end_date: str | None = None


@dataclass(frozen=True)
class QlibAsOfDateConfig(QlibBaseConfig):
    output_scores_path: str = "research/models/walk_forward_scores_qlib_as_of_date.parquet"
    output_metrics_path: str = "research/models/walk_forward_metrics_qlib_as_of_date.json"
    as_of_date: str | None = None
    train_window_months: int = 12
    validation_window_months: int = 1


@dataclass(frozen=True)
class QlibSingleDateConfig(QlibBaseConfig):
    output_scores_path: str = "research/models/walk_forward_scores_qlib_single_date.parquet"
    output_metrics_path: str = "research/models/walk_forward_metrics_qlib_single_date.json"
    test_month: str = "2026-02"
    as_of_date: str = "2026-02-27"
    train_window_months: int = 12
    validation_window_months: int = 1
