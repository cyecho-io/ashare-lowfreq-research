from __future__ import annotations

from ashare_backtest.cli.main import (
    resolve_month_range_output_path,
    train_walk_forward_from_config,
    train_walk_forward_single_date_from_config,
)


def test_resolve_month_range_output_path_appends_month_range() -> None:
    assert (
        resolve_month_range_output_path(
            "research/models/walk_forward_scores_industry_v4_v1_1.parquet",
            "2025-01",
            "2026-03",
        )
        == "research/models/walk_forward_scores_industry_v4_v1_1_2025-01_to_2026-03.parquet"
    )
    assert (
        resolve_month_range_output_path(
            "research/models/walk_forward_metrics_industry_v4_v1_1.json",
            "2025-01",
            "2026-03",
        )
        == "research/models/walk_forward_metrics_industry_v4_v1_1_2025-01_to_2026-03.json"
    )


def test_train_walk_forward_from_config_uses_defaults(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "research_industry_v4_v1_1.toml"
    config_path.write_text(
        """
[storage]
root = "storage"

[factor_spec]
id = "research_industry_v4_v1_1"
universe_name = "tradable_core"
start_date = "2024-01-02"

[research_snapshot]
as_of_date = "2026-03-10"

[factors]
output_path = "research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-10.parquet"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_scores_industry_v4_v1_1.parquet"
metric_output_path = "research/models/walk_forward_metrics_industry_v4_v1_1.json"

[analysis]
layer_output_path = "research/models/layers.json"

[model_backtest]
output_dir = "results/model_score_backtest"
start_date = "2025-01-02"
end_date = "2026-02-27"
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_train(config):  # type: ignore[no-untyped-def]
        captured["factor_panel_path"] = config.factor_panel_path
        captured["output_scores_path"] = config.output_scores_path
        captured["output_metrics_path"] = config.output_metrics_path
        captured["label_column"] = config.label_column
        captured["train_window_months"] = config.train_window_months
        captured["test_start_month"] = config.test_start_month
        captured["test_end_month"] = config.test_end_month
        return {
            "window_count": 2,
            "mean_mae": 0.1,
            "mean_rmse": 0.2,
            "mean_spearman_ic": 0.3,
        }

    monkeypatch.setattr("ashare_backtest.cli.main.train_lightgbm_walk_forward", fake_train)

    metrics = train_walk_forward_from_config(
        config_path=config_path.as_posix(),
        factor_panel_path="research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-26.parquet",
        test_start_month="2025-01",
        test_end_month="2026-03",
    )

    assert metrics["window_count"] == 2
    assert (
        captured["factor_panel_path"]
        == "research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-26.parquet"
    )
    assert (
        captured["output_scores_path"]
        == "research/models/walk_forward_scores_industry_v4_v1_1_2025-01_to_2026-03.parquet"
    )
    assert (
        captured["output_metrics_path"]
        == "research/models/walk_forward_metrics_industry_v4_v1_1_2025-01_to_2026-03.json"
    )
    assert captured["label_column"] == "industry_excess_fwd_return_5"
    assert captured["train_window_months"] == 12
    assert captured["test_start_month"] == "2025-01"
    assert captured["test_end_month"] == "2026-03"


def test_train_walk_forward_single_date_from_config_uses_defaults(monkeypatch, tmp_path) -> None:
    config_path = tmp_path / "research_industry_v4_v1_1.toml"
    config_path.write_text(
        """
[storage]
root = "storage"

[factor_spec]
id = "research_industry_v4_v1_1"
universe_name = "tradable_core"
start_date = "2024-01-02"

[research_snapshot]
as_of_date = "2026-03-10"

[factors]
output_path = "research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-10.parquet"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_scores_industry_v4_v1_1.parquet"
metric_output_path = "research/models/walk_forward_metrics_industry_v4_v1_1.json"

[analysis]
layer_output_path = "research/models/layers.json"

[model_backtest]
output_dir = "results/model_score_backtest"
start_date = "2025-01-02"
end_date = "2026-02-27"
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_train(config):  # type: ignore[no-untyped-def]
        captured["factor_panel_path"] = config.factor_panel_path
        captured["output_scores_path"] = config.output_scores_path
        captured["output_metrics_path"] = config.output_metrics_path
        captured["label_column"] = config.label_column
        captured["train_window_months"] = config.train_window_months
        captured["test_month"] = config.test_month
        captured["as_of_date"] = config.as_of_date
        return {
            "test_month": config.test_month,
            "as_of_date": config.as_of_date,
            "train_rows": 10,
            "scored_rows": 2,
        }

    monkeypatch.setattr("ashare_backtest.cli.main.train_lightgbm_walk_forward_single_date", fake_train)

    metrics = train_walk_forward_single_date_from_config(
        config_path=config_path.as_posix(),
        test_month="2026-03",
        as_of_date="2026-03-26",
        factor_panel_path="research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-26.parquet",
    )

    assert metrics["test_month"] == "2026-03"
    assert metrics["as_of_date"] == "2026-03-26"
    assert (
        captured["factor_panel_path"]
        == "research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-26.parquet"
    )
    assert (
        captured["output_scores_path"]
        == "research/models/walk_forward_scores_industry_v4_v1_1_2026-03-26.parquet"
    )
    assert (
        captured["output_metrics_path"]
        == "research/models/walk_forward_metrics_industry_v4_v1_1_2026-03-26.json"
    )
    assert captured["label_column"] == "industry_excess_fwd_return_5"
    assert captured["train_window_months"] == 12
    assert captured["test_month"] == "2026-03"
    assert captured["as_of_date"] == "2026-03-26"
