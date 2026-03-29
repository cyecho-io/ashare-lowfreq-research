from __future__ import annotations

import pandas as pd

from ashare_backtest.cli.main import infer_as_of_date_from_factor_panel, train_walk_forward_as_of_date_from_config


def test_train_walk_forward_as_of_date_from_config_uses_defaults(monkeypatch, tmp_path) -> None:
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
        captured["as_of_date"] = config.as_of_date
        return {
            "as_of_date": config.as_of_date,
            "train_rows": 10,
            "scored_rows": 2,
        }

    monkeypatch.setattr("ashare_backtest.cli.main.train_lightgbm_walk_forward_as_of_date", fake_train)

    metrics = train_walk_forward_as_of_date_from_config(
        config_path=config_path.as_posix(),
        as_of_date="2026-03-25",
    )

    assert metrics["as_of_date"] == "2026-03-25"
    assert (
        captured["factor_panel_path"]
        == "research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-25.parquet"
    )
    assert (
        captured["output_scores_path"]
        == "research/models/walk_forward_scores_industry_v4_v1_1_2026-03-25.parquet"
    )
    assert (
        captured["output_metrics_path"]
        == "research/models/walk_forward_metrics_industry_v4_v1_1_2026-03-25.json"
    )
    assert captured["label_column"] == "industry_excess_fwd_return_5"
    assert captured["train_window_months"] == 12
    assert captured["as_of_date"] == "2026-03-25"


def test_train_walk_forward_as_of_date_from_config_infers_date_from_factor_panel_path(monkeypatch, tmp_path) -> None:
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
        captured["as_of_date"] = config.as_of_date
        return {
            "as_of_date": config.as_of_date,
            "train_rows": 10,
            "scored_rows": 2,
        }

    monkeypatch.setattr("ashare_backtest.cli.main.train_lightgbm_walk_forward_as_of_date", fake_train)

    metrics = train_walk_forward_as_of_date_from_config(
        config_path=config_path.as_posix(),
        factor_panel_path="research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-26.parquet",
    )

    assert metrics["as_of_date"] == "2026-03-26"
    assert (
        captured["output_scores_path"]
        == "research/models/walk_forward_scores_industry_v4_v1_1_2026-03-26.parquet"
    )
    assert (
        captured["output_metrics_path"]
        == "research/models/walk_forward_metrics_industry_v4_v1_1_2026-03-26.json"
    )
    assert captured["as_of_date"] == "2026-03-26"


def test_infer_as_of_date_from_factor_panel_reads_trade_date_when_name_has_no_suffix(tmp_path) -> None:
    factor_panel_path = tmp_path / "panel.parquet"
    pd.DataFrame({"trade_date": ["2026-03-24", "2026-03-25"]}).to_parquet(factor_panel_path, index=False)

    assert infer_as_of_date_from_factor_panel(factor_panel_path.as_posix()) == "2026-03-25"
