from __future__ import annotations

from ashare_backtest.cli.main import (
    analyze_start_date_robustness_from_config,
    resolve_start_date_robustness_output_path,
)


def test_resolve_start_date_robustness_output_path_appends_window() -> None:
    assert (
        resolve_start_date_robustness_output_path(
            "research_industry_v4_v1_1",
            "2025-06-03",
            "2026-03-31",
            8,
            "monthly",
        )
        == "research/models/start_date_robustness_research_industry_v4_v1_1_2025-06-03_to_2026-03-31_8m_monthly.json"
    )


def test_analyze_start_date_robustness_from_config_uses_defaults(monkeypatch, tmp_path) -> None:
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
end_date = "2026-03-31"
top_k = 6
rebalance_every = 5
lookback_window = 20
min_hold_bars = 8
keep_buffer = 2
min_turnover_names = 3
initial_cash = 100000
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_analyze(config):  # type: ignore[no-untyped-def]
        captured["scores_path"] = config.scores_path
        captured["analysis_start_date"] = config.analysis_start_date
        captured["analysis_end_date"] = config.analysis_end_date
        captured["holding_months"] = config.holding_months
        captured["cadence"] = config.cadence
        captured["top_k"] = config.top_k
        captured["rebalance_every"] = config.rebalance_every
        captured["min_hold_bars"] = config.min_hold_bars
        captured["universe_name"] = config.universe_name
        captured["output_path"] = config.output_path
        return {"summary": {"sample_count": 3, "holding_months": config.holding_months, "win_rate": 2 / 3}}

    monkeypatch.setattr("ashare_backtest.cli.main.analyze_start_date_robustness", fake_analyze)

    payload = analyze_start_date_robustness_from_config(
        config_path=config_path.as_posix(),
        scores_path="research/models/walk_forward_scores_industry_v4_v1_1_2025-01_to_2026-03.parquet",
        analysis_start_date="2025-06-03",
        holding_months=8,
    )

    assert payload["summary"]["sample_count"] == 3
    assert captured["scores_path"] == "research/models/walk_forward_scores_industry_v4_v1_1_2025-01_to_2026-03.parquet"
    assert captured["analysis_start_date"] == "2025-06-03"
    assert captured["analysis_end_date"] == "2026-03-31"
    assert captured["holding_months"] == 8
    assert captured["cadence"] == "monthly"
    assert captured["top_k"] == 6
    assert captured["rebalance_every"] == 5
    assert captured["min_hold_bars"] == 8
    assert captured["universe_name"] == "tradable_core"
    assert (
        captured["output_path"]
        == "research/models/start_date_robustness_research_industry_v4_v1_1_2025-06-03_to_2026-03-31_8m_monthly.json"
    )
    assert payload["output_path"] == captured["output_path"]

