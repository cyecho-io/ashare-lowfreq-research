from __future__ import annotations

from ashare_backtest.cli.research_config import load_research_config, resolve_dated_output_path, resolve_research_config_path


def test_load_research_config_includes_model_backtest_optional_fields(tmp_path) -> None:
    config_path = tmp_path / "research.toml"
    config_path.write_text(
        """
[storage]
root = "storage"

[factors]
output_path = "research/factors/panel.parquet"
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-10"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/scores.parquet"
metric_output_path = "research/models/metrics.json"

[analysis]
layer_output_path = "research/models/layers.json"

[model_backtest]
output_dir = "results/model_score_backtest"
start_date = "2025-01-02"
end_date = "2026-02-27"
top_k = 6
rebalance_every = 5
lookback_window = 20
min_hold_bars = 8
keep_buffer = 2
min_turnover_names = 3
min_daily_amount = 100000
max_close_price = 50
max_names_per_industry = 2
max_position_weight = 0.2
max_trade_participation_rate = 0.05
max_pending_days = 2
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
""".strip(),
        encoding="utf-8",
    )

    config = load_research_config(config_path)

    assert config.factor_universe_name == "tradable_core"
    assert config.factor_spec_id == "research"
    assert config.factor_as_of_date == "2026-03-10"
    assert config.max_close_price == 50
    assert config.max_position_weight == 0.2
    assert config.max_trade_participation_rate == 0.05
    assert config.max_pending_days == 2


def test_load_research_config_resolves_standard_snapshot_path_when_output_path_missing(tmp_path) -> None:
    config_path = tmp_path / "demo_strategy.toml"
    config_path.write_text(
        """
[storage]
root = "storage"

[factors]
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-10"

[factor_spec]
id = "research_industry_v4_v1_1"

[research_snapshot]
as_of_date = "2026-03-25"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/scores.parquet"
metric_output_path = "research/models/metrics.json"

[analysis]
layer_output_path = "research/models/layers.json"

[model_backtest]
output_dir = "results/model_score_backtest"
start_date = "2025-01-02"
end_date = "2026-02-27"
""".strip(),
        encoding="utf-8",
    )

    config = load_research_config(config_path)

    assert config.factor_spec_id == "research_industry_v4_v1_1"
    assert config.factor_as_of_date == "2026-03-25"
    assert (
        config.factor_snapshot_path
        == "research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-25.parquet"
    )


def test_resolve_research_config_path_supports_factor_spec_id(tmp_path, monkeypatch) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    config_path = configs_dir / "research_industry_v4_v1_1.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    resolved = resolve_research_config_path(factor_spec_id="research_industry_v4_v1_1")

    assert resolved == config_path


def test_resolve_dated_output_path_appends_as_of_date() -> None:
    assert (
        resolve_dated_output_path("research/models/walk_forward_scores_industry_v4_v1_1.parquet", "2026-03-25")
        == "research/models/walk_forward_scores_industry_v4_v1_1_2026-03-25.parquet"
    )
    assert (
        resolve_dated_output_path("research/models/walk_forward_metrics_industry_v4_v1_1.json", "2026-03-25")
        == "research/models/walk_forward_metrics_industry_v4_v1_1_2026-03-25.json"
    )
