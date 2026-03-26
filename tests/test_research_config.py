from __future__ import annotations

from ashare_backtest.cli.research_config import load_research_config


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
    assert config.max_position_weight == 0.2
    assert config.max_trade_participation_rate == 0.05
    assert config.max_pending_days == 2
