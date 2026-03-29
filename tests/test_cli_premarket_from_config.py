from __future__ import annotations

from ashare_backtest.cli.main import generate_premarket_reference_from_config, resolve_premarket_output_path


def test_resolve_premarket_output_path_uses_factor_spec_and_trade_date() -> None:
    assert (
        resolve_premarket_output_path("research_industry_v4_v1_1", "2026-03-27")
        == "research/models/premarket_reference_research_industry_v4_v1_1_2026-03-27.json"
    )


def test_generate_premarket_reference_from_config_uses_defaults(monkeypatch, tmp_path) -> None:
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
top_k = 6
rebalance_every = 5
lookback_window = 20
min_hold_bars = 8
keep_buffer = 2
min_turnover_names = 3
min_daily_amount = 0
max_names_per_industry = 2
initial_cash = 100000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
""".strip(),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_generate(config, provider=None):  # type: ignore[no-untyped-def]
        captured["scores_path"] = config.scores_path
        captured["storage_root"] = config.storage_root
        captured["output_path"] = config.output_path
        captured["trade_date"] = config.trade_date
        captured["top_k"] = config.top_k
        captured["rebalance_every"] = config.rebalance_every
        captured["lookback_window"] = config.lookback_window
        captured["min_hold_bars"] = config.min_hold_bars
        captured["max_names_per_industry"] = config.max_names_per_industry
        return {
            "summary": {
                "signal_date": "2026-03-26",
                "execution_date": config.trade_date,
            },
            "actions": [],
        }

    monkeypatch.setattr("ashare_backtest.cli.main.generate_premarket_reference", fake_generate)

    payload = generate_premarket_reference_from_config(
        config_path=config_path.as_posix(),
        scores_path="research/models/walk_forward_scores_industry_v4_v1_1_2026-03-26.parquet",
        trade_date="2026-03-27",
    )

    assert payload["summary"]["execution_date"] == "2026-03-27"
    assert captured["scores_path"] == "research/models/walk_forward_scores_industry_v4_v1_1_2026-03-26.parquet"
    assert captured["storage_root"] == "storage"
    assert captured["output_path"] == "research/models/premarket_reference_research_industry_v4_v1_1_2026-03-27.json"
    assert captured["trade_date"] == "2026-03-27"
    assert captured["top_k"] == 6
    assert captured["rebalance_every"] == 5
    assert captured["lookback_window"] == 20
    assert captured["min_hold_bars"] == 8
    assert captured["max_names_per_industry"] == 2
