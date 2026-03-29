from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from ashare_backtest.web.app import (
    _find_previous_simulation_run,
    _next_open_trade_date,
    _resolve_selected_scores_path,
    _previous_open_trade_date,
    BacktestWebApp,
    build_dashboard_summary,
    build_simulation_readiness,
    build_paper_readiness,
    fill_scores_to_signal_date,
    load_latest_simulation_snapshot,
    load_simulation_detail,
    load_simulation_history_detail,
    load_simulation_plan_detail,
    load_score_source_manifest,
    load_latest_paper_snapshot,
    load_latest_paper_lineage,
    load_paper_history_detail,
    load_paper_trade_detail,
    load_run_detail,
    list_score_parquet_files,
    list_paper_trade_summaries,
    list_run_summaries,
    list_simulation_plan_summaries,
    list_simulation_summaries,
    list_strategy_presets,
)


def test_list_strategy_presets_reads_research_configs() -> None:
    presets = list_strategy_presets()
    ids = {preset.id for preset in presets}
    assert "research_industry_v4_v1_1" in ids


def test_build_dashboard_summary_reads_calendar_strategies_and_sqlite(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo_strategy.toml").write_text(
        """
[storage]
root = "storage"

[factor_spec]
id = "demo_strategy"
universe_name = "tradable_core"
start_date = "2024-01-02"

[factors]
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-27"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )
    calendar_path = tmp_path / "storage" / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-27", "is_open": True},
            {"trade_date": "2026-03-28", "is_open": False},
            {"trade_date": "2026-03-30", "is_open": True},
        ]
    ).to_parquet(calendar_path, index=False)
    sqlite_path = tmp_path / "storage" / "source" / "ashare_arena_sync.db"
    sqlite_path.parent.mkdir(parents=True)
    import sqlite3

    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("create table equity_daily_bars (symbol text, trade_date text)")
        conn.execute("create table equity_instruments (symbol text)")
        conn.executemany(
            "insert into equity_daily_bars(symbol, trade_date) values (?, ?)",
            [("AAA", "2024-01-02"), ("BBB", "2026-03-27"), ("AAA", "2026-03-27")],
        )
        conn.executemany("insert into equity_instruments(symbol) values (?)", [("AAA",), ("BBB",), ("CCC",)])
        conn.commit()
    catalog_path = tmp_path / "storage" / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "source_path": "storage/source/ashare_arena_sync.db",
                "imported_at": "2026-03-27T20:46:45",
                "datasets": [
                    {"name": "bars.daily", "min_date": "2024-01-02", "max_date": "2026-03-27"},
                    {"name": "calendar.ashare", "min_date": "2024-01-01", "max_date": "2026-03-30"},
                    {"name": "instruments.ashare", "min_date": "1990-12-01", "max_date": "2026-03-27"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    original_catalog_path = web_app.CATALOG_PATH
    try:
        web_app.REPO_ROOT = tmp_path
        web_app.CATALOG_PATH = catalog_path
        payload = build_dashboard_summary(repo_root=tmp_path, config_root=config_root)
    finally:
        web_app.REPO_ROOT = original_repo_root
        web_app.CATALOG_PATH = original_catalog_path

    assert payload["strategies"]["count"] == 1
    assert payload["sqlite"]["equity_symbol_count"] == 2
    assert payload["sqlite"]["instrument_count"] == 3
    assert payload["sqlite"]["date_min"] == "2024-01-02"
    assert payload["sqlite"]["date_max"] == "2026-03-27"
    assert payload["calendar"]["latest_open_date"] == "2026-03-30"
    assert payload["calendar"]["latest_calendar_date"] == "2026-03-30"


def test_list_score_parquet_files_only_returns_score_parquet_paths(tmp_path: Path) -> None:
    models_root = tmp_path / "research" / "models"
    models_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2025-01-02", "symbol": "AAA", "prediction": 0.1},
            {"trade_date": "2025-01-03", "symbol": "AAA", "prediction": 0.2},
        ]
    ).to_parquet(models_root / "walk_forward_scores_demo.parquet", index=False)
    (models_root / "metrics.json").write_text("{}", encoding="utf-8")
    pd.DataFrame([{"trade_date": "2025-01-01"}]).to_parquet(models_root / "latest_metrics_demo.parquet", index=False)
    nested = models_root / "latest" / "demo_strategy"
    nested.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-25", "symbol": "AAA", "prediction": 0.9},
        ]
    ).to_parquet(nested / "scores.parquet", index=False)

    files = list_score_parquet_files(models_root=models_root)

    assert files == [
        {
            "path": (nested / "scores.parquet").as_posix(),
            "start_date": "2026-03-25",
            "end_date": "2026-03-25",
        },
        {
            "path": (models_root / "walk_forward_scores_demo.parquet").as_posix(),
            "start_date": "2025-01-02",
            "end_date": "2025-01-03",
        },
    ]


def test_list_score_parquet_files_can_exclude_single_day_scores(tmp_path: Path) -> None:
    models_root = tmp_path / "research" / "models"
    models_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2025-01-02", "symbol": "AAA", "prediction": 0.1},
            {"trade_date": "2025-01-03", "symbol": "AAA", "prediction": 0.2},
        ]
    ).to_parquet(models_root / "walk_forward_scores_history.parquet", index=False)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-25", "symbol": "AAA", "prediction": 0.9},
        ]
    ).to_parquet(models_root / "walk_forward_scores_single_day.parquet", index=False)

    files = list_score_parquet_files(models_root=models_root, include_single_day=False)

    assert files == [
        {
            "path": (models_root / "walk_forward_scores_history.parquet").as_posix(),
            "start_date": "2025-01-02",
            "end_date": "2025-01-03",
        }
    ]


def test_load_score_source_manifest_maps_scores_to_upstream_metadata(tmp_path: Path) -> None:
    manifest_path = tmp_path / "research" / "models" / "score_source_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "strategy_id": "research_industry_v4_v1_1",
                        "score_source_id": "premarket_scores_industry_v4_v1_1_latest_active",
                        "scores_path": "research/models/premarket_scores_industry_v4_v1_1_latest_active.parquet",
                        "factor_spec_id": "research_industry_v4_v1_1",
                        "factor_snapshot_date": "2026-03-25",
                        "factor_snapshot_path": "research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-25.parquet",
                        "factor_panel_path": "research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-25.parquet",
                        "config_path": "configs/research_industry_v4_v1_1.toml",
                        "source_kind": "premarket_latest_active",
                        "description": "demo",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        manifest = load_score_source_manifest(manifest_path=manifest_path)
    finally:
        web_app.REPO_ROOT = original_repo_root

    assert manifest["research/models/premarket_scores_industry_v4_v1_1_latest_active.parquet"]["factor_snapshot_path"] == (
        "research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-25.parquet"
    )


def test_list_score_parquet_files_includes_manifest_metadata_when_available(tmp_path: Path) -> None:
    models_root = tmp_path / "research" / "models"
    models_root.mkdir(parents=True)
    score_path = models_root / "premarket_scores_industry_v4_v1_1_latest_active.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2026-03-24", "symbol": "AAA", "prediction": 0.1},
        ]
    ).to_parquet(score_path, index=False)
    (models_root / "score_source_manifest.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "strategy_id": "research_industry_v4_v1_1",
                        "score_source_id": "premarket_scores_industry_v4_v1_1_latest_active",
                        "scores_path": "research/models/premarket_scores_industry_v4_v1_1_latest_active.parquet",
                        "factor_spec_id": "research_industry_v4_v1_1",
                        "factor_snapshot_date": "2026-03-25",
                        "factor_snapshot_path": "research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-25.parquet",
                        "factor_panel_path": "research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-25.parquet",
                        "config_path": "configs/research_industry_v4_v1_1.toml",
                        "source_kind": "premarket_latest_active",
                        "description": "用于盘前生成的 latest active scores 数据源",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    original_manifest_path = web_app.SCORE_SOURCE_MANIFEST_PATH
    try:
        web_app.REPO_ROOT = tmp_path
        web_app.SCORE_SOURCE_MANIFEST_PATH = models_root / "score_source_manifest.json"
        files = list_score_parquet_files(models_root=models_root)
    finally:
        web_app.REPO_ROOT = original_repo_root
        web_app.SCORE_SOURCE_MANIFEST_PATH = original_manifest_path

    assert files == [
        {
            "path": "research/models/premarket_scores_industry_v4_v1_1_latest_active.parquet",
            "start_date": "2026-03-24",
            "end_date": "2026-03-24",
            "score_source_id": "premarket_scores_industry_v4_v1_1_latest_active",
            "strategy_id": "research_industry_v4_v1_1",
            "factor_spec_id": "research_industry_v4_v1_1",
            "factor_snapshot_date": "2026-03-25",
            "factor_snapshot_path": "research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-25.parquet",
            "factor_panel_path": "research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02/asof_2026-03-25.parquet",
            "config_path": "configs/research_industry_v4_v1_1.toml",
            "source_kind": "premarket_latest_active",
            "description": "用于盘前生成的 latest active scores 数据源",
            "supports_incremental_update": False,
        }
    ]


def test_previous_open_trade_date_uses_calendar(tmp_path: Path) -> None:
    calendar_path = tmp_path / "storage" / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-24", "is_open": True},
            {"trade_date": "2026-03-25", "is_open": True},
            {"trade_date": "2026-03-26", "is_open": True},
        ]
    ).to_parquet(calendar_path, index=False)

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        assert _previous_open_trade_date("2026-03-26", storage_root="storage") == "2026-03-25"
    finally:
        web_app.REPO_ROOT = original_repo_root


def test_next_open_trade_date_uses_calendar(tmp_path: Path) -> None:
    calendar_path = tmp_path / "storage" / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-24", "is_open": True},
            {"trade_date": "2026-03-25", "is_open": True},
            {"trade_date": "2026-03-26", "is_open": True},
        ]
    ).to_parquet(calendar_path, index=False)

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        assert _next_open_trade_date("2026-03-25", storage_root="storage") == "2026-03-26"
    finally:
        web_app.REPO_ROOT = original_repo_root


def test_build_simulation_readiness_reports_execution_date_and_factor_status(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo_strategy.toml").write_text(
        """
[storage]
root = "storage"

[factor_spec]
id = "demo_strategy"
universe_name = "tradable_core"
start_date = "2024-01-02"

[factors]
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-26"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )
    calendar_path = tmp_path / "storage" / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-25", "is_open": True},
            {"trade_date": "2026-03-26", "is_open": True},
            {"trade_date": "2026-03-27", "is_open": True},
        ]
    ).to_parquet(calendar_path, index=False)

    factor_panel_path = (
        tmp_path / "research" / "factors" / "demo_strategy" / "tradable_core" / "start_2024-01-02" / "asof_2026-03-26.parquet"
    )
    factor_panel_path.parent.mkdir(parents=True)
    pd.DataFrame([{"trade_date": "2026-03-26", "symbol": "AAA", "mom_5": 1.0}]).to_parquet(factor_panel_path, index=False)

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        payload = build_simulation_readiness(
            config_path="configs/demo_strategy.toml",
            signal_date="2026-03-26",
            repo_root=tmp_path,
        )
    finally:
        web_app.REPO_ROOT = original_repo_root

    assert payload["is_ready"] is True
    assert payload["execution_date"] == "2026-03-27"
    assert payload["factor_exists"] is True
    assert payload["scores_path"] == "research/models/walk_forward_demo_2026-03-26.parquet"
    assert payload["latest_signal_date"] == "2026-03-26"


def test_build_simulation_readiness_rejects_last_open_date_as_signal_date(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo_strategy.toml").write_text(
        """
[storage]
root = "storage"

[factor_spec]
id = "demo_strategy"
universe_name = "tradable_core"
start_date = "2024-01-02"

[factors]
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-27"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )
    calendar_path = tmp_path / "storage" / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-25", "is_open": True},
            {"trade_date": "2026-03-26", "is_open": True},
            {"trade_date": "2026-03-27", "is_open": True},
        ]
    ).to_parquet(calendar_path, index=False)

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        payload = build_simulation_readiness(
            config_path="configs/demo_strategy.toml",
            signal_date="2026-03-27",
            repo_root=tmp_path,
        )
    finally:
        web_app.REPO_ROOT = original_repo_root

    assert payload["is_ready"] is False
    assert payload["latest_signal_date"] == "2026-03-26"
    assert "请将信号截止日期选择到 2026-03-26 或更早" in payload["message"]


def test_run_simulation_job_uses_initial_entry_when_no_previous_state(tmp_path: Path, monkeypatch) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    config_path = config_root / "demo_strategy.toml"
    config_path.write_text(
        """
[storage]
root = "storage"

[factor_spec]
id = "demo_strategy"
universe_name = "tradable_core"
start_date = "2024-01-02"

[factors]
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-26"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    config = web_app.load_research_config(config_path)
    captured: dict[str, object] = {}

    def fake_train_walk_forward_single_date_from_config(**kwargs):
        captured["train_kwargs"] = kwargs
        return {"as_of_date": kwargs["as_of_date"], "test_month": kwargs["test_month"], "train_rows": 1, "scored_rows": 1}

    def fake_generate_strategy_state(strategy_config):
        captured["mode"] = strategy_config.mode
        captured["trade_date"] = strategy_config.trade_date
        captured["universe_name"] = strategy_config.universe_name
        Path(strategy_config.output_path).write_text(
            json.dumps(
                {
                    "summary": {
                        "signal_date": "2026-03-26",
                        "execution_date": "2026-03-27",
                        "current_position_count": 0,
                        "target_position_count": 2,
                    }
                }
            ),
            encoding="utf-8",
        )
        return {
            "summary": {
                "signal_date": "2026-03-26",
                "execution_date": "2026-03-27",
                "current_position_count": 0,
                "target_position_count": 2,
            }
        }

    monkeypatch.setattr(web_app, "train_walk_forward_single_date_from_config", fake_train_walk_forward_single_date_from_config)
    monkeypatch.setattr(web_app, "_ensure_simulation_factor_panel", lambda config, signal_date: ("research/factors/demo.parquet", True))
    monkeypatch.setattr(web_app, "_find_previous_simulation_run", lambda strategy_id, trade_date: None)
    monkeypatch.setattr(web_app, "generate_strategy_state", fake_generate_strategy_state)

    original_repo_root = web_app.REPO_ROOT
    original_simulation_latest_root = web_app.SIMULATION_LATEST_ROOT
    original_simulation_accounts_root = web_app.SIMULATION_ACCOUNTS_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        web_app.SIMULATION_LATEST_ROOT = tmp_path / "results" / "simulation_latest"
        web_app.SIMULATION_ACCOUNTS_ROOT = tmp_path / "results" / "simulation_accounts"
        app = BacktestWebApp(repo_root=tmp_path)
        output_dir = tmp_path / "results" / "simulation_runs" / "demo-run"
        app.job_store.create("demo-run", {"id": "demo-run", "type": "simulation", "status": "queued"})
        app._run_simulation_job(
            "demo-run",
            "demo_strategy",
            config,
            "2026-03-26",
            "2026-03-27",
            1_000_000.0,
            output_dir,
            "demo",
            "configs/demo_strategy.toml",
        )
    finally:
        web_app.REPO_ROOT = original_repo_root
        web_app.SIMULATION_LATEST_ROOT = original_simulation_latest_root
        web_app.SIMULATION_ACCOUNTS_ROOT = original_simulation_accounts_root

    assert captured["mode"] == "initial_entry"
    assert captured["trade_date"] == "2026-03-27"
    assert captured["universe_name"] == "tradable_core"
    assert captured["train_kwargs"] == {
        "config_path": config_path.as_posix(),
        "test_month": "2026-03",
        "as_of_date": "2026-03-26",
        "factor_panel_path": (tmp_path / "research" / "factors" / "demo.parquet").as_posix(),
    }
    account_state_path = tmp_path / "results" / "simulation_accounts" / "demo-run" / "strategy_state.json"
    account_meta_path = tmp_path / "results" / "simulation_accounts" / "demo-run" / "meta.json"
    account_events_path = tmp_path / "results" / "simulation_accounts" / "demo-run" / "events.csv"
    assert account_state_path.exists()
    assert account_meta_path.exists()
    assert account_events_path.exists()


def test_run_simulation_job_uses_explicit_previous_state_for_roll_forward(tmp_path: Path, monkeypatch) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    config_path = config_root / "demo_strategy.toml"
    config_path.write_text(
        """
[storage]
root = "storage"

[factor_spec]
id = "demo_strategy"
universe_name = "tradable_core"
start_date = "2024-01-02"

[factors]
output_path = "research/factors/demo.parquet"
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-27"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    config = web_app.load_research_config(config_path)
    captured: dict[str, object] = {}

    def fake_train_walk_forward_single_date_from_config(**kwargs):
        captured["train_kwargs"] = kwargs
        return {"as_of_date": kwargs["as_of_date"], "test_month": kwargs["test_month"], "train_rows": 1, "scored_rows": 1}

    def fake_generate_strategy_state(strategy_config):
        captured["mode"] = strategy_config.mode
        captured["trade_date"] = strategy_config.trade_date
        captured["universe_name"] = strategy_config.universe_name
        captured["previous_state_path"] = strategy_config.previous_state_path
        Path(strategy_config.output_path).write_text(
            json.dumps(
                {
                    "summary": {
                        "signal_date": "2026-03-27",
                        "execution_date": "2026-03-30",
                        "current_position_count": 6,
                        "target_position_count": 6,
                    }
                }
            ),
            encoding="utf-8",
        )
        return {
            "summary": {
                "signal_date": "2026-03-27",
                "execution_date": "2026-03-30",
                "current_position_count": 6,
                "target_position_count": 6,
            }
        }

    monkeypatch.setattr(web_app, "train_walk_forward_single_date_from_config", fake_train_walk_forward_single_date_from_config)
    monkeypatch.setattr(web_app, "_ensure_simulation_factor_panel", lambda config, signal_date: ("research/factors/demo.parquet", False))
    monkeypatch.setattr(web_app, "_find_previous_simulation_run", lambda strategy_id, trade_date: (_ for _ in ()).throw(AssertionError("should not auto-resolve previous run")))
    monkeypatch.setattr(web_app, "generate_strategy_state", fake_generate_strategy_state)

    original_repo_root = web_app.REPO_ROOT
    original_simulation_latest_root = web_app.SIMULATION_LATEST_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        web_app.SIMULATION_LATEST_ROOT = tmp_path / "results" / "simulation_latest"
        app = BacktestWebApp(repo_root=tmp_path)
        output_dir = tmp_path / "results" / "simulation_runs" / "demo-roll-forward"
        previous_state_path = (tmp_path / "results" / "simulation_runs" / "older" / "strategy_state.json").as_posix()
        Path(previous_state_path).parent.mkdir(parents=True, exist_ok=True)
        Path(previous_state_path).write_text(json.dumps({"next_state": {"as_of_trade_date": "2026-03-27"}}), encoding="utf-8")
        app.job_store.create("demo-roll-forward", {"id": "demo-roll-forward", "type": "simulation", "status": "queued"})
        app._run_simulation_job(
            "demo-roll-forward",
            "demo_strategy",
            config,
            "2026-03-27",
            "2026-03-30",
            1_000_000.0,
            output_dir,
            "demo-roll-forward",
            "configs/demo_strategy.toml",
            previous_state_path,
            "older-run",
        )
    finally:
        web_app.REPO_ROOT = original_repo_root
        web_app.SIMULATION_LATEST_ROOT = original_simulation_latest_root

    assert captured["mode"] == "continue"
    assert captured["trade_date"] == "2026-03-30"
    assert captured["universe_name"] == "tradable_core"
    assert captured["previous_state_path"] == previous_state_path


def test_submit_simulation_roll_forward_uses_selected_run_as_previous_state(tmp_path: Path, monkeypatch) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo_strategy.toml").write_text(
        """
[storage]
root = "storage"

[factor_spec]
id = "demo_strategy"
universe_name = "tradable_core"
start_date = "2024-01-02"

[factors]
output_path = "research/factors/demo.parquet"
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-27"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )
    calendar_path = tmp_path / "storage" / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-27", "is_open": True},
            {"trade_date": "2026-03-30", "is_open": True},
            {"trade_date": "2026-03-31", "is_open": True},
        ]
    ).to_parquet(calendar_path, index=False)
    run_dir = tmp_path / "results" / "simulation_runs" / "selected-run"
    run_dir.mkdir(parents=True)
    (run_dir / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"signal_date": "2026-03-26", "execution_date": "2026-03-27"},
                "strategy_config": {"initial_cash": 100000.0},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n",
        encoding="utf-8",
    )
    (run_dir / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n",
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "demo-run",
                "strategy_id": "demo_strategy",
                "config_path": "configs/demo_strategy.toml",
                "created_at": "2026-03-27T21:31:50",
            }
        ),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    submitted: dict[str, object] = {}

    class DummyExecutor:
        def submit(self, fn, *args):
            submitted["fn"] = fn.__name__
            submitted["args"] = args
            return None

    original_repo_root = web_app.REPO_ROOT
    original_simulation_runs_root = web_app.SIMULATION_RUNS_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        web_app.SIMULATION_RUNS_ROOT = tmp_path / "results" / "simulation_runs"
        app = BacktestWebApp(repo_root=tmp_path)
        app.executor = DummyExecutor()
        job = app.submit_simulation_roll_forward("selected-run")
    finally:
        web_app.REPO_ROOT = original_repo_root
        web_app.SIMULATION_RUNS_ROOT = original_simulation_runs_root

    assert job["signal_date"] == "2026-03-27"
    assert job["trade_date"] == "2026-03-30"
    assert job["previous_run_id"] == "selected-run"
    assert job["source_kind"] == "roll_forward"
    assert job["previous_state_path"].endswith("selected-run/strategy_state.json")
    assert submitted["fn"] == "_run_simulation_job"
    assert submitted["args"][-3].endswith("selected-run/strategy_state.json")
    assert submitted["args"][-2] == "selected-run"
    assert submitted["args"][-1] == "selected-run"


def test_submit_simulation_plan_creates_new_account_without_reusing_previous_run(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo_strategy.toml").write_text(
        """
[storage]
root = "storage"

[factor_spec]
id = "demo_strategy"
universe_name = "tradable_core"
start_date = "2024-01-02"

[factors]
output_path = "research/factors/demo.parquet"
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-27"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )
    calendar_path = tmp_path / "storage" / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-27", "is_open": True},
            {"trade_date": "2026-03-30", "is_open": True},
        ]
    ).to_parquet(calendar_path, index=False)
    previous_run_dir = tmp_path / "results" / "simulation_runs" / "older-run"
    previous_run_dir.mkdir(parents=True)
    (previous_run_dir / "strategy_state.json").write_text(
        json.dumps({"summary": {"signal_date": "2026-03-26", "execution_date": "2026-03-27"}}),
        encoding="utf-8",
    )
    (previous_run_dir / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n",
        encoding="utf-8",
    )
    (previous_run_dir / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n",
        encoding="utf-8",
    )
    (previous_run_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "older-run",
                "account_id": "old-account",
                "strategy_id": "demo_strategy",
                "config_path": "configs/demo_strategy.toml",
                "created_at": "2026-03-27T21:31:50",
            }
        ),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    submitted: dict[str, object] = {}

    class DummyExecutor:
        def submit(self, fn, *args):
            submitted["fn"] = fn.__name__
            submitted["args"] = args
            return None

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        app = BacktestWebApp(repo_root=tmp_path)
        app.executor = DummyExecutor()
        job = app.submit_simulation_plan(
            config_path="configs/demo_strategy.toml",
            signal_date="2026-03-27",
            initial_cash=500000.0,
            label="fresh-account",
        )
    finally:
        web_app.REPO_ROOT = original_repo_root

    assert job["account_id"] == job["id"]
    assert job["previous_run_id"] == ""
    assert job["previous_state_path"] == ""
    assert submitted["fn"] == "_run_simulation_plan_job"
    assert submitted["args"][-2] == ""
    assert submitted["args"][-1] == ""


def test_submit_simulation_next_plan_uses_latest_executed_run_for_account(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo_strategy.toml").write_text(
        """
[storage]
root = "storage"

[factor_spec]
id = "demo_strategy"
universe_name = "tradable_core"
start_date = "2024-01-02"

[factors]
output_path = "research/factors/demo.parquet"
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-30"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )
    calendar_path = tmp_path / "storage" / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-27", "is_open": True},
            {"trade_date": "2026-03-30", "is_open": True},
            {"trade_date": "2026-03-31", "is_open": True},
        ]
    ).to_parquet(calendar_path, index=False)
    plan_dir = tmp_path / "results" / "simulation_plans" / "plan-one"
    plan_dir.mkdir(parents=True)
    (plan_dir / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"signal_date": "2026-03-26", "execution_date": "2026-03-27"},
                "strategy_config": {"storage_root": "storage", "initial_cash": 100000.0},
            }
        ),
        encoding="utf-8",
    )
    (plan_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "account-one",
                "account_id": "account-one",
                "strategy_id": "demo_strategy",
                "config_path": "configs/demo_strategy.toml",
                "scores_path": "research/models/walk_forward_demo_2026-03-26.parquet",
                "executed_run_id": "run-one",
                "created_at": "2026-03-27T09:00:00",
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "results" / "simulation_runs" / "run-one"
    run_dir.mkdir(parents=True)
    (run_dir / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"signal_date": "2026-03-26", "execution_date": "2026-03-27"},
                "strategy_config": {"initial_cash": 100000.0},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n",
        encoding="utf-8",
    )
    (run_dir / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n",
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "account-one",
                "account_id": "account-one",
                "strategy_id": "demo_strategy",
                "config_path": "configs/demo_strategy.toml",
                "created_at": "2026-03-27T15:00:00",
            }
        ),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    submitted: dict[str, object] = {}

    class DummyExecutor:
        def submit(self, fn, *args):
            submitted["fn"] = fn.__name__
            submitted["args"] = args
            return None

    original_repo_root = web_app.REPO_ROOT
    original_simulation_runs_root = web_app.SIMULATION_RUNS_ROOT
    original_simulation_plans_root = web_app.SIMULATION_PLANS_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        web_app.SIMULATION_RUNS_ROOT = tmp_path / "results" / "simulation_runs"
        web_app.SIMULATION_PLANS_ROOT = tmp_path / "results" / "simulation_plans"
        app = BacktestWebApp(repo_root=tmp_path)
        app.executor = DummyExecutor()
        job = app.submit_simulation_next_plan("plan-one")
    finally:
        web_app.REPO_ROOT = original_repo_root
        web_app.SIMULATION_RUNS_ROOT = original_simulation_runs_root
        web_app.SIMULATION_PLANS_ROOT = original_simulation_plans_root

    assert job["type"] == "simulation_plan"
    assert job["signal_date"] == "2026-03-27"
    assert job["trade_date"] == "2026-03-30"
    assert job["account_id"] == "account-one"
    assert job["previous_run_id"] == "run-one"
    assert job["source_kind"] == "simulation_plan_roll_forward"
    assert submitted["fn"] == "_run_simulation_plan_job"
    assert submitted["args"][-2].endswith("run-one/strategy_state.json")
    assert submitted["args"][-1] == "run-one"


def test_submit_simulation_next_plan_allows_skipping_execute_when_plan_has_no_pending_orders(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo_strategy.toml").write_text(
        """
[storage]
root = "storage"

[factor_spec]
id = "demo_strategy"
universe_name = "tradable_core"
start_date = "2024-01-02"

[factors]
output_path = "research/factors/demo.parquet"
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-31"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )
    calendar_path = tmp_path / "storage" / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-27", "is_open": True},
            {"trade_date": "2026-03-30", "is_open": True},
            {"trade_date": "2026-03-31", "is_open": True},
        ]
    ).to_parquet(calendar_path, index=False)
    plan_dir = tmp_path / "results" / "simulation_plans" / "account-one" / "plans" / "plan-no-exec"
    plan_dir.mkdir(parents=True)
    (plan_dir / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"signal_date": "2026-03-27", "execution_date": "2026-03-30"},
                "strategy_config": {"storage_root": "storage", "initial_cash": 100000.0},
                "next_state": {
                    "as_of_trade_date": "2026-03-30",
                    "execution_pending": False,
                    "planned_target_weights": {},
                },
            }
        ),
        encoding="utf-8",
    )
    (plan_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "account-one",
                "account_id": "account-one",
                "strategy_id": "demo_strategy",
                "config_path": "configs/demo_strategy.toml",
                "scores_path": "research/models/walk_forward_demo_2026-03-27.parquet",
                "previous_run_id": "run-one",
                "created_at": "2026-03-30T09:00:00",
            }
        ),
        encoding="utf-8",
    )
    account_dir = tmp_path / "results" / "simulation_accounts" / "account-one"
    account_dir.mkdir(parents=True)
    (account_dir / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"signal_date": "2026-03-27", "execution_date": "2026-03-30"},
                "strategy_config": {"storage_root": "storage", "initial_cash": 100000.0},
                "next_state": {
                    "as_of_trade_date": "2026-03-30",
                    "execution_pending": False,
                    "planned_target_weights": {},
                },
            }
        ),
        encoding="utf-8",
    )
    (account_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "account-one",
                "account_id": "account-one",
                "strategy_id": "demo_strategy",
                "config_path": "configs/demo_strategy.toml",
            }
        ),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    submitted: dict[str, object] = {}

    class DummyExecutor:
        def submit(self, fn, *args):
            submitted["fn"] = fn.__name__
            submitted["args"] = args
            return None

    original_repo_root = web_app.REPO_ROOT
    original_simulation_runs_root = web_app.SIMULATION_RUNS_ROOT
    original_simulation_plans_root = web_app.SIMULATION_PLANS_ROOT
    original_simulation_accounts_root = web_app.SIMULATION_ACCOUNTS_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        web_app.SIMULATION_RUNS_ROOT = tmp_path / "results" / "simulation_runs"
        web_app.SIMULATION_PLANS_ROOT = tmp_path / "results" / "simulation_plans"
        web_app.SIMULATION_ACCOUNTS_ROOT = tmp_path / "results" / "simulation_accounts"
        app = BacktestWebApp(repo_root=tmp_path)
        app.executor = DummyExecutor()
        detail = load_simulation_plan_detail("plan-no-exec", results_root=web_app.SIMULATION_PLANS_ROOT)
        job = app.submit_simulation_next_plan("plan-no-exec")
    finally:
        web_app.REPO_ROOT = original_repo_root
        web_app.SIMULATION_RUNS_ROOT = original_simulation_runs_root
        web_app.SIMULATION_PLANS_ROOT = original_simulation_plans_root
        web_app.SIMULATION_ACCOUNTS_ROOT = original_simulation_accounts_root

    assert detail["execution_pending"] is False
    assert detail["execution_ready"] is False
    assert detail["next_plan_ready"] is True
    assert job["signal_date"] == "2026-03-30"
    assert job["trade_date"] == "2026-03-31"
    assert job["previous_run_id"] == "run-one"
    assert submitted["fn"] == "_run_simulation_plan_job"
    assert submitted["args"][-2].endswith("plan-no-exec/strategy_state.json")
    assert submitted["args"][-1] == "run-one"


def test_submit_simulation_execute_plan_rejects_when_plan_has_no_pending_orders(tmp_path: Path) -> None:
    plan_dir = tmp_path / "results" / "simulation_plans" / "account-one" / "plans" / "plan-no-exec"
    plan_dir.mkdir(parents=True)
    (plan_dir / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"signal_date": "2026-03-27", "execution_date": "2026-03-30"},
                "strategy_config": {"storage_root": "storage", "initial_cash": 100000.0},
                "next_state": {
                    "as_of_trade_date": "2026-03-30",
                    "execution_pending": False,
                    "planned_target_weights": {},
                },
            }
        ),
        encoding="utf-8",
    )
    (plan_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "account-one",
                "account_id": "account-one",
                "strategy_id": "demo_strategy",
                "config_path": "configs/demo_strategy.toml",
                "created_at": "2026-03-30T09:00:00",
            }
        ),
        encoding="utf-8",
    )
    account_dir = tmp_path / "results" / "simulation_accounts" / "account-one"
    account_dir.mkdir(parents=True)
    (account_dir / "strategy_state.json").write_text((plan_dir / "strategy_state.json").read_text(encoding="utf-8"), encoding="utf-8")
    (account_dir / "meta.json").write_text((plan_dir / "meta.json").read_text(encoding="utf-8"), encoding="utf-8")

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    original_simulation_plans_root = web_app.SIMULATION_PLANS_ROOT
    original_simulation_accounts_root = web_app.SIMULATION_ACCOUNTS_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        web_app.SIMULATION_PLANS_ROOT = tmp_path / "results" / "simulation_plans"
        web_app.SIMULATION_ACCOUNTS_ROOT = tmp_path / "results" / "simulation_accounts"
        app = BacktestWebApp(repo_root=tmp_path)
        try:
            app.submit_simulation_execute_plan("plan-no-exec")
            raise AssertionError("expected submit_simulation_execute_plan to fail")
        except ValueError as exc:
            assert "无需模拟下单" in str(exc)
    finally:
        web_app.REPO_ROOT = original_repo_root
        web_app.SIMULATION_PLANS_ROOT = original_simulation_plans_root
        web_app.SIMULATION_ACCOUNTS_ROOT = original_simulation_accounts_root


def test_build_paper_readiness_reports_not_ready_when_scores_lag_signal_date(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo_strategy.toml").write_text(
        """
[storage]
root = "storage"

[factors]
output_path = "research/factors/demo.parquet"
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-10"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )
    calendar_path = tmp_path / "storage" / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-24", "is_open": True},
            {"trade_date": "2026-03-25", "is_open": True},
            {"trade_date": "2026-03-26", "is_open": True},
        ]
    ).to_parquet(calendar_path, index=False)
    scores_path = tmp_path / "research" / "models" / "premarket_scores_demo.parquet"
    scores_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-24", "symbol": "AAA", "prediction": 0.1},
        ]
    ).to_parquet(scores_path, index=False)

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        payload = build_paper_readiness(
            config_path="configs/demo_strategy.toml",
            trade_date="2026-03-26",
            scores_path="research/models/premarket_scores_demo.parquet",
            repo_root=tmp_path,
        )
    finally:
        web_app.REPO_ROOT = original_repo_root

    assert payload["required_signal_date"] == "2026-03-25"
    assert payload["score_end_date"] == "2026-03-24"
    assert payload["is_ready"] is False
    assert payload["missing_score_days"] == 1


def test_build_paper_readiness_reports_ready_when_scores_cover_signal_date(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo_strategy.toml").write_text(
        """
[storage]
root = "storage"

[factors]
output_path = "research/factors/demo.parquet"
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-10"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )
    calendar_path = tmp_path / "storage" / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-24", "is_open": True},
            {"trade_date": "2026-03-25", "is_open": True},
            {"trade_date": "2026-03-26", "is_open": True},
        ]
    ).to_parquet(calendar_path, index=False)
    scores_path = tmp_path / "research" / "models" / "premarket_scores_demo.parquet"
    scores_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-24", "symbol": "AAA", "prediction": 0.1},
            {"trade_date": "2026-03-25", "symbol": "AAA", "prediction": 0.2},
        ]
    ).to_parquet(scores_path, index=False)

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        payload = build_paper_readiness(
            config_path="configs/demo_strategy.toml",
            trade_date="2026-03-26",
            scores_path="research/models/premarket_scores_demo.parquet",
            repo_root=tmp_path,
        )
    finally:
        web_app.REPO_ROOT = original_repo_root

    assert payload["required_signal_date"] == "2026-03-25"
    assert payload["score_end_date"] == "2026-03-25"
    assert payload["is_ready"] is True
    assert payload["missing_score_days"] == 0


def test_build_paper_readiness_reports_not_ready_for_non_open_trade_date(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo_strategy.toml").write_text(
        """
[storage]
root = "storage"

[factors]
output_path = "research/factors/demo.parquet"
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-10"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )
    calendar_path = tmp_path / "storage" / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-24", "is_open": True},
            {"trade_date": "2026-03-25", "is_open": True},
            {"trade_date": "2026-03-26", "is_open": True},
        ]
    ).to_parquet(calendar_path, index=False)
    scores_path = tmp_path / "research" / "models" / "premarket_scores_demo.parquet"
    scores_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-24", "symbol": "AAA", "prediction": 0.1},
            {"trade_date": "2026-03-25", "symbol": "AAA", "prediction": 0.2},
        ]
    ).to_parquet(scores_path, index=False)

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        payload = build_paper_readiness(
            config_path="configs/demo_strategy.toml",
            trade_date="2026-03-27",
            scores_path="research/models/premarket_scores_demo.parquet",
            repo_root=tmp_path,
        )
    finally:
        web_app.REPO_ROOT = original_repo_root

    assert payload["is_ready"] is False
    assert payload["required_signal_date"] == ""
    assert payload["missing_score_days"] == 0
    assert "不是交易日" in payload["message"]


def test_fill_scores_to_signal_date_appends_missing_latest_inference_rows(tmp_path: Path, monkeypatch) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo_strategy.toml").write_text(
        """
[storage]
root = "storage"

[factors]
output_path = "research/factors/demo.parquet"
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-10"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )
    calendar_path = tmp_path / "storage" / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-24", "is_open": True},
            {"trade_date": "2026-03-25", "is_open": True},
            {"trade_date": "2026-03-26", "is_open": True},
        ]
    ).to_parquet(calendar_path, index=False)

    scores_path = tmp_path / "research" / "models" / "premarket_scores_demo.parquet"
    scores_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-24", "symbol": "AAA", "prediction": 0.1},
        ]
    ).to_parquet(scores_path, index=False)

    factor_panel_path = tmp_path / "research" / "factors" / "demo_latest.parquet"
    factor_panel_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"trade_date": "2026-03-25", "symbol": "AAA", "mom_5": 1.0},
        ]
    ).to_parquet(factor_panel_path, index=False)

    manifest_path = tmp_path / "research" / "models" / "score_source_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "strategy_id": "demo_strategy",
                        "score_source_id": "premarket_scores_demo",
                        "scores_path": "research/models/premarket_scores_demo.parquet",
                        "factor_snapshot_path": "research/factors/demo_latest.parquet",
                        "factor_panel_path": "research/factors/demo_latest.parquet",
                        "config_path": "configs/demo_strategy.toml",
                        "label_column": "industry_excess_fwd_return_5",
                        "train_window_months": 12,
                        "source_kind": "premarket_latest_active",
                        "supports_incremental_update": True,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fake_walk_forward_as_of_date(config) -> dict[str, object]:
        pd.DataFrame(
            [
                {
                    "trade_date": config.as_of_date,
                    "symbol": "AAA",
                    "label": 0.02,
                    "prediction": 0.3,
                    "train_end_date": "2026-03-24",
                    "as_of_date": config.as_of_date,
                }
            ]
        ).to_parquet(config.output_scores_path, index=False)
        Path(config.output_metrics_path).write_text("{}", encoding="utf-8")
        return {"as_of_date": config.as_of_date}

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    original_manifest_path = web_app.SCORE_SOURCE_MANIFEST_PATH
    monkeypatch.setattr(web_app, "train_lightgbm_walk_forward_as_of_date", fake_walk_forward_as_of_date)
    try:
        web_app.REPO_ROOT = tmp_path
        web_app.SCORE_SOURCE_MANIFEST_PATH = manifest_path
        payload = fill_scores_to_signal_date(
            config_path="configs/demo_strategy.toml",
            trade_date="2026-03-26",
            scores_path="research/models/premarket_scores_demo.parquet",
            repo_root=tmp_path,
        )
    finally:
        web_app.REPO_ROOT = original_repo_root
        web_app.SCORE_SOURCE_MANIFEST_PATH = original_manifest_path

    assert payload["filled_dates"] == ["2026-03-25"]
    assert payload["filled_count"] == 1
    merged = pd.read_parquet(scores_path).sort_values(["trade_date", "symbol"])
    assert merged["trade_date"].dt.date.astype(str).tolist() == ["2026-03-24", "2026-03-25"]
    assert payload["score_end_date"] == "2026-03-25"


def test_resolve_selected_scores_path_prefers_explicit_user_choice(tmp_path: Path) -> None:
    models_root = tmp_path / "research" / "models"
    models_root.mkdir(parents=True)
    score_path = models_root / "custom_scores.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2026-03-24", "symbol": "AAA", "prediction": 0.1},
        ]
    ).to_parquet(score_path, index=False)

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        resolved_path, manifest, source_kind = _resolve_selected_scores_path(
            "research/models/custom_scores.parquet",
            "demo_strategy",
            "research/models/fallback_scores.parquet",
        )
    finally:
        web_app.REPO_ROOT = original_repo_root

    assert resolved_path == "research/models/custom_scores.parquet"
    assert manifest == {}
    assert source_kind == "user_selected"


def test_resolve_selected_scores_path_raises_for_missing_explicit_file(tmp_path: Path) -> None:
    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        try:
            _resolve_selected_scores_path(
                "research/models/missing_scores.parquet",
                "demo_strategy",
                "research/models/fallback_scores.parquet",
            )
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError as exc:
            assert str(exc) == "score file not found: research/models/missing_scores.parquet"
    finally:
        web_app.REPO_ROOT = original_repo_root


def test_list_strategy_presets_prefers_latest_manifest_when_present(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo_strategy.toml").write_text(
        """
[storage]
root = "storage"

[factors]
output_path = "research/factors/demo.parquet"
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-10"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )
    latest_dir = tmp_path / "research" / "models" / "latest" / "demo_strategy"
    latest_dir.mkdir(parents=True)
    (latest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_scores_path": "research/models/latest_scores_demo_strategy_2026-03-25.parquet",
                "scores_path": "research/models/latest/demo_strategy/scores.parquet",
                "signal_date": "2026-03-25",
                "execution_date": "2026-03-26",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        presets = list_strategy_presets(config_root=config_root)
    finally:
        web_app.REPO_ROOT = original_repo_root

    assert presets[0].paper_score_output_path == "research/models/latest_scores_demo_strategy_2026-03-25.parquet"
    assert presets[0].paper_source_kind == "latest_manifest_source"
    assert presets[0].latest_signal_date == "2026-03-25"


def test_list_strategy_presets_merges_history_and_latest_score_ranges(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo_strategy.toml").write_text(
        """
[storage]
root = "storage"

[factors]
output_path = "research/factors/demo.parquet"
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-10"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )
    history_scores = tmp_path / "research" / "models" / "walk_forward_demo.parquet"
    history_scores.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"trade_date": "2025-01-02", "symbol": "AAA", "prediction": 1.0},
            {"trade_date": "2025-01-03", "symbol": "AAA", "prediction": 1.1},
        ]
    ).to_parquet(history_scores, index=False)
    latest_scores = tmp_path / "research" / "models" / "latest_scores_demo_2026-03-25.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2026-03-25", "symbol": "AAA", "prediction": 2.0},
        ]
    ).to_parquet(latest_scores, index=False)
    latest_dir = tmp_path / "research" / "models" / "latest" / "demo_strategy"
    latest_dir.mkdir(parents=True)
    (latest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source_scores_path": "research/models/latest_scores_demo_2026-03-25.parquet",
                "scores_path": "research/models/latest/demo_strategy/scores.parquet",
                "signal_date": "2026-03-25",
                "execution_date": "2026-03-26",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        presets = list_strategy_presets(config_root=config_root)
    finally:
        web_app.REPO_ROOT = original_repo_root

    assert presets[0].paper_source_kind == "merged_history"
    assert presets[0].paper_score_start_date == "2025-01-02"
    assert presets[0].paper_score_end_date == "2026-03-25"


def test_load_run_detail_reads_summary_equity_and_trades(tmp_path: Path) -> None:
    result_dir = tmp_path / "demo_run"
    result_dir.mkdir(parents=True)
    (result_dir / "summary.json").write_text(
        json.dumps({"total_return": 0.1, "trade_count": 2}, ensure_ascii=False),
        encoding="utf-8",
    )
    with (result_dir / "equity_curve.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trade_date", "equity"])
        writer.writerow(["2025-01-02", "1000000"])
    with (result_dir / "trades.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["trade_date", "symbol", "side", "quantity", "price", "amount", "commission", "tax", "slippage", "status", "reason"]
        )
        writer.writerow(["2025-01-02", "AAA", "BUY", "100", "10", "1000", "1", "0", "0.5", "filled", "rebalance_entry"])
    bars_path = tmp_path / "daily.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-02", "symbol": "AAA", "close_adj": 10.0, "close": 10.0, "is_suspended": False},
            {"trade_date": "2025-01-02", "symbol": "BBB", "close_adj": 20.0, "close": 20.0, "is_suspended": False},
        ]
    ).assign(trade_date=lambda df: pd.to_datetime(df["trade_date"])).to_parquet(bars_path, index=False)

    detail = load_run_detail(
        "demo_run",
        results_root=tmp_path,
        bars_path=bars_path,
        benchmark_path=tmp_path / "missing_hs300.parquet",
    )

    assert detail["summary"]["total_return"] == 0.1
    assert detail["equity_curve"][0]["equity"] == 1000000.0
    assert detail["trades"][0]["symbol"] == "AAA"
    assert detail["backtest_start_date"] == "2025-01-02"
    assert detail["backtest_end_date"] == "2025-01-02"
    assert detail["benchmark_label"] == "A股等权基准"


def test_load_run_detail_prefers_cached_hs300_benchmark(tmp_path: Path) -> None:
    result_dir = tmp_path / "demo_run"
    result_dir.mkdir(parents=True)
    (result_dir / "summary.json").write_text(json.dumps({"total_return": 0.1}), encoding="utf-8")
    (result_dir / "equity_curve.csv").write_text(
        "trade_date,equity\n2025-01-02,1000000\n2025-01-03,1010000\n",
        encoding="utf-8",
    )
    (result_dir / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n",
        encoding="utf-8",
    )
    bars_path = tmp_path / "daily.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-02", "symbol": "AAA", "close_adj": 10.0, "close": 10.0, "is_suspended": False},
            {"trade_date": "2025-01-03", "symbol": "AAA", "close_adj": 10.5, "close": 10.5, "is_suspended": False},
        ]
    ).assign(trade_date=lambda df: pd.to_datetime(df["trade_date"])).to_parquet(bars_path, index=False)
    benchmark_path = tmp_path / "000300.SH.parquet"
    pd.DataFrame(
        [
            {"symbol": "000300.SH", "trade_date": "2025-01-02", "close": 4000.0},
            {"symbol": "000300.SH", "trade_date": "2025-01-03", "close": 4040.0},
        ]
    ).assign(trade_date=lambda df: pd.to_datetime(df["trade_date"])).to_parquet(benchmark_path, index=False)

    detail = load_run_detail("demo_run", results_root=tmp_path, bars_path=bars_path, benchmark_path=benchmark_path)

    assert detail["benchmark_label"] == "沪深300"
    assert detail["benchmark_curve"][1]["equity"] == 1010000.0


def test_load_run_detail_reads_strategy_state_snapshot(tmp_path: Path) -> None:
    result_dir = tmp_path / "demo_run"
    result_dir.mkdir(parents=True)
    (result_dir / "summary.json").write_text(json.dumps({"total_return": 0.1}), encoding="utf-8")
    (result_dir / "equity_curve.csv").write_text("trade_date,equity\n2025-01-02,1000000\n", encoding="utf-8")
    (result_dir / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n",
        encoding="utf-8",
    )
    (result_dir / "strategy_state_latest.json").write_text(
            json.dumps(
                {
                    "plan": {"selected_symbols": ["AAA", "BBB"]},
                    "strategy_config": {
                        "scores_path": "research/models/demo_scores.parquet",
                        "initial_cash": 1000000,
                    },
                    "next_state": {
                        "positions": [
                            {"symbol": "AAA", "weight": 0.5, "market_value": 500000},
                            {"symbol": "BBB", "weight": 0.5, "market_value": 500000},
                    ]
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    scores_path = tmp_path / "research" / "models" / "demo_scores.parquet"
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"trade_date": "2025-01-02", "symbol": "AAA", "prediction": 0.1},
            {"trade_date": "2025-01-03", "symbol": "AAA", "prediction": 0.2},
        ]
    ).assign(trade_date=lambda df: pd.to_datetime(df["trade_date"])).to_parquet(scores_path, index=False)

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        detail = load_run_detail(
            "demo_run",
            results_root=tmp_path,
            bars_path=tmp_path / "missing.parquet",
            benchmark_path=tmp_path / "missing.parquet",
        )
    finally:
        web_app.REPO_ROOT = original_repo_root

    assert detail["strategy_state"]["plan"]["selected_symbols"] == ["AAA", "BBB"]
    assert detail["scores_path"] == "research/models/demo_scores.parquet"
    assert detail["score_start_date"] == "2025-01-02"
    assert detail["score_end_date"] == "2025-01-03"


def test_list_paper_trade_summaries_filters_valid_snapshot_dirs(tmp_path: Path) -> None:
    valid = tmp_path / "paper-1"
    valid.mkdir()
    (valid / "strategy_state.json").write_text(
        json.dumps({"summary": {"execution_date": "2025-01-10", "current_position_count": 1, "target_position_count": 2}}),
        encoding="utf-8",
    )
    (valid / "meta.json").write_text(
        json.dumps({"name": "demo paper", "config_path": "configs/demo.toml"}, ensure_ascii=False),
        encoding="utf-8",
    )
    invalid = tmp_path / "paper-2"
    invalid.mkdir()

    runs = list_paper_trade_summaries(results_root=tmp_path)

    assert [run["id"] for run in runs] == ["paper-1"]
    assert runs[0]["name"] == "demo paper"
    assert runs[0]["trade_date"] == "2025-01-10"


def test_load_paper_trade_detail_reads_snapshot_and_meta(tmp_path: Path) -> None:
    result_dir = tmp_path / "paper-demo"
    result_dir.mkdir(parents=True)
    (result_dir / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"execution_date": "2025-01-10"},
                "plan": {"selected_symbols": ["AAA", "BBB"]},
                "pre_open": {"positions": []},
                "next_state": {"positions": []},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (result_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "paper demo",
                "config_path": "configs/research_industry_v4_v1_1.toml",
                "created_at": "2025-01-10T09:00:00",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    detail = load_paper_trade_detail("paper-demo", results_root=tmp_path)

    assert detail["name"] == "paper demo"
    assert detail["config_path"] == "configs/research_industry_v4_v1_1.toml"
    assert detail["strategy_state"]["plan"]["selected_symbols"] == ["AAA", "BBB"]


def test_load_latest_paper_snapshot_reads_manifest_target(tmp_path: Path) -> None:
    latest_dir = tmp_path / "research" / "models" / "latest" / "demo_strategy"
    latest_dir.mkdir(parents=True)
    (latest_dir / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"signal_date": "2025-01-09", "execution_date": "2025-01-10"},
                "plan": {"selected_symbols": ["AAA"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (latest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "strategy_id": "demo_strategy",
                "signal_date": "2025-01-09",
                "execution_date": "2025-01-10",
                "scores_path": "research/models/latest/demo_strategy/scores.parquet",
                "strategy_state_path": "research/models/latest/demo_strategy/strategy_state.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        detail = load_latest_paper_snapshot("demo_strategy")
    finally:
        web_app.REPO_ROOT = original_repo_root

    assert detail["paper_source_kind"] == "latest_manifest"
    assert detail["scores_path"] == "research/models/latest/demo_strategy/scores.parquet"
    assert detail["strategy_state"]["plan"]["selected_symbols"] == ["AAA"]


def test_list_simulation_summaries_and_load_detail(tmp_path: Path) -> None:
    result_dir = tmp_path / "simulation-demo"
    result_dir.mkdir(parents=True)
    (result_dir / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"execution_date": "2025-01-10", "signal_date": "2025-01-09"},
                "plan": {"selected_symbols": ["AAA"]},
                "pre_open": {"positions": []},
                "next_state": {"positions": [], "pending_orders": [], "as_of_trade_date": "2025-01-10"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (result_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": "simulation demo",
                "strategy_id": "demo_strategy",
                "config_path": "configs/demo.toml",
                "scores_path": "research/models/demo_scores.parquet",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (result_dir / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n",
        encoding="utf-8",
    )
    (result_dir / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n",
        encoding="utf-8",
    )

    runs = list_simulation_summaries(results_root=tmp_path)
    detail = load_simulation_detail("simulation-demo", results_root=tmp_path)

    assert runs[0]["strategy_id"] == "demo_strategy"
    assert detail["scores_path"] == "research/models/demo_scores.parquet"
    assert detail["strategy_state"]["plan"]["selected_symbols"] == ["AAA"]


def test_list_simulation_summaries_keeps_only_latest_run_per_account(tmp_path: Path) -> None:
    older = tmp_path / "older"
    older.mkdir(parents=True)
    (older / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"execution_date": "2025-01-09", "signal_date": "2025-01-08"},
                "plan": {"selected_symbols": ["AAA"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (older / "meta.json").write_text(
        json.dumps(
            {
                "name": "demo account",
                "strategy_id": "demo_strategy",
                "config_path": "configs/demo.toml",
                "account_id": "account-1",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (older / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n",
        encoding="utf-8",
    )
    (older / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n",
        encoding="utf-8",
    )

    newer = tmp_path / "newer"
    newer.mkdir(parents=True)
    (newer / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"execution_date": "2025-01-10", "signal_date": "2025-01-09"},
                "plan": {"selected_symbols": ["BBB"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (newer / "meta.json").write_text(
        json.dumps(
            {
                "name": "demo account",
                "strategy_id": "demo_strategy",
                "config_path": "configs/demo.toml",
                "account_id": "account-1",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (newer / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n",
        encoding="utf-8",
    )
    (newer / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n",
        encoding="utf-8",
    )

    separate = tmp_path / "separate"
    separate.mkdir(parents=True)
    (separate / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"execution_date": "2025-01-08", "signal_date": "2025-01-07"},
                "plan": {"selected_symbols": ["CCC"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (separate / "meta.json").write_text(
        json.dumps(
            {
                "name": "another account",
                "strategy_id": "demo_strategy",
                "config_path": "configs/demo.toml",
                "account_id": "account-2",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (separate / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n",
        encoding="utf-8",
    )
    (separate / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n",
        encoding="utf-8",
    )

    runs = list_simulation_summaries(results_root=tmp_path)

    assert len(runs) == 2
    assert runs[0]["id"] == "newer"
    assert runs[0]["account_id"] == "account-1"
    assert runs[1]["account_id"] == "account-2"


def test_list_simulation_plan_summaries_keeps_only_latest_plan_per_account(tmp_path: Path) -> None:
    account_plan_old = tmp_path / "account-1" / "plans" / "plan-old"
    account_plan_old.mkdir(parents=True)
    (account_plan_old / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"execution_date": "2025-01-09", "signal_date": "2025-01-08"},
                "strategy_config": {"storage_root": "storage"},
                "plan": {"selected_symbols": ["AAA"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (account_plan_old / "meta.json").write_text(
        json.dumps(
            {
                "name": "demo account",
                "strategy_id": "demo_strategy",
                "config_path": "configs/demo.toml",
                "account_id": "account-1",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    account_plan_new = tmp_path / "account-1" / "plans" / "plan-new"
    account_plan_new.mkdir(parents=True)
    (account_plan_new / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"execution_date": "2025-01-10", "signal_date": "2025-01-09"},
                "strategy_config": {"storage_root": "storage"},
                "plan": {"selected_symbols": ["BBB"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (account_plan_new / "meta.json").write_text(
        json.dumps(
            {
                "name": "demo account",
                "strategy_id": "demo_strategy",
                "config_path": "configs/demo.toml",
                "account_id": "account-1",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    second_account_plan = tmp_path / "account-2" / "plans" / "plan-other"
    second_account_plan.mkdir(parents=True)
    (second_account_plan / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"execution_date": "2025-01-08", "signal_date": "2025-01-07"},
                "strategy_config": {"storage_root": "storage"},
                "plan": {"selected_symbols": ["CCC"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (second_account_plan / "meta.json").write_text(
        json.dumps(
            {
                "name": "another account",
                "strategy_id": "demo_strategy",
                "config_path": "configs/demo.toml",
                "account_id": "account-2",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    plans = list_simulation_plan_summaries(results_root=tmp_path)

    assert len(plans) == 2
    assert plans[0]["account_id"] == "account-1"
    assert plans[0]["id"] == "plan-new"
    assert plans[1]["account_id"] == "account-2"


def test_load_simulation_history_detail_aggregates_only_current_account(tmp_path: Path) -> None:
    account_one_old = tmp_path / "account-one-old"
    account_one_old.mkdir(parents=True)
    (account_one_old / "strategy_state.json").write_text(
        json.dumps({"summary": {"execution_date": "2025-01-09", "signal_date": "2025-01-08"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (account_one_old / "meta.json").write_text(
        json.dumps({"strategy_id": "demo_strategy", "account_id": "account-1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (account_one_old / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n"
        "2025-01-09,AAA,BUY,100,10,1000,1,0,0,filled,entry\n",
        encoding="utf-8",
    )
    (account_one_old / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n",
        encoding="utf-8",
    )

    account_one_new = tmp_path / "account-one-new"
    account_one_new.mkdir(parents=True)
    (account_one_new / "strategy_state.json").write_text(
        json.dumps({"summary": {"execution_date": "2025-01-10", "signal_date": "2025-01-09"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (account_one_new / "meta.json").write_text(
        json.dumps({"strategy_id": "demo_strategy", "account_id": "account-1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (account_one_new / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n",
        encoding="utf-8",
    )
    (account_one_new / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n",
        encoding="utf-8",
    )

    account_two = tmp_path / "account-two"
    account_two.mkdir(parents=True)
    (account_two / "strategy_state.json").write_text(
        json.dumps({"summary": {"execution_date": "2025-01-11", "signal_date": "2025-01-10"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (account_two / "meta.json").write_text(
        json.dumps({"strategy_id": "demo_strategy", "account_id": "account-2"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (account_two / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n"
        "2025-01-11,BBB,BUY,200,20,4000,1,0,0,filled,entry\n",
        encoding="utf-8",
    )
    (account_two / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n",
        encoding="utf-8",
    )

    detail = load_simulation_history_detail("demo_strategy", run_id="account-one-new", results_root=tmp_path)

    assert detail["account_id"] == "account-1"
    assert detail["summary"]["trade_count"] == 1
    assert len(detail["trades"]) == 1
    assert detail["trades"][0]["symbol"] == "AAA"


def test_load_simulation_history_detail_ignores_unexecuted_plan_trades_in_account_snapshot(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-one"
    run_dir.mkdir(parents=True)
    (run_dir / "strategy_state.json").write_text(
        json.dumps({"summary": {"execution_date": "2025-01-09", "signal_date": "2025-01-08"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"strategy_id": "demo_strategy", "account_id": "account-1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n"
        "2025-01-09,AAA,BUY,100,10,1000,1,0,0,filled,entry\n",
        encoding="utf-8",
    )
    (run_dir / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n",
        encoding="utf-8",
    )

    account_snapshot_dir = tmp_path / "accounts" / "account-1"
    account_snapshot_dir.mkdir(parents=True)
    (account_snapshot_dir / "strategy_state.json").write_text(
        json.dumps({"summary": {"execution_date": "2025-01-10", "signal_date": "2025-01-09"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (account_snapshot_dir / "meta.json").write_text(
        json.dumps({"strategy_id": "demo_strategy", "account_id": "account-1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (account_snapshot_dir / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n"
        "2025-01-09,AAA,BUY,100,10,1000,1,0,0,filled,entry\n"
        "2025-01-10,BBB,BUY,100,12,1200,1,0,0,filled,planned_only\n",
        encoding="utf-8",
    )
    (account_snapshot_dir / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n",
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    original_simulation_accounts_root = web_app.SIMULATION_ACCOUNTS_ROOT
    try:
        web_app.SIMULATION_ACCOUNTS_ROOT = tmp_path / "accounts"
        detail = load_simulation_history_detail("demo_strategy", run_id="run-one", results_root=tmp_path)
    finally:
        web_app.SIMULATION_ACCOUNTS_ROOT = original_simulation_accounts_root

    assert detail["account_id"] == "account-1"
    assert detail["summary"]["trade_count"] == 1
    assert len(detail["trades"]) == 1
    assert detail["trades"][0]["symbol"] == "AAA"


def test_load_simulation_history_detail_does_not_double_count_account_root_and_nested_run(tmp_path: Path) -> None:
    account_dir = tmp_path / "account-1"
    account_dir.mkdir(parents=True)
    (account_dir / "strategy_state.json").write_text(
        json.dumps({"summary": {"execution_date": "2025-01-09", "signal_date": "2025-01-08"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (account_dir / "meta.json").write_text(
        json.dumps({"strategy_id": "demo_strategy", "account_id": "account-1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (account_dir / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n"
        "2025-01-09,AAA,BUY,100,10,1000,1,0,0,filled,entry\n",
        encoding="utf-8",
    )
    (account_dir / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n",
        encoding="utf-8",
    )

    run_dir = account_dir / "runs" / "run-one"
    run_dir.mkdir(parents=True)
    (run_dir / "strategy_state.json").write_text(
        json.dumps({"summary": {"execution_date": "2025-01-09", "signal_date": "2025-01-08"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"strategy_id": "demo_strategy", "account_id": "account-1"}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n"
        "2025-01-09,AAA,BUY,100,10,1000,1,0,0,filled,entry\n",
        encoding="utf-8",
    )
    (run_dir / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n",
        encoding="utf-8",
    )

    detail = load_simulation_history_detail("demo_strategy", run_id="run-one", results_root=tmp_path)

    assert detail["account_id"] == "account-1"
    assert detail["summary"]["trade_count"] == 1
    assert len(detail["trades"]) == 1
    assert detail["trades"][0]["symbol"] == "AAA"


def test_find_previous_simulation_run_and_load_latest_snapshot(tmp_path: Path) -> None:
    older = tmp_path / "results" / "simulation_runs" / "older"
    older.mkdir(parents=True)
    (older / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"execution_date": "2025-01-09", "signal_date": "2025-01-08"},
                "next_state": {"positions": [], "pending_orders": [], "as_of_trade_date": "2025-01-09"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (older / "meta.json").write_text(
        json.dumps({"strategy_id": "demo_strategy"}, ensure_ascii=False),
        encoding="utf-8",
    )
    newer = tmp_path / "results" / "simulation_runs" / "newer"
    newer.mkdir(parents=True)
    (newer / "strategy_state.json").write_text(
        json.dumps(
            {
                "summary": {"execution_date": "2025-01-10", "signal_date": "2025-01-09"},
                "plan": {"selected_symbols": ["AAA"]},
                "next_state": {"positions": [], "pending_orders": [], "as_of_trade_date": "2025-01-10"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (newer / "meta.json").write_text(
        json.dumps({"strategy_id": "demo_strategy", "name": "newer run"}, ensure_ascii=False),
        encoding="utf-8",
    )
    latest_dir = tmp_path / "results" / "simulation_latest" / "demo_strategy"
    latest_dir.mkdir(parents=True)
    (latest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "strategy_id": "demo_strategy",
                "run_id": "newer",
                "name": "newer run",
                "signal_date": "2025-01-09",
                "execution_date": "2025-01-10",
                "scores_path": "research/models/demo_scores.parquet",
                "strategy_state_path": "results/simulation_runs/newer/strategy_state.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    original_sim_root = web_app.SIMULATION_RUNS_ROOT
    original_sim_latest_root = web_app.SIMULATION_LATEST_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        web_app.SIMULATION_RUNS_ROOT = tmp_path / "results" / "simulation_runs"
        web_app.SIMULATION_LATEST_ROOT = tmp_path / "results" / "simulation_latest"
        previous_run = _find_previous_simulation_run("demo_strategy", "2025-01-11", results_root=web_app.SIMULATION_RUNS_ROOT)
        detail = load_latest_simulation_snapshot("demo_strategy")
    finally:
        web_app.REPO_ROOT = original_repo_root
        web_app.SIMULATION_RUNS_ROOT = original_sim_root
        web_app.SIMULATION_LATEST_ROOT = original_sim_latest_root

    assert previous_run is not None
    assert previous_run["run_id"] == "newer"
    assert detail["strategy_id"] == "demo_strategy"
    assert detail["strategy_state"]["plan"]["selected_symbols"] == ["AAA"]


def test_load_paper_history_detail_raises_when_latest_trade_log_missing(tmp_path: Path) -> None:
    latest_dir = tmp_path / "research" / "models" / "latest" / "demo_strategy"
    latest_dir.mkdir(parents=True)
    (latest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "strategy_id": "demo_strategy",
                "strategy_state_path": "research/models/latest/demo_strategy/strategy_state.json",
                "trades_path": "research/models/latest/demo_strategy/trades.csv",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        try:
            load_paper_history_detail("demo_strategy")
            raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError as exc:
            assert str(exc) == "latest trade log not found: demo_strategy"
    finally:
        web_app.REPO_ROOT = original_repo_root


def test_load_paper_history_detail_prefers_latest_trade_log_when_present(tmp_path: Path) -> None:
    config_root = tmp_path / "configs"
    config_root.mkdir()
    (config_root / "demo_strategy.toml").write_text(
        """
[storage]
root = "storage"

[factors]
output_path = "research/factors/demo.parquet"
universe_name = "tradable_core"
start_date = "2024-01-02"
end_date = "2026-03-10"

[training]
label_column = "industry_excess_fwd_return_5"
train_window_months = 12
test_start_month = "2025-01"
test_end_month = "2026-02"
score_output_path = "research/models/walk_forward_demo.parquet"
metric_output_path = "research/models/walk_forward_demo.json"

[analysis]
layer_output_path = "research/models/layer_demo.json"

[model_backtest]
output_dir = "results/demo_backtest"
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
initial_cash = 1000000
commission_rate = 0.0003
stamp_tax_rate = 0.001
slippage_rate = 0.0005
        """.strip(),
        encoding="utf-8",
    )
    latest_dir = tmp_path / "research" / "models" / "latest" / "demo_strategy"
    latest_dir.mkdir(parents=True)
    (latest_dir / "strategy_state.json").write_text(
        json.dumps({"summary": {"signal_date": "2025-01-09", "execution_date": "2025-01-10"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (latest_dir / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n"
        "2025-01-10,BBB,BUY,200,20,4000,1,0,0.5,filled,rebalance_entry\n",
        encoding="utf-8",
    )
    (latest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "strategy_id": "demo_strategy",
                "strategy_state_path": "research/models/latest/demo_strategy/strategy_state.json",
                "trades_path": "research/models/latest/demo_strategy/trades.csv",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        detail = load_paper_history_detail("demo_strategy", config_root=config_root, results_root=tmp_path / "results")
    finally:
        web_app.REPO_ROOT = original_repo_root

    assert detail["run_id"] == "latest"
    assert detail["source_kind"] == "latest_trade_log"
    assert detail["trades"][0]["symbol"] == "BBB"


def test_load_latest_paper_lineage_reads_decision_log_and_trades(tmp_path: Path) -> None:
    latest_dir = tmp_path / "research" / "models" / "latest" / "demo_strategy"
    latest_dir.mkdir(parents=True)
    (latest_dir / "strategy_state.json").write_text(
        json.dumps({"summary": {"signal_date": "2025-01-09", "execution_date": "2025-01-10"}}, ensure_ascii=False),
        encoding="utf-8",
    )
    (latest_dir / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n"
        "2025-01-10,BBB,BUY,200,20,4000,1,0,0.5,filled,rebalance_entry\n",
        encoding="utf-8",
    )
    (latest_dir / "decision_log.csv").write_text(
        "trade_date,signal_date,decision_reason,should_rebalance,selected_symbols,current_position_count,target_position_count,cash_pre_decision\n"
        "2025-01-10,2025-01-09,model_score_schedule,True,BBB,0,1,1000000\n",
        encoding="utf-8",
    )
    (latest_dir / "manifest.json").write_text(
        json.dumps(
            {
                "strategy_id": "demo_strategy",
                "signal_date": "2025-01-09",
                "execution_date": "2025-01-10",
                "strategy_state_path": "research/models/latest/demo_strategy/strategy_state.json",
                "trades_path": "research/models/latest/demo_strategy/trades.csv",
                "decision_log_path": "research/models/latest/demo_strategy/decision_log.csv",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    from ashare_backtest.web import app as web_app

    original_repo_root = web_app.REPO_ROOT
    try:
        web_app.REPO_ROOT = tmp_path
        detail = load_latest_paper_lineage("demo_strategy")
    finally:
        web_app.REPO_ROOT = original_repo_root

    assert detail["source_kind"] == "latest_lineage"
    assert detail["decision_log"][0]["decision_reason"] == "model_score_schedule"
    assert detail["trades"][0]["symbol"] == "BBB"


def test_list_run_summaries_filters_valid_result_dirs(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    valid.mkdir()
    (valid / "summary.json").write_text(json.dumps({"total_return": 0.2}), encoding="utf-8")
    (valid / "equity_curve.csv").write_text("trade_date,equity\n2025-01-02,100\n", encoding="utf-8")
    (valid / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n",
        encoding="utf-8",
    )
    invalid = tmp_path / "invalid"
    invalid.mkdir()

    runs = list_run_summaries(results_root=tmp_path)

    assert [run["id"] for run in runs] == ["valid"]


def test_load_run_detail_finds_nested_web_run_dir(tmp_path: Path) -> None:
    nested = tmp_path / "web_runs" / "nested_run"
    nested.mkdir(parents=True)
    (nested / "summary.json").write_text(json.dumps({"total_return": 0.3}), encoding="utf-8")
    (nested / "equity_curve.csv").write_text("trade_date,equity\n2025-01-02,100\n", encoding="utf-8")
    (nested / "trades.csv").write_text(
        "trade_date,symbol,side,quantity,price,amount,commission,tax,slippage,status,reason\n",
        encoding="utf-8",
    )

    detail = load_run_detail("nested_run", results_root=tmp_path, bars_path=tmp_path / "missing.parquet")

    assert detail["id"] == "nested_run"
    assert detail["summary"]["total_return"] == 0.3
