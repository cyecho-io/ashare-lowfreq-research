from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from ashare_backtest.web.app import (
    load_latest_paper_snapshot,
    load_latest_paper_lineage,
    load_paper_history_detail,
    load_paper_trade_detail,
    load_run_detail,
    list_score_parquet_files,
    list_paper_trade_summaries,
    list_run_summaries,
    list_strategy_presets,
)


def test_list_strategy_presets_reads_research_configs() -> None:
    presets = list_strategy_presets()
    ids = {preset.id for preset in presets}
    assert "research_industry_v4_v1_1" in ids


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

    detail = load_run_detail("demo_run", results_root=tmp_path, bars_path=tmp_path / "missing.parquet", benchmark_path=tmp_path / "missing.parquet")

    assert detail["strategy_state"]["plan"]["selected_symbols"] == ["AAA", "BBB"]


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
