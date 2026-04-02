from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd

from ashare_backtest.data import (
    DEFAULT_BENCHMARK_OUTPUT,
    DEFAULT_BENCHMARK_SYMBOL,
    DEFAULT_SQLITE_SOURCE,
    SQLiteParquetImporter,
    TushareBenchmarkSync,
    TushareClient,
    TushareSQLiteSync,
    resolve_tushare_token,
)
from ashare_backtest.factors import FactorBuildConfig, FactorBuilder, resolve_factor_snapshot_path
from ashare_backtest.research import (
    CapacityAnalysisConfig,
    LayeredAnalysisConfig,
    ModelTrainConfig,
    MonthlyComparisonConfig,
    PremarketReferenceConfig,
    RiskExposureConfig,
    StartDateRobustnessConfig,
    StrategyStateConfig,
    SweepConfig,
    WalkForwardAsOfDateConfig,
    WalkForwardConfig,
    WalkForwardSingleDateConfig,
    analyze_start_date_robustness,
    generate_premarket_reference,
    generate_strategy_state,
    analyze_score_layers,
    analyze_trade_capacity,
    analyze_monthly_risk_exposures,
    compare_backtest_monthly_returns,
    run_model_sweep,
    train_lightgbm_model,
    train_lightgbm_walk_forward_as_of_date,
    train_lightgbm_walk_forward,
    train_lightgbm_walk_forward_single_date,
)
from ashare_backtest.cli.commands.backtest import list_universes, run_backtest, run_model_backtest
from ashare_backtest.cli.commands.research import (
    analyze_start_date_robustness_from_config,
    generate_premarket_reference_from_config,
    resolve_month_range_output_path,
    resolve_premarket_output_path,
    resolve_start_date_robustness_output_path,
    run_research_pipeline,
)
from ashare_backtest.protocol import BacktestConfig
from ashare_backtest.registry import StrategyLibrary
from ashare_backtest.sandbox import StrategyValidationError, StrategyValidator
from .config import load_run_config
from .research_config import load_research_config, resolve_dated_output_path, resolve_research_config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Personal A-share low-frequency backtest tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a strategy script")
    validate_parser.add_argument("path", help="Path to the strategy script")

    register_parser = subparsers.add_parser("register", help="Register a validated strategy script")
    register_parser.add_argument("path", help="Path to the strategy script")
    register_parser.add_argument(
        "--library",
        default="strategies",
        help="Directory where validated strategies are stored",
    )

    import_parser = subparsers.add_parser("import-sqlite", help="Import SQLite market data into Parquet storage")
    import_parser.add_argument(
        "sqlite_path",
        nargs="?",
        default=DEFAULT_SQLITE_SOURCE,
        help="Path to the source SQLite database",
    )
    import_parser.add_argument(
        "--storage-root",
        default="storage",
        help="Directory where standardized Parquet data is stored",
    )

    sync_tushare_parser = subparsers.add_parser(
        "sync-tushare-sqlite",
        help="Sync Tushare daily data into the source SQLite database",
    )
    sync_tushare_parser.add_argument(
        "--sqlite-path",
        default=DEFAULT_SQLITE_SOURCE,
        help="Path to the source SQLite database",
    )
    sync_tushare_parser.add_argument("--start", default=None, help="Sync start date, YYYYMMDD")
    sync_tushare_parser.add_argument("--end", default=None, help="Sync end date, YYYYMMDD")
    sync_tushare_parser.add_argument("--token", default=None, help="Tushare token, defaults to TUSHARE_TOKEN env var")

    benchmark_parser = subparsers.add_parser(
        "sync-tushare-benchmark",
        help="Sync benchmark index daily data from Tushare into project parquet storage",
    )
    benchmark_parser.add_argument("--symbol", default=DEFAULT_BENCHMARK_SYMBOL, help="Benchmark ts_code, e.g. 000300.SH")
    benchmark_parser.add_argument("--start", default=None, help="Sync start date, YYYYMMDD")
    benchmark_parser.add_argument("--end", default=None, help="Sync end date, YYYYMMDD")
    benchmark_parser.add_argument("--output-path", default=DEFAULT_BENCHMARK_OUTPUT)
    benchmark_parser.add_argument("--token", default=None, help="Tushare token, defaults to TUSHARE_TOKEN env var")

    universe_parser = subparsers.add_parser("list-universes", help="List available universe memberships")
    universe_parser.add_argument("--storage-root", default="storage", help="Parquet storage root")

    run_parser = subparsers.add_parser("run-backtest", help="Run a backtest on imported Parquet data")
    run_parser.add_argument("strategy_path", help="Path to the strategy script")
    run_parser.add_argument("--storage-root", default="storage", help="Parquet storage root")
    run_parser.add_argument("--start-date", required=True, help="Backtest start date, YYYY-MM-DD")
    run_parser.add_argument("--end-date", required=True, help="Backtest end date, YYYY-MM-DD")
    run_parser.add_argument(
        "--universe",
        required=True,
        help="Comma-separated symbol list, e.g. 600519.SH,000001.SZ",
    )
    run_parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    run_parser.add_argument("--commission-rate", type=float, default=0.0003)
    run_parser.add_argument("--stamp-tax-rate", type=float, default=0.001)
    run_parser.add_argument("--slippage-rate", type=float, default=0.0005)
    run_parser.add_argument("--max-trade-participation-rate", type=float, default=0.0)
    run_parser.add_argument("--max-pending-days", type=int, default=0)
    run_parser.add_argument("--output-dir", default="results/latest")

    run_config_parser = subparsers.add_parser("run-config", help="Run a backtest from a TOML config file")
    run_config_parser.add_argument("config_path", help="Path to the TOML config file")

    factor_parser = subparsers.add_parser("build-factors", help="Build a basic factor panel from Parquet bars")
    factor_parser.add_argument("--storage-root", default="storage", help="Parquet storage root")
    factor_parser.add_argument(
        "--output-path",
        default="",
        help="Optional factor snapshot output path; defaults to research/factors/<factor-spec-id>/<as-of-date>.parquet when both are provided",
    )
    factor_parser.add_argument("--factor-spec-id", default="", help="Optional factor spec identifier used for standard snapshot paths")
    factor_parser.add_argument("--symbols", default="", help="Optional comma-separated symbols")
    factor_parser.add_argument("--universe-name", default="", help="Optional universe membership name")
    factor_parser.add_argument("--start-date", default=None, help="Optional start date, YYYY-MM-DD")
    factor_parser.add_argument("--as-of-date", dest="as_of_date", default=None, help="Optional factor snapshot date, YYYY-MM-DD")

    model_parser = subparsers.add_parser("train-lgbm", help="Train a minimal LightGBM model on factor panel")
    model_parser.add_argument(
        "--factor-panel-path",
        default="research/factors/basic_factor_panel.parquet",
        help="Input factor panel parquet path",
    )
    model_parser.add_argument("--label-column", default="fwd_return_5")
    model_parser.add_argument("--train-end-date", default="2024-09-30")
    model_parser.add_argument("--test-start-date", default="2024-10-01")
    model_parser.add_argument("--test-end-date", default="2024-12-31")
    model_parser.add_argument("--output-scores-path", default="research/models/model_scores.parquet")
    model_parser.add_argument("--output-metrics-path", default="research/models/model_metrics.json")

    wf_parser = subparsers.add_parser(
        "train-lgbm-walk-forward",
        help="Train LightGBM in a monthly walk-forward manner",
    )
    wf_parser.add_argument(
        "--factor-panel-path",
        default="research/factors/full_factor_panel_v2.parquet",
        help="Input factor panel parquet path",
    )
    wf_parser.add_argument("--label-column", default="fwd_return_5")
    wf_parser.add_argument("--train-window-months", type=int, default=12)
    wf_parser.add_argument("--test-start-month", default="2025-07")
    wf_parser.add_argument("--test-end-month", default="2026-02")
    wf_parser.add_argument("--output-scores-path", default="research/models/walk_forward_scores.parquet")
    wf_parser.add_argument("--output-metrics-path", default="research/models/walk_forward_metrics.json")

    wf_from_config_parser = subparsers.add_parser(
        "train-lgbm-walk-forward-from-config",
        help="Train monthly walk-forward scores using defaults loaded from a research TOML config",
    )
    wf_from_config_parser.add_argument("config_path", nargs="?", default="", help="Optional research TOML path")
    wf_from_config_parser.add_argument("--factor-spec-id", default="", help="Resolve configs/<factor-spec-id>.toml when config path is omitted")
    wf_from_config_parser.add_argument("--factor-panel-path", default="", help="Input factor panel parquet path")
    wf_from_config_parser.add_argument("--test-start-month", required=True)
    wf_from_config_parser.add_argument("--test-end-month", required=True)
    wf_from_config_parser.add_argument(
        "--output-scores-path",
        default="",
        help="Optional score parquet override; defaults to <score_output_path> with _<start>_to_<end> suffix",
    )
    wf_from_config_parser.add_argument(
        "--output-metrics-path",
        default="",
        help="Optional metrics json override; defaults to <metric_output_path> with _<start>_to_<end> suffix",
    )

    wf_asof_parser = subparsers.add_parser(
        "train-lgbm-walk-forward-as-of-date",
        help="Score a single as-of date using the same walk-forward training window semantics",
    )
    wf_asof_parser.add_argument(
        "--factor-panel-path",
        default="research/factors/full_factor_panel_v2.parquet",
        help="Input factor panel parquet path",
    )
    wf_asof_parser.add_argument("--label-column", default="fwd_return_5")
    wf_asof_parser.add_argument("--as-of-date", default=None)
    wf_asof_parser.add_argument("--train-window-months", type=int, default=12)
    wf_asof_parser.add_argument("--output-scores-path", default="research/models/walk_forward_scores_as_of_date.parquet")
    wf_asof_parser.add_argument("--output-metrics-path", default="research/models/walk_forward_metrics_as_of_date.json")

    wf_asof_from_config_parser = subparsers.add_parser(
        "train-lgbm-walk-forward-as-of-date-from-config",
        help="Score a single as-of date using defaults loaded from a research TOML config",
    )
    wf_asof_from_config_parser.add_argument("config_path", nargs="?", default="", help="Optional research TOML path")
    wf_asof_from_config_parser.add_argument("--factor-spec-id", default="", help="Resolve configs/<factor-spec-id>.toml when config path is omitted")
    wf_asof_from_config_parser.add_argument(
        "--as-of-date",
        default="",
        help="Optional score date; when omitted and --factor-panel-path is provided, infer from the factor parquet",
    )
    wf_asof_from_config_parser.add_argument(
        "--factor-panel-path",
        default="",
        help="Optional factor panel override; recommended primary input for dated inference runs",
    )
    wf_asof_from_config_parser.add_argument(
        "--output-scores-path",
        default="",
        help="Optional score parquet override; defaults to <score_output_path> with _<as-of-date> suffix",
    )
    wf_asof_from_config_parser.add_argument(
        "--output-metrics-path",
        default="",
        help="Optional metrics json override; defaults to <metric_output_path> with _<as-of-date> suffix",
    )

    wf_single_from_config_parser = subparsers.add_parser(
        "train-lgbm-walk-forward-single-date-from-config",
        help="Score a single date using the model implied by a specific walk-forward test month",
    )
    wf_single_from_config_parser.add_argument("config_path", nargs="?", default="", help="Optional research TOML path")
    wf_single_from_config_parser.add_argument("--factor-spec-id", default="", help="Resolve configs/<factor-spec-id>.toml when config path is omitted")
    wf_single_from_config_parser.add_argument("--factor-panel-path", default="", help="Optional factor panel override")
    wf_single_from_config_parser.add_argument("--test-month", required=True, help="Walk-forward test month, YYYY-MM")
    wf_single_from_config_parser.add_argument("--as-of-date", required=True, help="Target scoring date, YYYY-MM-DD")
    wf_single_from_config_parser.add_argument(
        "--output-scores-path",
        default="",
        help="Optional score parquet override; defaults to <score_output_path> with _<as-of-date> suffix",
    )
    wf_single_from_config_parser.add_argument(
        "--output-metrics-path",
        default="",
        help="Optional metrics json override; defaults to <metric_output_path> with _<as-of-date> suffix",
    )

    latest_parser = subparsers.add_parser(
        "train-lgbm-latest-inference",
        help="Deprecated alias for train-lgbm-walk-forward-as-of-date",
    )
    latest_parser.add_argument(
        "--factor-panel-path",
        default="research/factors/full_factor_panel_v2.parquet",
        help="Input factor panel parquet path",
    )
    latest_parser.add_argument("--label-column", default="fwd_return_5")
    latest_parser.add_argument("--inference-date", default=None)
    latest_parser.add_argument("--train-window-months", type=int, default=12)
    latest_parser.add_argument("--output-scores-path", default="research/models/walk_forward_scores_as_of_date.parquet")
    latest_parser.add_argument("--output-metrics-path", default="research/models/walk_forward_metrics_as_of_date.json")

    score_bt_parser = subparsers.add_parser(
        "run-model-backtest",
        help="Run a backtest driven by model score parquet output",
    )
    score_bt_parser.add_argument("--scores-path", default="research/models/walk_forward_scores.parquet")
    score_bt_parser.add_argument("--storage-root", default="storage")
    score_bt_parser.add_argument("--start-date", required=True)
    score_bt_parser.add_argument("--end-date", required=True)
    score_bt_parser.add_argument("--top-k", type=int, default=5)
    score_bt_parser.add_argument("--rebalance-every", type=int, default=3)
    score_bt_parser.add_argument("--lookback-window", type=int, default=20)
    score_bt_parser.add_argument("--min-hold-bars", type=int, default=5)
    score_bt_parser.add_argument("--keep-buffer", type=int, default=2)
    score_bt_parser.add_argument("--min-turnover-names", type=int, default=2)
    score_bt_parser.add_argument("--min-daily-amount", type=float, default=0.0)
    score_bt_parser.add_argument("--max-close-price", type=float, default=0.0)
    score_bt_parser.add_argument("--max-names-per-industry", type=int, default=0)
    score_bt_parser.add_argument("--max-position-weight", type=float, default=0.0)
    score_bt_parser.add_argument("--exit-policy", default="buffered_rank")
    score_bt_parser.add_argument("--grace-rank-buffer", type=int, default=0)
    score_bt_parser.add_argument("--grace-momentum-window", type=int, default=3)
    score_bt_parser.add_argument("--grace-min-return", type=float, default=0.0)
    score_bt_parser.add_argument("--trailing-stop-window", type=int, default=10)
    score_bt_parser.add_argument("--trailing-stop-drawdown", type=float, default=0.12)
    score_bt_parser.add_argument("--trailing-stop-min-gain", type=float, default=0.15)
    score_bt_parser.add_argument("--score-reversal-confirm-days", type=int, default=3)
    score_bt_parser.add_argument("--score-reversal-threshold", type=float, default=0.0)
    score_bt_parser.add_argument("--hybrid-price-window", type=int, default=5)
    score_bt_parser.add_argument("--hybrid-price-threshold", type=float, default=0.0)
    score_bt_parser.add_argument("--strong-keep-extra-buffer", type=int, default=0)
    score_bt_parser.add_argument("--strong-keep-momentum-window", type=int, default=5)
    score_bt_parser.add_argument("--strong-keep-min-return", type=float, default=0.0)
    score_bt_parser.add_argument("--strong-trim-slowdown", type=float, default=0.0)
    score_bt_parser.add_argument("--strong-trim-momentum-window", type=int, default=5)
    score_bt_parser.add_argument("--strong-trim-min-return", type=float, default=0.0)
    score_bt_parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    score_bt_parser.add_argument("--commission-rate", type=float, default=0.0003)
    score_bt_parser.add_argument("--stamp-tax-rate", type=float, default=0.001)
    score_bt_parser.add_argument("--slippage-rate", type=float, default=0.0005)
    score_bt_parser.add_argument("--max-trade-participation-rate", type=float, default=0.0)
    score_bt_parser.add_argument("--max-pending-days", type=int, default=0)
    score_bt_parser.add_argument("--output-dir", default="results/model_score_backtest")

    layer_parser = subparsers.add_parser(
        "analyze-score-layers",
        help="Analyze layered forward returns based on model scores",
    )
    layer_parser.add_argument("--scores-path", required=True)
    layer_parser.add_argument("--output-path", default="research/models/layer_analysis.json")
    layer_parser.add_argument("--bins", type=int, default=5)

    capacity_parser = subparsers.add_parser(
        "analyze-trade-capacity",
        help="Estimate trade participation and capacity degradation from exported trades",
    )
    capacity_parser.add_argument("--trades-path", required=True)
    capacity_parser.add_argument("--storage-root", default="storage")
    capacity_parser.add_argument("--output-path", default="research/models/capacity_analysis.json")
    capacity_parser.add_argument("--base-capital", type=float, default=1_000_000.0)
    capacity_parser.add_argument("--scale-capitals", default="100000,300000,500000,1000000,3000000,5000000")
    capacity_parser.add_argument("--participation-thresholds", default="0.01,0.02,0.05,0.10")
    capacity_parser.add_argument("--top-trade-count", type=int, default=20)

    monthly_parser = subparsers.add_parser(
        "compare-backtest-monthly",
        help="Compare monthly returns across multiple backtest result directories",
    )
    monthly_parser.add_argument("--result-dirs", required=True, help="Comma-separated result directories")
    monthly_parser.add_argument("--labels", required=True, help="Comma-separated labels matching result directories")
    monthly_parser.add_argument("--output-path", default="research/models/backtest_monthly_comparison.json")

    risk_parser = subparsers.add_parser(
        "analyze-risk-exposures",
        help="Analyze monthly risk exposures from backtest trades and market data",
    )
    risk_parser.add_argument("--result-dir", required=True)
    risk_parser.add_argument("--storage-root", default="storage")
    risk_parser.add_argument("--output-path", default="research/models/risk_exposures.json")
    risk_parser.add_argument("--top-industries", type=int, default=5)
    risk_parser.add_argument("--volatility-window", type=int, default=20)

    robustness_parser = subparsers.add_parser(
        "analyze-start-date-robustness",
        help="Run rolling start-date backtests to measure sensitivity to entry timing",
    )
    robustness_parser.add_argument("--scores-path", required=True)
    robustness_parser.add_argument("--storage-root", default="storage")
    robustness_parser.add_argument("--analysis-start-date", required=True)
    robustness_parser.add_argument("--analysis-end-date", required=True)
    robustness_parser.add_argument("--holding-months", type=int, default=8)
    robustness_parser.add_argument("--cadence", choices=("daily", "monthly"), default="monthly")
    robustness_parser.add_argument("--universe-name", default="")
    robustness_parser.add_argument("--top-k", type=int, default=5)
    robustness_parser.add_argument("--rebalance-every", type=int, default=3)
    robustness_parser.add_argument("--lookback-window", type=int, default=20)
    robustness_parser.add_argument("--min-hold-bars", type=int, default=5)
    robustness_parser.add_argument("--keep-buffer", type=int, default=2)
    robustness_parser.add_argument("--min-turnover-names", type=int, default=2)
    robustness_parser.add_argument("--min-daily-amount", type=float, default=0.0)
    robustness_parser.add_argument("--max-close-price", type=float, default=0.0)
    robustness_parser.add_argument("--max-names-per-industry", type=int, default=0)
    robustness_parser.add_argument("--max-position-weight", type=float, default=0.0)
    robustness_parser.add_argument("--exit-policy", default="buffered_rank")
    robustness_parser.add_argument("--grace-rank-buffer", type=int, default=0)
    robustness_parser.add_argument("--grace-momentum-window", type=int, default=3)
    robustness_parser.add_argument("--grace-min-return", type=float, default=0.0)
    robustness_parser.add_argument("--trailing-stop-window", type=int, default=10)
    robustness_parser.add_argument("--trailing-stop-drawdown", type=float, default=0.12)
    robustness_parser.add_argument("--trailing-stop-min-gain", type=float, default=0.15)
    robustness_parser.add_argument("--score-reversal-confirm-days", type=int, default=3)
    robustness_parser.add_argument("--score-reversal-threshold", type=float, default=0.0)
    robustness_parser.add_argument("--hybrid-price-window", type=int, default=5)
    robustness_parser.add_argument("--hybrid-price-threshold", type=float, default=0.0)
    robustness_parser.add_argument("--strong-keep-extra-buffer", type=int, default=0)
    robustness_parser.add_argument("--strong-keep-momentum-window", type=int, default=5)
    robustness_parser.add_argument("--strong-keep-min-return", type=float, default=0.0)
    robustness_parser.add_argument("--strong-trim-slowdown", type=float, default=0.0)
    robustness_parser.add_argument("--strong-trim-momentum-window", type=int, default=5)
    robustness_parser.add_argument("--strong-trim-min-return", type=float, default=0.0)
    robustness_parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    robustness_parser.add_argument("--commission-rate", type=float, default=0.0003)
    robustness_parser.add_argument("--stamp-tax-rate", type=float, default=0.001)
    robustness_parser.add_argument("--slippage-rate", type=float, default=0.0005)
    robustness_parser.add_argument("--max-trade-participation-rate", type=float, default=0.0)
    robustness_parser.add_argument("--max-pending-days", type=int, default=0)
    robustness_parser.add_argument("--output-path", default="research/models/start_date_robustness.json")

    robustness_from_config_parser = subparsers.add_parser(
        "analyze-start-date-robustness-from-config",
        help="Measure start-date sensitivity using defaults loaded from a research TOML config",
    )
    robustness_from_config_parser.add_argument("config_path", nargs="?", default="", help="Optional research TOML path")
    robustness_from_config_parser.add_argument("--factor-spec-id", default="", help="Resolve configs/<factor-spec-id>.toml when config path is omitted")
    robustness_from_config_parser.add_argument("--scores-path", default="", help="Optional override score parquet path")
    robustness_from_config_parser.add_argument("--analysis-start-date", default="")
    robustness_from_config_parser.add_argument("--analysis-end-date", default="")
    robustness_from_config_parser.add_argument("--holding-months", type=int, default=8)
    robustness_from_config_parser.add_argument("--cadence", choices=("daily", "monthly"), default="monthly")
    robustness_from_config_parser.add_argument("--output-path", default="")

    premarket_parser = subparsers.add_parser(
        "generate-premarket-reference",
        help="Generate a premarket buy/sell/hold reference from the model strategy",
    )
    premarket_parser.add_argument("--scores-path", default="research/models/walk_forward_scores.parquet")
    premarket_parser.add_argument("--storage-root", default="storage")
    premarket_parser.add_argument("--trade-date", required=True)
    premarket_parser.add_argument("--top-k", type=int, default=5)
    premarket_parser.add_argument("--rebalance-every", type=int, default=3)
    premarket_parser.add_argument("--lookback-window", type=int, default=20)
    premarket_parser.add_argument("--min-hold-bars", type=int, default=5)
    premarket_parser.add_argument("--keep-buffer", type=int, default=2)
    premarket_parser.add_argument("--min-turnover-names", type=int, default=2)
    premarket_parser.add_argument("--min-daily-amount", type=float, default=0.0)
    premarket_parser.add_argument("--max-close-price", type=float, default=0.0)
    premarket_parser.add_argument("--max-names-per-industry", type=int, default=0)
    premarket_parser.add_argument("--max-position-weight", type=float, default=0.0)
    premarket_parser.add_argument("--exit-policy", default="buffered_rank")
    premarket_parser.add_argument("--grace-rank-buffer", type=int, default=0)
    premarket_parser.add_argument("--grace-momentum-window", type=int, default=3)
    premarket_parser.add_argument("--grace-min-return", type=float, default=0.0)
    premarket_parser.add_argument("--trailing-stop-window", type=int, default=10)
    premarket_parser.add_argument("--trailing-stop-drawdown", type=float, default=0.12)
    premarket_parser.add_argument("--trailing-stop-min-gain", type=float, default=0.15)
    premarket_parser.add_argument("--score-reversal-confirm-days", type=int, default=3)
    premarket_parser.add_argument("--score-reversal-threshold", type=float, default=0.0)
    premarket_parser.add_argument("--hybrid-price-window", type=int, default=5)
    premarket_parser.add_argument("--hybrid-price-threshold", type=float, default=0.0)
    premarket_parser.add_argument("--strong-keep-extra-buffer", type=int, default=0)
    premarket_parser.add_argument("--strong-keep-momentum-window", type=int, default=5)
    premarket_parser.add_argument("--strong-keep-min-return", type=float, default=0.0)
    premarket_parser.add_argument("--strong-trim-slowdown", type=float, default=0.0)
    premarket_parser.add_argument("--strong-trim-momentum-window", type=int, default=5)
    premarket_parser.add_argument("--strong-trim-min-return", type=float, default=0.0)
    premarket_parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    premarket_parser.add_argument("--commission-rate", type=float, default=0.0003)
    premarket_parser.add_argument("--stamp-tax-rate", type=float, default=0.001)
    premarket_parser.add_argument("--slippage-rate", type=float, default=0.0005)
    premarket_parser.add_argument("--max-trade-participation-rate", type=float, default=0.0)
    premarket_parser.add_argument("--max-pending-days", type=int, default=0)
    premarket_parser.add_argument("--output-path", default="research/models/premarket_reference.json")

    premarket_from_config_parser = subparsers.add_parser(
        "generate-premarket-reference-from-config",
        help="Generate a premarket reference using defaults loaded from a research TOML config",
    )
    premarket_from_config_parser.add_argument("config_path", nargs="?", default="", help="Optional research TOML path")
    premarket_from_config_parser.add_argument("--factor-spec-id", default="", help="Resolve configs/<factor-spec-id>.toml when config path is omitted")
    premarket_from_config_parser.add_argument("--scores-path", required=True)
    premarket_from_config_parser.add_argument("--trade-date", required=True)
    premarket_from_config_parser.add_argument(
        "--output-path",
        default="",
        help="Optional output path override; defaults to research/models/premarket_reference_<factor-spec-id>_<trade-date>.json",
    )

    state_parser = subparsers.add_parser(
        "generate-strategy-state",
        help="Generate a reusable strategy state file for initial build or continued execution",
    )
    state_parser.add_argument("--scores-path", default="research/models/walk_forward_scores.parquet")
    state_parser.add_argument("--storage-root", default="storage")
    state_parser.add_argument("--trade-date", required=True)
    state_parser.add_argument("--mode", choices=("initial_entry", "continue", "historical"), default="continue")
    state_parser.add_argument("--previous-state-path", default="")
    state_parser.add_argument("--top-k", type=int, default=5)
    state_parser.add_argument("--rebalance-every", type=int, default=3)
    state_parser.add_argument("--lookback-window", type=int, default=20)
    state_parser.add_argument("--min-hold-bars", type=int, default=5)
    state_parser.add_argument("--keep-buffer", type=int, default=2)
    state_parser.add_argument("--min-turnover-names", type=int, default=2)
    state_parser.add_argument("--min-daily-amount", type=float, default=0.0)
    state_parser.add_argument("--max-close-price", type=float, default=0.0)
    state_parser.add_argument("--max-names-per-industry", type=int, default=0)
    state_parser.add_argument("--max-position-weight", type=float, default=0.0)
    state_parser.add_argument("--exit-policy", default="buffered_rank")
    state_parser.add_argument("--grace-rank-buffer", type=int, default=0)
    state_parser.add_argument("--grace-momentum-window", type=int, default=3)
    state_parser.add_argument("--grace-min-return", type=float, default=0.0)
    state_parser.add_argument("--trailing-stop-window", type=int, default=10)
    state_parser.add_argument("--trailing-stop-drawdown", type=float, default=0.12)
    state_parser.add_argument("--trailing-stop-min-gain", type=float, default=0.15)
    state_parser.add_argument("--score-reversal-confirm-days", type=int, default=3)
    state_parser.add_argument("--score-reversal-threshold", type=float, default=0.0)
    state_parser.add_argument("--hybrid-price-window", type=int, default=5)
    state_parser.add_argument("--hybrid-price-threshold", type=float, default=0.0)
    state_parser.add_argument("--strong-keep-extra-buffer", type=int, default=0)
    state_parser.add_argument("--strong-keep-momentum-window", type=int, default=5)
    state_parser.add_argument("--strong-keep-min-return", type=float, default=0.0)
    state_parser.add_argument("--strong-trim-slowdown", type=float, default=0.0)
    state_parser.add_argument("--strong-trim-momentum-window", type=int, default=5)
    state_parser.add_argument("--strong-trim-min-return", type=float, default=0.0)
    state_parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    state_parser.add_argument("--commission-rate", type=float, default=0.0003)
    state_parser.add_argument("--stamp-tax-rate", type=float, default=0.001)
    state_parser.add_argument("--slippage-rate", type=float, default=0.0005)
    state_parser.add_argument("--max-trade-participation-rate", type=float, default=0.0)
    state_parser.add_argument("--max-pending-days", type=int, default=0)
    state_parser.add_argument("--output-path", default="research/models/strategy_state.json")

    state_from_config_parser = subparsers.add_parser(
        "generate-strategy-state-from-config",
        help="Generate strategy state from a research TOML config with optional score and cash overrides",
    )
    state_from_config_parser.add_argument("config_path", help="Path to the TOML config file")
    state_from_config_parser.add_argument("--scores-path", default="", help="Optional override score parquet path")
    state_from_config_parser.add_argument("--trade-date", required=True)
    state_from_config_parser.add_argument("--mode", choices=("initial_entry", "continue", "historical"), default="historical")
    state_from_config_parser.add_argument("--previous-state-path", default="")
    state_from_config_parser.add_argument("--initial-cash", type=float, default=None)
    state_from_config_parser.add_argument("--output-path", default="research/models/strategy_state.json")

    pipeline_parser = subparsers.add_parser(
        "run-research-config",
        help="Run the standard factor -> model -> layer analysis -> model backtest pipeline from TOML",
    )
    pipeline_parser.add_argument("config_path", help="Path to the research TOML config")

    sweep_parser = subparsers.add_parser(
        "sweep-model-backtest",
        help="Run a light parameter sweep on model-driven portfolio settings",
    )
    sweep_parser.add_argument("--scores-path", required=True)
    sweep_parser.add_argument("--storage-root", default="storage")
    sweep_parser.add_argument("--start-date", required=True)
    sweep_parser.add_argument("--end-date", required=True)
    sweep_parser.add_argument("--top-k-values", default="5,8,10")
    sweep_parser.add_argument("--rebalance-every-values", default="3,5")
    sweep_parser.add_argument("--min-hold-bars-values", default="5,10")
    sweep_parser.add_argument("--keep-buffer", type=int, default=2)
    sweep_parser.add_argument("--min-turnover-names", type=int, default=3)
    sweep_parser.add_argument("--min-daily-amount", type=float, default=0.0)
    sweep_parser.add_argument("--max-names-per-industry", type=int, default=0)
    sweep_parser.add_argument("--lookback-window", type=int, default=20)
    sweep_parser.add_argument("--output-csv-path", default="research/models/model_sweep.csv")

    subparsers.add_parser("show-template", help="Print the default strategy template path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "validate":
            report = StrategyValidator().validate_file(args.path)
            print(f"VALID: {report.class_name} ({report.path})")
            return

        if args.command == "register":
            library = StrategyLibrary(args.library)
            record = library.register(args.path)
            print(f"REGISTERED: {record.strategy_id} -> {record.file_name}")
            return

        if args.command == "import-sqlite":
            datasets = SQLiteParquetImporter(
                sqlite_path=args.sqlite_path,
                storage_root=args.storage_root,
            ).run()
            for dataset in datasets:
                print(
                    f"IMPORTED: {dataset.name} rows={dataset.rows} "
                    f"range={dataset.min_date or '-'}..{dataset.max_date or '-'}"
                )
            return

        if args.command == "sync-tushare-sqlite":
            token = resolve_tushare_token(args.token)
            if not token:
                raise SystemExit("Missing Tushare token. Pass --token or set TUSHARE_TOKEN.")
            summary = TushareSQLiteSync(
                sqlite_path=args.sqlite_path,
                client=TushareClient(token),
            ).sync(start_date=args.start, end_date=args.end)
            print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
            return

        if args.command == "sync-tushare-benchmark":
            token = resolve_tushare_token(args.token)
            if not token:
                raise SystemExit("Missing Tushare token. Pass --token or set TUSHARE_TOKEN.")
            summary = TushareBenchmarkSync(
                client=TushareClient(token),
            ).sync(
                symbol=args.symbol,
                start_date=args.start,
                end_date=args.end,
                output_path=args.output_path,
            )
            print(json.dumps(summary.__dict__, ensure_ascii=False, indent=2))
            return

        if args.command == "list-universes":
            list_universes(args.storage_root)
            return

        if args.command == "run-backtest":
            run_backtest(
                backtest=BacktestConfig(
                    strategy_path=args.strategy_path,
                    start_date=date.fromisoformat(args.start_date),
                    end_date=date.fromisoformat(args.end_date),
                    universe=tuple(symbol.strip() for symbol in args.universe.split(",") if symbol.strip()),
                    initial_cash=args.initial_cash,
                    commission_rate=args.commission_rate,
                    stamp_tax_rate=args.stamp_tax_rate,
                    slippage_rate=args.slippage_rate,
                    max_trade_participation_rate=args.max_trade_participation_rate,
                    max_pending_days=args.max_pending_days,
                ),
                storage_root=args.storage_root,
                output_dir=args.output_dir,
            )
            return

        if args.command == "run-config":
            run_config = load_run_config(args.config_path)
            run_backtest(
                backtest=run_config.backtest,
                storage_root=run_config.storage_root,
                output_dir=run_config.output_dir,
            )
            return

        if args.command == "build-factors":
            symbols = tuple(symbol.strip() for symbol in args.symbols.split(",") if symbol.strip())
            output_path = args.output_path
            if not output_path and args.factor_spec_id and args.as_of_date:
                output_path = resolve_factor_snapshot_path(
                    args.factor_spec_id,
                    args.as_of_date,
                    universe_name=args.universe_name,
                    start_date=args.start_date or "",
                )
            if not output_path:
                output_path = "research/factors/basic_factor_panel.parquet"
            panel = FactorBuilder(
                FactorBuildConfig(
                    storage_root=args.storage_root,
                    output_path=output_path,
                    symbols=symbols,
                    universe_name=args.universe_name,
                    start_date=args.start_date,
                    as_of_date=args.as_of_date,
                )
            ).build()
            print(
                "FACTORS "
                f"rows={len(panel)} "
                f"symbols={panel['symbol'].nunique() if not panel.empty else 0} "
                f"output={output_path}"
            )
            return

        if args.command == "train-lgbm":
            metrics = train_lightgbm_model(
                ModelTrainConfig(
                    factor_panel_path=args.factor_panel_path,
                    output_scores_path=args.output_scores_path,
                    output_metrics_path=args.output_metrics_path,
                    label_column=args.label_column,
                    train_end_date=args.train_end_date,
                    test_start_date=args.test_start_date,
                    test_end_date=args.test_end_date,
                )
            )
            print(
                "MODEL "
                f"mae={metrics['mae']:.6f} "
                f"rmse={metrics['rmse']:.6f} "
                f"spearman_ic={metrics['spearman_ic']:.6f} "
                f"scores={args.output_scores_path}"
            )
            return

        if args.command == "train-lgbm-walk-forward":
            metrics = train_lightgbm_walk_forward(
                WalkForwardConfig(
                    factor_panel_path=args.factor_panel_path,
                    output_scores_path=args.output_scores_path,
                    output_metrics_path=args.output_metrics_path,
                    label_column=args.label_column,
                    train_window_months=args.train_window_months,
                    test_start_month=args.test_start_month,
                    test_end_month=args.test_end_month,
                )
            )
            print(
                "WALK_FORWARD "
                f"windows={metrics['window_count']} "
                f"mean_mae={metrics['mean_mae']:.6f} "
                f"mean_rmse={metrics['mean_rmse']:.6f} "
                f"mean_spearman_ic={metrics['mean_spearman_ic']:.6f} "
                f"scores={args.output_scores_path}"
            )
            return

        if args.command == "train-lgbm-walk-forward-from-config":
            config_path = resolve_research_config_path(args.config_path, args.factor_spec_id)
            metrics = train_walk_forward_from_config(
                config_path=config_path.as_posix(),
                factor_panel_path=args.factor_panel_path,
                test_start_month=args.test_start_month,
                test_end_month=args.test_end_month,
                output_scores_path=args.output_scores_path,
                output_metrics_path=args.output_metrics_path,
            )
            resolved_scores_path = args.output_scores_path or resolve_month_range_output_path(
                load_research_config(config_path).score_output_path,
                args.test_start_month,
                args.test_end_month,
            )
            print(
                "WALK_FORWARD "
                f"windows={metrics['window_count']} "
                f"mean_mae={metrics['mean_mae']:.6f} "
                f"mean_rmse={metrics['mean_rmse']:.6f} "
                f"mean_spearman_ic={metrics['mean_spearman_ic']:.6f} "
                f"scores={resolved_scores_path}"
            )
            return

        if args.command == "train-lgbm-walk-forward-as-of-date":
            metrics = train_lightgbm_walk_forward_as_of_date(
                WalkForwardAsOfDateConfig(
                    factor_panel_path=args.factor_panel_path,
                    output_scores_path=args.output_scores_path,
                    output_metrics_path=args.output_metrics_path,
                    label_column=args.label_column,
                    as_of_date=args.as_of_date,
                    train_window_months=args.train_window_months,
                )
            )
            print(
                "WALK_FORWARD_AS_OF_DATE "
                f"as_of_date={metrics['as_of_date']} "
                f"train_rows={metrics['train_rows']} "
                f"scored_rows={metrics['scored_rows']} "
                f"scores={args.output_scores_path}"
            )
            return

        if args.command == "train-lgbm-walk-forward-as-of-date-from-config":
            config_path = resolve_research_config_path(args.config_path, args.factor_spec_id)
            metrics = train_walk_forward_as_of_date_from_config(
                config_path=config_path.as_posix(),
                as_of_date=args.as_of_date,
                factor_panel_path=args.factor_panel_path,
                output_scores_path=args.output_scores_path,
                output_metrics_path=args.output_metrics_path,
            )
            resolved_scores_path = args.output_scores_path or resolve_dated_output_path(
                load_research_config(config_path).score_output_path,
                str(metrics["as_of_date"]),
            )
            print(
                "WALK_FORWARD_AS_OF_DATE "
                f"as_of_date={metrics['as_of_date']} "
                f"train_rows={metrics['train_rows']} "
                f"scored_rows={metrics['scored_rows']} "
                f"scores={resolved_scores_path}"
            )
            return

        if args.command == "train-lgbm-walk-forward-single-date-from-config":
            config_path = resolve_research_config_path(args.config_path, args.factor_spec_id)
            metrics = train_walk_forward_single_date_from_config(
                config_path=config_path.as_posix(),
                test_month=args.test_month,
                as_of_date=args.as_of_date,
                factor_panel_path=args.factor_panel_path,
                output_scores_path=args.output_scores_path,
                output_metrics_path=args.output_metrics_path,
            )
            resolved_scores_path = args.output_scores_path or resolve_dated_output_path(
                load_research_config(config_path).score_output_path,
                args.as_of_date,
            )
            print(
                "WALK_FORWARD_SINGLE_DATE "
                f"test_month={metrics['test_month']} "
                f"as_of_date={metrics['as_of_date']} "
                f"train_rows={metrics['train_rows']} "
                f"scored_rows={metrics['scored_rows']} "
                f"scores={resolved_scores_path}"
            )
            return

        if args.command == "train-lgbm-latest-inference":
            metrics = train_lightgbm_walk_forward_as_of_date(
                WalkForwardAsOfDateConfig(
                    factor_panel_path=args.factor_panel_path,
                    output_scores_path=args.output_scores_path,
                    output_metrics_path=args.output_metrics_path,
                    label_column=args.label_column,
                    as_of_date=args.inference_date,
                    train_window_months=args.train_window_months,
                )
            )
            print(
                "WALK_FORWARD_AS_OF_DATE "
                f"as_of_date={metrics['as_of_date']} "
                f"train_rows={metrics['train_rows']} "
                f"scored_rows={metrics['scored_rows']} "
                f"scores={args.output_scores_path}"
            )
            return

        if args.command == "run-model-backtest":
            run_model_backtest(
                scores_path=args.scores_path,
                storage_root=args.storage_root,
                start_date=args.start_date,
                end_date=args.end_date,
                top_k=args.top_k,
                rebalance_every=args.rebalance_every,
                lookback_window=args.lookback_window,
                min_hold_bars=args.min_hold_bars,
                keep_buffer=args.keep_buffer,
                min_turnover_names=args.min_turnover_names,
                min_daily_amount=args.min_daily_amount,
                max_close_price=args.max_close_price,
                max_names_per_industry=args.max_names_per_industry,
                max_position_weight=args.max_position_weight,
                exit_policy=args.exit_policy,
                grace_rank_buffer=args.grace_rank_buffer,
                grace_momentum_window=args.grace_momentum_window,
                grace_min_return=args.grace_min_return,
                trailing_stop_window=args.trailing_stop_window,
                trailing_stop_drawdown=args.trailing_stop_drawdown,
                trailing_stop_min_gain=args.trailing_stop_min_gain,
                score_reversal_confirm_days=args.score_reversal_confirm_days,
                score_reversal_threshold=args.score_reversal_threshold,
                hybrid_price_window=args.hybrid_price_window,
                hybrid_price_threshold=args.hybrid_price_threshold,
                strong_keep_extra_buffer=args.strong_keep_extra_buffer,
                strong_keep_momentum_window=args.strong_keep_momentum_window,
                strong_keep_min_return=args.strong_keep_min_return,
                strong_trim_slowdown=args.strong_trim_slowdown,
                strong_trim_momentum_window=args.strong_trim_momentum_window,
                strong_trim_min_return=args.strong_trim_min_return,
                initial_cash=args.initial_cash,
                commission_rate=args.commission_rate,
                stamp_tax_rate=args.stamp_tax_rate,
                slippage_rate=args.slippage_rate,
                max_trade_participation_rate=args.max_trade_participation_rate,
                max_pending_days=args.max_pending_days,
                output_dir=args.output_dir,
            )
            return

        if args.command == "analyze-score-layers":
            payload = analyze_score_layers(
                LayeredAnalysisConfig(
                    scores_path=args.scores_path,
                    output_path=args.output_path,
                    bins=args.bins,
                )
            )
            summary = payload["summary"]
            print(
                "LAYER_ANALYSIS "
                f"spread={summary['mean_top_bottom_spread']:.6f} "
                f"positive_ratio={summary['positive_spread_ratio']:.4f} "
                f"output={args.output_path}"
            )
            return

        if args.command == "analyze-trade-capacity":
            payload = analyze_trade_capacity(
                CapacityAnalysisConfig(
                    trades_path=args.trades_path,
                    storage_root=args.storage_root,
                    output_path=args.output_path,
                    base_capital=args.base_capital,
                    scale_capitals=tuple(float(item) for item in args.scale_capitals.split(",") if item),
                    participation_thresholds=tuple(
                        float(item) for item in args.participation_thresholds.split(",") if item
                    ),
                    top_trade_count=args.top_trade_count,
                )
            )
            first_scale = payload["by_scale"][0]
            print(
                "CAPACITY_ANALYSIS "
                f"scales={len(payload['by_scale'])} "
                f"base_capital={payload['summary']['base_capital']:.0f} "
                f"first_scale_max_participation={first_scale['participation_max']:.4f} "
                f"output={args.output_path}"
            )
            return

        if args.command == "compare-backtest-monthly":
            payload = compare_backtest_monthly_returns(
                MonthlyComparisonConfig(
                    result_dirs=tuple(item for item in args.result_dirs.split(",") if item),
                    labels=tuple(item for item in args.labels.split(",") if item),
                    output_path=args.output_path,
                )
            )
            print(
                "MONTHLY_COMPARISON "
                f"labels={len(payload['summary']['labels'])} "
                f"months={len(payload['by_month'])} "
                f"output={args.output_path}"
            )
            return

        if args.command == "analyze-risk-exposures":
            payload = analyze_monthly_risk_exposures(
                RiskExposureConfig(
                    result_dir=args.result_dir,
                    storage_root=args.storage_root,
                    output_path=args.output_path,
                    top_industries=args.top_industries,
                    volatility_window=args.volatility_window,
                )
            )
            print(
                "RISK_EXPOSURES "
                f"months={len(payload['by_month'])} "
                f"output={args.output_path}"
            )
            return

        if args.command == "analyze-start-date-robustness":
            payload = analyze_start_date_robustness(
                StartDateRobustnessConfig(
                    scores_path=args.scores_path,
                    storage_root=args.storage_root,
                    output_path=args.output_path,
                    analysis_start_date=args.analysis_start_date,
                    analysis_end_date=args.analysis_end_date,
                    holding_months=args.holding_months,
                    cadence=args.cadence,
                    universe_name=args.universe_name,
                    top_k=args.top_k,
                    rebalance_every=args.rebalance_every,
                    lookback_window=args.lookback_window,
                    min_hold_bars=args.min_hold_bars,
                    keep_buffer=args.keep_buffer,
                    min_turnover_names=args.min_turnover_names,
                    min_daily_amount=args.min_daily_amount,
                    max_close_price=args.max_close_price,
                    max_names_per_industry=args.max_names_per_industry,
                    max_position_weight=args.max_position_weight,
                    exit_policy=args.exit_policy,
                    grace_rank_buffer=args.grace_rank_buffer,
                    grace_momentum_window=args.grace_momentum_window,
                    grace_min_return=args.grace_min_return,
                    trailing_stop_window=args.trailing_stop_window,
                    trailing_stop_drawdown=args.trailing_stop_drawdown,
                    trailing_stop_min_gain=args.trailing_stop_min_gain,
                    score_reversal_confirm_days=args.score_reversal_confirm_days,
                    score_reversal_threshold=args.score_reversal_threshold,
                    hybrid_price_window=args.hybrid_price_window,
                    hybrid_price_threshold=args.hybrid_price_threshold,
                    strong_keep_extra_buffer=args.strong_keep_extra_buffer,
                    strong_keep_momentum_window=args.strong_keep_momentum_window,
                    strong_keep_min_return=args.strong_keep_min_return,
                    strong_trim_slowdown=args.strong_trim_slowdown,
                    strong_trim_momentum_window=args.strong_trim_momentum_window,
                    strong_trim_min_return=args.strong_trim_min_return,
                    initial_cash=args.initial_cash,
                    commission_rate=args.commission_rate,
                    stamp_tax_rate=args.stamp_tax_rate,
                    slippage_rate=args.slippage_rate,
                    max_trade_participation_rate=args.max_trade_participation_rate,
                    max_pending_days=args.max_pending_days,
                )
            )
            print(
                "START_DATE_ROBUSTNESS "
                f"samples={payload['summary']['sample_count']} "
                f"holding_months={payload['summary']['holding_months']} "
                f"win_rate={payload['summary']['win_rate']:.4f} "
                f"output={args.output_path}"
            )
            return

        if args.command == "analyze-start-date-robustness-from-config":
            payload = analyze_start_date_robustness_from_config(
                config_path=args.config_path,
                factor_spec_id=args.factor_spec_id,
                scores_path=args.scores_path,
                analysis_start_date=args.analysis_start_date,
                analysis_end_date=args.analysis_end_date,
                holding_months=args.holding_months,
                cadence=args.cadence,
                output_path=args.output_path,
            )
            print(
                "START_DATE_ROBUSTNESS "
                f"samples={payload['summary']['sample_count']} "
                f"holding_months={payload['summary']['holding_months']} "
                f"win_rate={payload['summary']['win_rate']:.4f} "
                f"output={payload['output_path']}"
            )
            return

        if args.command == "generate-premarket-reference":
            payload = generate_premarket_reference(
                PremarketReferenceConfig(
                    scores_path=args.scores_path,
                    storage_root=args.storage_root,
                    output_path=args.output_path,
                    trade_date=args.trade_date,
                    top_k=args.top_k,
                    rebalance_every=args.rebalance_every,
                    lookback_window=args.lookback_window,
                    min_hold_bars=args.min_hold_bars,
                    keep_buffer=args.keep_buffer,
                    min_turnover_names=args.min_turnover_names,
                    min_daily_amount=args.min_daily_amount,
                    max_close_price=args.max_close_price,
                    max_names_per_industry=args.max_names_per_industry,
                    max_position_weight=args.max_position_weight,
                    exit_policy=args.exit_policy,
                    grace_rank_buffer=args.grace_rank_buffer,
                    grace_momentum_window=args.grace_momentum_window,
                    grace_min_return=args.grace_min_return,
                    trailing_stop_window=args.trailing_stop_window,
                    trailing_stop_drawdown=args.trailing_stop_drawdown,
                    trailing_stop_min_gain=args.trailing_stop_min_gain,
                    score_reversal_confirm_days=args.score_reversal_confirm_days,
                    score_reversal_threshold=args.score_reversal_threshold,
                    hybrid_price_window=args.hybrid_price_window,
                    hybrid_price_threshold=args.hybrid_price_threshold,
                    strong_keep_extra_buffer=args.strong_keep_extra_buffer,
                    strong_keep_momentum_window=args.strong_keep_momentum_window,
                    strong_keep_min_return=args.strong_keep_min_return,
                    strong_trim_slowdown=args.strong_trim_slowdown,
                    strong_trim_momentum_window=args.strong_trim_momentum_window,
                    strong_trim_min_return=args.strong_trim_min_return,
                    initial_cash=args.initial_cash,
                    commission_rate=args.commission_rate,
                    stamp_tax_rate=args.stamp_tax_rate,
                    slippage_rate=args.slippage_rate,
                    max_trade_participation_rate=args.max_trade_participation_rate,
                    max_pending_days=args.max_pending_days,
                )
            )
            print(
                "PREMARKET_REFERENCE "
                f"signal_date={payload['summary']['signal_date']} "
                f"execution_date={payload['summary']['execution_date']} "
                f"actions={len(payload['actions'])} "
                f"output={args.output_path}"
            )
            return

        if args.command == "generate-premarket-reference-from-config":
            config_path = resolve_research_config_path(args.config_path, args.factor_spec_id)
            output_path = args.output_path or resolve_premarket_output_path(
                load_research_config(config_path).factor_spec_id,
                args.trade_date,
            )
            payload = generate_premarket_reference_from_config(
                config_path=config_path.as_posix(),
                scores_path=args.scores_path,
                trade_date=args.trade_date,
                output_path=args.output_path,
            )
            print(
                "PREMARKET_REFERENCE "
                f"signal_date={payload['summary']['signal_date']} "
                f"execution_date={payload['summary']['execution_date']} "
                f"actions={len(payload['actions'])} "
                f"output={output_path}"
            )
            return

        if args.command == "generate-strategy-state":
            payload = generate_strategy_state(
                StrategyStateConfig(
                    scores_path=args.scores_path,
                    storage_root=args.storage_root,
                    output_path=args.output_path,
                    trade_date=args.trade_date,
                    mode=args.mode,
                    previous_state_path=args.previous_state_path,
                    top_k=args.top_k,
                    rebalance_every=args.rebalance_every,
                    lookback_window=args.lookback_window,
                    min_hold_bars=args.min_hold_bars,
                    keep_buffer=args.keep_buffer,
                    min_turnover_names=args.min_turnover_names,
                    min_daily_amount=args.min_daily_amount,
                    max_close_price=args.max_close_price,
                    max_names_per_industry=args.max_names_per_industry,
                    max_position_weight=args.max_position_weight,
                    exit_policy=args.exit_policy,
                    grace_rank_buffer=args.grace_rank_buffer,
                    grace_momentum_window=args.grace_momentum_window,
                    grace_min_return=args.grace_min_return,
                    trailing_stop_window=args.trailing_stop_window,
                    trailing_stop_drawdown=args.trailing_stop_drawdown,
                    trailing_stop_min_gain=args.trailing_stop_min_gain,
                    score_reversal_confirm_days=args.score_reversal_confirm_days,
                    score_reversal_threshold=args.score_reversal_threshold,
                    hybrid_price_window=args.hybrid_price_window,
                    hybrid_price_threshold=args.hybrid_price_threshold,
                    strong_keep_extra_buffer=args.strong_keep_extra_buffer,
                    strong_keep_momentum_window=args.strong_keep_momentum_window,
                    strong_keep_min_return=args.strong_keep_min_return,
                    strong_trim_slowdown=args.strong_trim_slowdown,
                    strong_trim_momentum_window=args.strong_trim_momentum_window,
                    strong_trim_min_return=args.strong_trim_min_return,
                    initial_cash=args.initial_cash,
                    commission_rate=args.commission_rate,
                    stamp_tax_rate=args.stamp_tax_rate,
                    slippage_rate=args.slippage_rate,
                    max_trade_participation_rate=args.max_trade_participation_rate,
                    max_pending_days=args.max_pending_days,
                )
            )
            print(
                "STRATEGY_STATE "
                f"signal_date={payload['summary']['signal_date']} "
                f"execution_date={payload['summary']['execution_date']} "
                f"mode={payload['summary']['state_mode']} "
                f"positions={len(payload['next_state']['positions'])} "
                f"output={args.output_path}"
            )
            return

        if args.command == "generate-strategy-state-from-config":
            payload = generate_strategy_state_from_config(
                config_path=args.config_path,
                trade_date=args.trade_date,
                output_path=args.output_path,
                scores_path=args.scores_path,
                initial_cash=args.initial_cash,
                mode=args.mode,
                previous_state_path=args.previous_state_path,
            )
            print(
                "STRATEGY_STATE "
                f"signal_date={payload['summary']['signal_date']} "
                f"execution_date={payload['summary']['execution_date']} "
                f"mode={payload['summary']['state_mode']} "
                f"positions={payload['summary']['current_position_count']} "
                f"output={args.output_path}"
            )
            return

        if args.command == "run-research-config":
            run_research_pipeline(args.config_path)
            return

        if args.command == "sweep-model-backtest":
            rows = run_model_sweep(
                SweepConfig(
                    scores_path=args.scores_path,
                    storage_root=args.storage_root,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    output_csv_path=args.output_csv_path,
                    top_k_values=tuple(int(item) for item in args.top_k_values.split(",") if item),
                    rebalance_every_values=tuple(int(item) for item in args.rebalance_every_values.split(",") if item),
                    min_hold_bars_values=tuple(int(item) for item in args.min_hold_bars_values.split(",") if item),
                    keep_buffer=args.keep_buffer,
                    min_turnover_names=args.min_turnover_names,
                    min_daily_amount=args.min_daily_amount,
                    max_names_per_industry=args.max_names_per_industry,
                    lookback_window=args.lookback_window,
                )
            )
            best = max(rows, key=lambda item: float(item["sharpe_ratio"]))
            print(
                "SWEEP "
                f"rows={len(rows)} "
                f"best_sharpe={best['sharpe_ratio']:.4f} "
                f"best_top_k={best['top_k']} "
                f"best_rebalance_every={best['rebalance_every']} "
                f"best_min_hold_bars={best['min_hold_bars']} "
                f"output={args.output_csv_path}"
            )
            return

        if args.command == "show-template":
            template = Path("examples") / "strategy_template.py"
            print(template.as_posix())
            return

    except StrategyValidationError as exc:
        print(f"INVALID: {exc}")
        raise SystemExit(1) from exc


def train_walk_forward_from_config(
    config_path: str,
    factor_panel_path: str,
    test_start_month: str,
    test_end_month: str,
    output_scores_path: str = "",
    output_metrics_path: str = "",
) -> dict[str, float | int | str]:
    config = load_research_config(config_path)
    resolved_factor_panel_path = factor_panel_path or config.factor_snapshot_path
    resolved_scores_path = output_scores_path or resolve_month_range_output_path(
        config.score_output_path,
        test_start_month,
        test_end_month,
    )
    resolved_metrics_path = output_metrics_path or resolve_month_range_output_path(
        config.metric_output_path,
        test_start_month,
        test_end_month,
    )
    return train_lightgbm_walk_forward(
        WalkForwardConfig(
            factor_panel_path=resolved_factor_panel_path,
            output_scores_path=resolved_scores_path,
            output_metrics_path=resolved_metrics_path,
            label_column=config.label_column,
            train_window_months=config.train_window_months,
            test_start_month=test_start_month,
            test_end_month=test_end_month,
        )
    )


def train_walk_forward_as_of_date_from_config(
    config_path: str,
    as_of_date: str = "",
    factor_panel_path: str = "",
    output_scores_path: str = "",
    output_metrics_path: str = "",
) -> dict[str, float | int | str]:
    config = load_research_config(config_path)
    resolved_as_of_date = as_of_date or infer_as_of_date_from_factor_panel(factor_panel_path)
    resolved_factor_panel_path = factor_panel_path or resolve_factor_snapshot_path(
        factor_spec_id=config.factor_spec_id,
        as_of_date=resolved_as_of_date,
        universe_name=config.factor_universe_name,
        start_date=config.factor_start_date,
    )
    resolved_scores_path = output_scores_path or resolve_dated_output_path(config.score_output_path, resolved_as_of_date)
    resolved_metrics_path = output_metrics_path or resolve_dated_output_path(config.metric_output_path, resolved_as_of_date)
    return train_lightgbm_walk_forward_as_of_date(
        WalkForwardAsOfDateConfig(
            factor_panel_path=resolved_factor_panel_path,
            output_scores_path=resolved_scores_path,
            output_metrics_path=resolved_metrics_path,
            label_column=config.label_column,
            as_of_date=resolved_as_of_date,
            train_window_months=config.train_window_months,
        )
    )


def train_walk_forward_single_date_from_config(
    config_path: str,
    test_month: str,
    as_of_date: str,
    factor_panel_path: str = "",
    output_scores_path: str = "",
    output_metrics_path: str = "",
) -> dict[str, float | int | str]:
    config = load_research_config(config_path)
    resolved_factor_panel_path = factor_panel_path or config.factor_snapshot_path
    resolved_scores_path = output_scores_path or resolve_dated_output_path(config.score_output_path, as_of_date)
    resolved_metrics_path = output_metrics_path or resolve_dated_output_path(config.metric_output_path, as_of_date)
    return train_lightgbm_walk_forward_single_date(
        WalkForwardSingleDateConfig(
            factor_panel_path=resolved_factor_panel_path,
            output_scores_path=resolved_scores_path,
            output_metrics_path=resolved_metrics_path,
            label_column=config.label_column,
            test_month=test_month,
            as_of_date=as_of_date,
            train_window_months=config.train_window_months,
        )
    )


def infer_as_of_date_from_factor_panel(factor_panel_path: str) -> str:
    if not factor_panel_path:
        raise ValueError("as_of_date is required when factor_panel_path is not provided")

    match = re.search(r"asof_(\d{4}-\d{2}-\d{2})\.parquet$", factor_panel_path)
    if match:
        return match.group(1)

    frame = pd.read_parquet(factor_panel_path, columns=["trade_date"])
    if frame.empty:
        raise ValueError("factor panel is empty; cannot infer as_of_date")
    return pd.to_datetime(frame["trade_date"]).max().date().isoformat()


def generate_strategy_state_from_config(
    config_path: str,
    trade_date: str,
    output_path: str,
    scores_path: str = "",
    initial_cash: float | None = None,
    mode: str = "historical",
    previous_state_path: str = "",
) -> dict[str, object]:
    config = load_research_config(config_path)
    resolved_scores_path = scores_path or config.score_output_path
    resolved_initial_cash = config.initial_cash if initial_cash is None else initial_cash
    return generate_strategy_state(
        StrategyStateConfig(
            scores_path=resolved_scores_path,
            storage_root=config.storage_root,
            output_path=output_path,
            trade_date=trade_date,
            universe_name=config.factor_universe_name,
            mode=mode,
            previous_state_path=previous_state_path,
            top_k=config.top_k,
            rebalance_every=config.rebalance_every,
            lookback_window=config.lookback_window,
            min_hold_bars=config.min_hold_bars,
            keep_buffer=config.keep_buffer,
            min_turnover_names=config.min_turnover_names,
            min_daily_amount=config.min_daily_amount,
            max_close_price=config.max_close_price,
            max_names_per_industry=config.max_names_per_industry,
            max_position_weight=config.max_position_weight,
            exit_policy=config.exit_policy,
            grace_rank_buffer=config.grace_rank_buffer,
            grace_momentum_window=config.grace_momentum_window,
            grace_min_return=config.grace_min_return,
            trailing_stop_window=config.trailing_stop_window,
            trailing_stop_drawdown=config.trailing_stop_drawdown,
            trailing_stop_min_gain=config.trailing_stop_min_gain,
            score_reversal_confirm_days=config.score_reversal_confirm_days,
            score_reversal_threshold=config.score_reversal_threshold,
            hybrid_price_window=config.hybrid_price_window,
            hybrid_price_threshold=config.hybrid_price_threshold,
            strong_keep_extra_buffer=config.strong_keep_extra_buffer,
            strong_keep_momentum_window=config.strong_keep_momentum_window,
            strong_keep_min_return=config.strong_keep_min_return,
            strong_trim_slowdown=config.strong_trim_slowdown,
            strong_trim_momentum_window=config.strong_trim_momentum_window,
            strong_trim_min_return=config.strong_trim_min_return,
            initial_cash=resolved_initial_cash,
            commission_rate=config.commission_rate,
            stamp_tax_rate=config.stamp_tax_rate,
            slippage_rate=config.slippage_rate,
            max_trade_participation_rate=config.max_trade_participation_rate,
            max_pending_days=config.max_pending_days,
        )
    )


if __name__ == "__main__":
    main()
