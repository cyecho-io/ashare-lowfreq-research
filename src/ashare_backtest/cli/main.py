from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from ashare_backtest.data import DEFAULT_SQLITE_SOURCE, ParquetDataProvider, SQLiteParquetImporter
from ashare_backtest.factors import FactorBuildConfig, FactorBuilder
from ashare_backtest.research import (
    LayeredAnalysisConfig,
    ModelTrainConfig,
    ScoreStrategyConfig,
    ScoreTopKStrategy,
    SweepConfig,
    WalkForwardConfig,
    analyze_score_layers,
    run_model_sweep,
    train_lightgbm_model,
    train_lightgbm_walk_forward,
)
from ashare_backtest.engine import BacktestEngine
from ashare_backtest.engine.loader import load_strategy
from ashare_backtest.protocol import BacktestConfig
from ashare_backtest.registry import StrategyLibrary
from ashare_backtest.reporting import export_backtest_result
from ashare_backtest.sandbox import StrategyValidationError, StrategyValidator
from .config import load_run_config
from .research_config import load_research_config


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
    run_parser.add_argument("--output-dir", default="results/latest")

    run_config_parser = subparsers.add_parser("run-config", help="Run a backtest from a TOML config file")
    run_config_parser.add_argument("config_path", help="Path to the TOML config file")

    factor_parser = subparsers.add_parser("build-factors", help="Build a basic factor panel from Parquet bars")
    factor_parser.add_argument("--storage-root", default="storage", help="Parquet storage root")
    factor_parser.add_argument("--output-path", default="research/factors/basic_factor_panel.parquet")
    factor_parser.add_argument("--symbols", default="", help="Optional comma-separated symbols")
    factor_parser.add_argument("--start-date", default=None, help="Optional start date, YYYY-MM-DD")
    factor_parser.add_argument("--end-date", default=None, help="Optional end date, YYYY-MM-DD")

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
    model_parser.add_argument("--output-scores-path", default="research/models/latest_scores.parquet")
    model_parser.add_argument("--output-metrics-path", default="research/models/latest_metrics.json")

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

    score_bt_parser = subparsers.add_parser(
        "run-model-backtest",
        help="Run a backtest driven by model score parquet output",
    )
    score_bt_parser.add_argument("--scores-path", default="research/models/latest_scores.parquet")
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
    score_bt_parser.add_argument("--max-names-per-industry", type=int, default=0)
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
    score_bt_parser.add_argument("--output-dir", default="results/model_score_backtest")

    layer_parser = subparsers.add_parser(
        "analyze-score-layers",
        help="Analyze layered forward returns based on model scores",
    )
    layer_parser.add_argument("--scores-path", required=True)
    layer_parser.add_argument("--output-path", default="research/models/layer_analysis.json")
    layer_parser.add_argument("--bins", type=int, default=5)

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
            panel = FactorBuilder(
                FactorBuildConfig(
                    storage_root=args.storage_root,
                    output_path=args.output_path,
                    symbols=symbols,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
            ).build()
            print(
                "FACTORS "
                f"rows={len(panel)} "
                f"symbols={panel['symbol'].nunique() if not panel.empty else 0} "
                f"output={args.output_path}"
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
                max_names_per_industry=args.max_names_per_industry,
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


def run_backtest(backtest: BacktestConfig, storage_root: str, output_dir: str) -> None:
    provider = ParquetDataProvider(storage_root)
    strategy = load_strategy(backtest.strategy_path)
    provider.preload(
        symbols=backtest.universe,
        start_date=backtest.start_date,
        end_date=backtest.end_date,
        lookback=strategy.metadata.lookback_window,
    )
    engine = BacktestEngine(provider)
    result = engine.run(backtest)
    export_backtest_result(result, output_dir)
    print(
        "RESULT "
        f"total_return={result.total_return:.4f} "
        f"annual_return={result.annual_return:.4f} "
        f"max_drawdown={result.max_drawdown:.4f} "
        f"sharpe={result.sharpe_ratio:.4f} "
        f"trades={len(result.trades)} "
        f"output={output_dir}"
    )


def run_model_backtest(
    scores_path: str,
    storage_root: str,
    start_date: str,
    end_date: str,
    top_k: int,
    rebalance_every: int,
    lookback_window: int,
    min_hold_bars: int,
    keep_buffer: int,
    min_turnover_names: int,
    min_daily_amount: float,
    max_names_per_industry: int,
    exit_policy: str,
    grace_rank_buffer: int,
    grace_momentum_window: int,
    grace_min_return: float,
    trailing_stop_window: int,
    trailing_stop_drawdown: float,
    trailing_stop_min_gain: float,
    score_reversal_confirm_days: int,
    score_reversal_threshold: float,
    hybrid_price_window: int,
    hybrid_price_threshold: float,
    strong_keep_extra_buffer: int,
    strong_keep_momentum_window: int,
    strong_keep_min_return: float,
    strong_trim_slowdown: float,
    strong_trim_momentum_window: int,
    strong_trim_min_return: float,
    initial_cash: float,
    commission_rate: float,
    stamp_tax_rate: float,
    slippage_rate: float,
    output_dir: str,
) -> None:
    import pandas as pd

    scores = pd.read_parquet(scores_path)
    universe = tuple(sorted(scores["symbol"].astype(str).unique().tolist()))
    provider = ParquetDataProvider(storage_root)
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=top_k,
            rebalance_every=rebalance_every,
            lookback_window=lookback_window,
            min_hold_bars=min_hold_bars,
            keep_buffer=keep_buffer,
            min_turnover_names=min_turnover_names,
            min_daily_amount=min_daily_amount,
            max_names_per_industry=max_names_per_industry,
            exit_policy=exit_policy,
            grace_rank_buffer=grace_rank_buffer,
            grace_momentum_window=grace_momentum_window,
            grace_min_return=grace_min_return,
            trailing_stop_window=trailing_stop_window,
            trailing_stop_drawdown=trailing_stop_drawdown,
            trailing_stop_min_gain=trailing_stop_min_gain,
            score_reversal_confirm_days=score_reversal_confirm_days,
            score_reversal_threshold=score_reversal_threshold,
            hybrid_price_window=hybrid_price_window,
            hybrid_price_threshold=hybrid_price_threshold,
            strong_keep_extra_buffer=strong_keep_extra_buffer,
            strong_keep_momentum_window=strong_keep_momentum_window,
            strong_keep_min_return=strong_keep_min_return,
            strong_trim_slowdown=strong_trim_slowdown,
            strong_trim_momentum_window=strong_trim_momentum_window,
            strong_trim_min_return=strong_trim_min_return,
        )
    )
    backtest = BacktestConfig(
        strategy_path="__model_score__",
        start_date=date.fromisoformat(start_date),
        end_date=date.fromisoformat(end_date),
        universe=universe,
        initial_cash=initial_cash,
        commission_rate=commission_rate,
        stamp_tax_rate=stamp_tax_rate,
        slippage_rate=slippage_rate,
    )
    provider.preload(
        symbols=backtest.universe,
        start_date=backtest.start_date,
        end_date=backtest.end_date,
        lookback=strategy.metadata.lookback_window,
    )
    engine = BacktestEngine(provider)
    result = engine.run_with_strategy(backtest, strategy)
    export_backtest_result(result, output_dir)
    print(
        "MODEL_RESULT "
        f"total_return={result.total_return:.4f} "
        f"annual_return={result.annual_return:.4f} "
        f"max_drawdown={result.max_drawdown:.4f} "
        f"sharpe={result.sharpe_ratio:.4f} "
        f"trades={len(result.trades)} "
        f"output={output_dir}"
    )


def run_research_pipeline(config_path: str) -> None:
    config = load_research_config(config_path)

    FactorBuilder(
        FactorBuildConfig(
            storage_root=config.storage_root,
            output_path=config.factor_output_path,
            start_date=config.factor_start_date,
            end_date=config.factor_end_date,
        )
    ).build()

    train_lightgbm_walk_forward(
        WalkForwardConfig(
            factor_panel_path=config.factor_output_path,
            output_scores_path=config.score_output_path,
            output_metrics_path=config.metric_output_path,
            label_column=config.label_column,
            train_window_months=config.train_window_months,
            test_start_month=config.test_start_month,
            test_end_month=config.test_end_month,
        )
    )

    analyze_score_layers(
        LayeredAnalysisConfig(
            scores_path=config.score_output_path,
            output_path=config.layer_output_path,
            bins=5,
        )
    )

    run_model_backtest(
        scores_path=config.score_output_path,
        storage_root=config.storage_root,
        start_date=config.backtest_start_date,
        end_date=config.backtest_end_date,
        top_k=config.top_k,
        rebalance_every=config.rebalance_every,
        lookback_window=config.lookback_window,
        min_hold_bars=config.min_hold_bars,
        keep_buffer=config.keep_buffer,
        min_turnover_names=config.min_turnover_names,
        min_daily_amount=config.min_daily_amount,
        max_names_per_industry=config.max_names_per_industry,
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
        initial_cash=config.initial_cash,
        commission_rate=config.commission_rate,
        stamp_tax_rate=config.stamp_tax_rate,
        slippage_rate=config.slippage_rate,
        output_dir=config.model_backtest_output_dir,
    )


if __name__ == "__main__":
    main()
