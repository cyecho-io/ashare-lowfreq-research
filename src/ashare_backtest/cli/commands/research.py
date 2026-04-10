from __future__ import annotations

from pathlib import Path

from ashare_backtest.cli.research_config import (
    load_research_config,
    resolve_dated_output_path,
    resolve_research_config_path,
    resolve_research_run_output_paths,
)
from ashare_backtest.factors import FactorBuildConfig, FactorBuilder
from ashare_backtest.research import (
    DEFAULT_FEATURE_COLUMNS,
    LayeredAnalysisConfig,
    PremarketReferenceConfig,
    StartDateRobustnessConfig,
    WalkForwardConfig,
    analyze_score_layers,
    analyze_start_date_robustness,
    generate_premarket_reference,
    train_lightgbm_walk_forward,
)


def run_research_pipeline(config_path: str, output_dir: str | Path | None = None) -> dict[str, object]:
    config = load_research_config(config_path)
    resolved_config_path = Path(config_path).resolve()
    resolved_output_paths = (
        resolve_research_run_output_paths(config, output_dir) if output_dir is not None else None
    )
    factor_snapshot_path = (
        resolved_output_paths.factor_snapshot_path if resolved_output_paths is not None else config.factor_snapshot_path
    )
    score_output_path = (
        resolved_output_paths.score_output_path if resolved_output_paths is not None else config.score_output_path
    )
    metric_output_path = (
        resolved_output_paths.metric_output_path if resolved_output_paths is not None else config.metric_output_path
    )
    layer_output_path = (
        resolved_output_paths.layer_output_path if resolved_output_paths is not None else config.layer_output_path
    )

    print(
        "RESEARCH_PIPELINE "
        f"config={resolved_config_path.as_posix()} "
        f"factor_spec_id={config.factor_spec_id} "
        f"test_start_month={config.test_start_month} "
        f"test_end_month={config.test_end_month}"
    )
    print(
        "RESEARCH_STEP "
        f"name=build_factors "
        f"output={factor_snapshot_path}"
    )

    FactorBuilder(
        FactorBuildConfig(
            storage_root=config.storage_root,
            output_path=factor_snapshot_path,
            universe_name=config.factor_universe_name,
            start_date=config.factor_start_date,
            as_of_date=config.factor_as_of_date,
        )
    ).build()
    print(
        "RESEARCH_STEP_DONE "
        f"name=build_factors "
        f"output={factor_snapshot_path}"
    )

    print(
        "RESEARCH_STEP "
        f"name=train_walk_forward "
        f"factor_panel={factor_snapshot_path} "
        f"scores={score_output_path} "
        f"metrics={metric_output_path}"
    )

    training_metrics = train_lightgbm_walk_forward(
        WalkForwardConfig(
            factor_panel_path=factor_snapshot_path,
            output_scores_path=score_output_path,
            output_metrics_path=metric_output_path,
            label_column=config.label_column,
            feature_columns=config.feature_columns or tuple(DEFAULT_FEATURE_COLUMNS),
            train_window_months=config.train_window_months,
            validation_window_months=config.validation_window_months,
            test_start_month=config.test_start_month,
            test_end_month=config.test_end_month,
        )
    )
    print(
        "RESEARCH_STEP_DONE "
        f"name=train_walk_forward "
        f"windows={training_metrics['window_count']} "
        f"mean_spearman_ic={training_metrics['mean_spearman_ic']:.6f} "
        f"scores={score_output_path}"
    )

    print(
        "RESEARCH_STEP "
        f"name=analyze_score_layers "
        f"scores={score_output_path} "
        f"output={layer_output_path}"
    )
    layer_payload = analyze_score_layers(
        LayeredAnalysisConfig(
            scores_path=score_output_path,
            output_path=layer_output_path,
            bins=5,
        )
    )
    layer_summary = dict(layer_payload.get("summary", {}))
    print(
        "RESEARCH_STEP_DONE "
        f"name=analyze_score_layers "
        f"rows={layer_summary.get('rows', 0)} "
        f"mean_top_bottom_spread={float(layer_summary.get('mean_top_bottom_spread', 0.0)):.6f} "
        f"output={layer_output_path}"
    )
    return {
        "backend": "native",
        "model": "lgbm",
        "config_id": config.factor_spec_id,
        "config_path": resolved_config_path.as_posix(),
        "factor_path": factor_snapshot_path,
        "scores_path": score_output_path,
        "metrics_path": metric_output_path,
        "layer_output_path": layer_output_path,
        "configured_factor_path": config.factor_snapshot_path,
        "configured_scores_path": config.score_output_path,
        "configured_metrics_path": config.metric_output_path,
        "configured_layer_output_path": config.layer_output_path,
        "training_metrics": training_metrics,
        "layer_summary": layer_summary,
    }


def resolve_month_range_output_path(base_path: str | Path, test_start_month: str, test_end_month: str) -> str:
    path = Path(base_path)
    suffix = f"{test_start_month}_to_{test_end_month}".replace(":", "-")
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}").as_posix()


def resolve_premarket_output_path(factor_spec_id: str, trade_date: str) -> str:
    return (Path("research/models") / f"premarket_reference_{factor_spec_id}_{trade_date}.json").as_posix()


def resolve_start_date_robustness_output_path(
    factor_spec_id: str,
    analysis_start_date: str,
    analysis_end_date: str,
    holding_months: int,
    cadence: str,
) -> str:
    return (
        Path("research/models")
        / f"start_date_robustness_{factor_spec_id}_{analysis_start_date}_to_{analysis_end_date}_{holding_months}m_{cadence}.json"
    ).as_posix()


def generate_premarket_reference_from_config(
    config_path: str,
    scores_path: str,
    trade_date: str,
    output_path: str = "",
) -> dict[str, object]:
    config = load_research_config(config_path)
    resolved_output_path = output_path or resolve_premarket_output_path(config.factor_spec_id, trade_date)
    return generate_premarket_reference(
        PremarketReferenceConfig(
            scores_path=scores_path,
            storage_root=config.storage_root,
            output_path=resolved_output_path,
            trade_date=trade_date,
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
            initial_cash=config.initial_cash,
            commission_rate=config.commission_rate,
            stamp_tax_rate=config.stamp_tax_rate,
            slippage_rate=config.slippage_rate,
            max_trade_participation_rate=config.max_trade_participation_rate,
            max_pending_days=config.max_pending_days,
        )
    )


def analyze_start_date_robustness_from_config(
    config_path: str = "",
    factor_spec_id: str = "",
    scores_path: str = "",
    analysis_start_date: str = "",
    analysis_end_date: str = "",
    holding_months: int = 8,
    cadence: str = "monthly",
    output_path: str = "",
) -> dict[str, object]:
    resolved_config_path = resolve_research_config_path(config_path, factor_spec_id)
    config = load_research_config(resolved_config_path)
    resolved_scores_path = scores_path or config.score_output_path
    resolved_analysis_start_date = analysis_start_date or config.backtest_start_date
    resolved_analysis_end_date = analysis_end_date or config.backtest_end_date
    resolved_output_path = output_path or resolve_start_date_robustness_output_path(
        config.factor_spec_id,
        resolved_analysis_start_date,
        resolved_analysis_end_date,
        holding_months,
        cadence,
    )
    payload = analyze_start_date_robustness(
        StartDateRobustnessConfig(
            scores_path=resolved_scores_path,
            storage_root=config.storage_root,
            output_path=resolved_output_path,
            analysis_start_date=resolved_analysis_start_date,
            analysis_end_date=resolved_analysis_end_date,
            holding_months=holding_months,
            cadence=cadence,
            universe_name=config.factor_universe_name,
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
            initial_cash=config.initial_cash,
            commission_rate=config.commission_rate,
            stamp_tax_rate=config.stamp_tax_rate,
            slippage_rate=config.slippage_rate,
            max_trade_participation_rate=config.max_trade_participation_rate,
            max_pending_days=config.max_pending_days,
        )
    )
    payload["output_path"] = resolved_output_path
    return payload
