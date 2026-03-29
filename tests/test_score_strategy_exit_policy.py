from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ashare_backtest.protocol import Bar, Position, StrategyContext
from ashare_backtest.research.score_strategy import ScoreStrategyConfig, ScoreTopKStrategy


def _write_inputs(tmp_path, predictions: list[tuple[str, float]]) -> tuple[str, str]:
    scores_path = tmp_path / "scores.parquet"
    instruments_path = tmp_path / "storage" / "parquet" / "instruments" / "ashare_instruments.parquet"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"trade_date": "2025-01-08", "symbol": symbol, "prediction": prediction}
            for symbol, prediction in predictions
        ]
    ).to_parquet(scores_path, index=False)
    pd.DataFrame(
        [
            {"symbol": symbol, "industry_level_1": f"industry_{index}"}
            for index, (symbol, _) in enumerate(predictions)
        ]
    ).to_parquet(instruments_path, index=False)
    return scores_path.as_posix(), (tmp_path / "storage").as_posix()


def _history(symbol: str, closes: list[float]) -> list[Bar]:
    start = date(2025, 1, 6)
    return [
        Bar(
            symbol=symbol,
            trade_date=start + timedelta(days=index),
            open=close,
            high=close,
            low=close,
            close=close,
            amount=1_000_000.0,
        )
        for index, close in enumerate(closes)
    ]


def test_rank_momentum_grace_keeps_positive_trend_position(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
            ("CCC", 0.70),
            ("DDD", 0.60),
        ],
    )
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            min_hold_bars=1,
            keep_buffer=1,
            exit_policy="rank_momentum_grace",
            grace_rank_buffer=0,
            grace_momentum_window=2,
            grace_min_return=0.02,
        )
    )
    strategy._hold_days = {"CCC": 3}
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC", "DDD"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.2]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
            "CCC": _history("CCC", [10.0, 10.3, 10.6]),
            "DDD": _history("DDD", [10.0, 9.9, 9.8]),
        },
        positions={"CCC": Position("CCC", 100, 10.0, 10.6)},
        cash=0.0,
    )

    selected = strategy.select(context)

    assert "CCC" in selected


def test_allocate_does_not_keep_stale_position_under_turnover_floor(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
            ("CCC", 0.70),
        ],
    )
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            min_hold_bars=1,
            keep_buffer=0,
            min_turnover_names=3,
        )
    )
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.2]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
            "CCC": _history("CCC", [10.0]),
        },
        positions={
            "AAA": Position("AAA", 100, 10.0, 10.2),
            "CCC": Position("CCC", 100, 10.0, 10.0),
        },
        cash=0.0,
    )

    allocation = strategy.allocate(context, ["AAA", "BBB"])

    assert set(allocation.target_weights) == {"AAA", "BBB"}
    assert "CCC" not in allocation.target_weights


def test_select_filters_out_symbols_below_liquidity_threshold(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
            ("CCC", 0.70),
        ],
    )
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            min_daily_amount=500_000.0,
        )
    )
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.2]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
            "CCC": _history("CCC", [10.0, 10.0, 10.0]),
        },
        positions={},
        cash=0.0,
    )
    context.bars["BBB"][-1] = Bar(
        symbol="BBB",
        trade_date=context.bars["BBB"][-1].trade_date,
        open=10.0,
        high=10.0,
        low=10.0,
        close=10.0,
        amount=300_000.0,
    )

    selected = strategy.select(context)

    assert "AAA" in selected
    assert "BBB" not in selected


def test_select_filters_out_symbols_above_price_cap(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
            ("CCC", 0.70),
        ],
    )
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            max_close_price=50.0,
        )
    )
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.2]),
            "BBB": _history("BBB", [60.0, 60.0, 60.0]),
            "CCC": _history("CCC", [10.0, 10.0, 10.0]),
        },
        positions={},
        cash=0.0,
    )

    selected = strategy.select(context)

    assert "AAA" in selected
    assert "BBB" not in selected


def test_select_filters_out_st_symbols_even_when_they_have_top_scores(tmp_path) -> None:
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-08", "symbol": "AAA", "prediction": 0.95},
            {"trade_date": "2025-01-08", "symbol": "BBB", "prediction": 0.90},
            {"trade_date": "2025-01-08", "symbol": "CCC", "prediction": 0.80},
        ]
    ).to_parquet(scores_path, index=False)
    instruments_path = tmp_path / "storage" / "parquet" / "instruments" / "ashare_instruments.parquet"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "industry_level_1": "industry_0", "is_st": True},
            {"symbol": "BBB", "industry_level_1": "industry_1", "is_st": False},
            {"symbol": "CCC", "industry_level_1": "industry_2", "is_st": False},
        ]
    ).to_parquet(instruments_path, index=False)

    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path.as_posix(),
            storage_root=(tmp_path / "storage").as_posix(),
            top_k=2,
        )
    )
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.2]),
            "BBB": _history("BBB", [10.0, 10.1, 10.2]),
            "CCC": _history("CCC", [10.0, 10.1, 10.2]),
        },
        positions={},
        cash=0.0,
    )

    selected = strategy.select(context)

    assert selected == ["BBB", "CCC"]


def test_rank_momentum_grace_exits_when_rank_and_momentum_both_break(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
            ("CCC", 0.70),
            ("DDD", 0.60),
        ],
    )
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            min_hold_bars=1,
            keep_buffer=1,
            exit_policy="rank_momentum_grace",
            grace_rank_buffer=0,
            grace_momentum_window=2,
            grace_min_return=0.02,
        )
    )
    strategy._hold_days = {"CCC": 3}
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC", "DDD"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.2]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
            "CCC": _history("CCC", [10.0, 9.7, 9.4]),
            "DDD": _history("DDD", [10.0, 9.7, 9.4]),
        },
        positions={"CCC": Position("CCC", 100, 10.0, 9.4)},
        cash=0.0,
    )

    selected = strategy.select(context)

    assert "CCC" not in selected


def test_trailing_drawdown_exits_profitable_position_after_pullback(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
            ("CCC", 0.70),
        ],
    )
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            min_hold_bars=1,
            keep_buffer=1,
            exit_policy="trailing_drawdown",
            trailing_stop_window=3,
            trailing_stop_drawdown=0.10,
            trailing_stop_min_gain=0.15,
        )
    )
    strategy._hold_days = {"CCC": 4}
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.2]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
            "CCC": _history("CCC", [10.0, 12.0, 10.7]),
        },
        positions={"CCC": Position("CCC", 100, 9.0, 10.7)},
        cash=0.0,
    )

    selected = strategy.select(context)

    assert "CCC" not in selected


def test_trailing_drawdown_keeps_position_before_stop_threshold(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
            ("CCC", 0.70),
        ],
    )
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            min_hold_bars=1,
            keep_buffer=1,
            exit_policy="trailing_drawdown",
            trailing_stop_window=3,
            trailing_stop_drawdown=0.10,
            trailing_stop_min_gain=0.15,
        )
    )
    strategy._hold_days = {"CCC": 4}
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.2]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
            "CCC": _history("CCC", [10.0, 12.0, 11.3]),
        },
        positions={"CCC": Position("CCC", 100, 9.0, 11.3)},
        cash=0.0,
    )

    selected = strategy.select(context)

    assert "CCC" in selected


def test_score_reversal_exits_after_consecutive_negative_scores(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
            ("CCC", -0.10),
            ("DDD", -0.20),
        ],
    )
    pd.DataFrame(
        [
            {"trade_date": "2025-01-06", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-06", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-06", "symbol": "CCC", "prediction": -0.10},
            {"trade_date": "2025-01-06", "symbol": "DDD", "prediction": -0.20},
            {"trade_date": "2025-01-07", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-07", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-07", "symbol": "CCC", "prediction": -0.05},
            {"trade_date": "2025-01-07", "symbol": "DDD", "prediction": -0.20},
            {"trade_date": "2025-01-08", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-08", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-08", "symbol": "CCC", "prediction": -0.02},
            {"trade_date": "2025-01-08", "symbol": "DDD", "prediction": -0.20},
        ]
    ).to_parquet(scores_path, index=False)
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            min_hold_bars=1,
            keep_buffer=0,
            exit_policy="score_reversal",
            score_reversal_confirm_days=3,
            score_reversal_threshold=0.0,
        )
    )
    strategy._hold_days = {"CCC": 4}
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC", "DDD"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.2]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
            "CCC": _history("CCC", [10.0, 10.0, 10.0]),
            "DDD": _history("DDD", [10.0, 10.0, 10.0]),
        },
        positions={"CCC": Position("CCC", 100, 10.0, 10.0)},
        cash=0.0,
    )

    selected = strategy.select(context)

    assert "CCC" not in selected


def test_score_reversal_keeps_when_negative_streak_not_confirmed(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
            ("CCC", 0.01),
            ("DDD", -0.20),
        ],
    )
    pd.DataFrame(
        [
            {"trade_date": "2025-01-06", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-06", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-06", "symbol": "CCC", "prediction": -0.10},
            {"trade_date": "2025-01-06", "symbol": "DDD", "prediction": -0.20},
            {"trade_date": "2025-01-07", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-07", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-07", "symbol": "CCC", "prediction": -0.05},
            {"trade_date": "2025-01-07", "symbol": "DDD", "prediction": -0.20},
            {"trade_date": "2025-01-08", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-08", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-08", "symbol": "CCC", "prediction": 0.01},
            {"trade_date": "2025-01-08", "symbol": "DDD", "prediction": -0.20},
        ]
    ).to_parquet(scores_path, index=False)
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            min_hold_bars=1,
            keep_buffer=0,
            exit_policy="score_reversal",
            score_reversal_confirm_days=3,
            score_reversal_threshold=0.0,
        )
    )
    strategy._hold_days = {"CCC": 4}
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC", "DDD"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.2]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
            "CCC": _history("CCC", [10.0, 10.0, 10.0]),
            "DDD": _history("DDD", [10.0, 10.0, 10.0]),
        },
        positions={"CCC": Position("CCC", 100, 10.0, 10.0)},
        cash=0.0,
    )

    selected = strategy.select(context)

    assert "CCC" in selected


def test_score_price_hybrid_exits_only_when_score_and_price_both_weak(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
            ("CCC", -0.02),
            ("DDD", -0.20),
        ],
    )
    pd.DataFrame(
        [
            {"trade_date": "2025-01-06", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-06", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-06", "symbol": "CCC", "prediction": -0.10},
            {"trade_date": "2025-01-06", "symbol": "DDD", "prediction": -0.20},
            {"trade_date": "2025-01-07", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-07", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-07", "symbol": "CCC", "prediction": -0.05},
            {"trade_date": "2025-01-07", "symbol": "DDD", "prediction": -0.20},
            {"trade_date": "2025-01-08", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-08", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-08", "symbol": "CCC", "prediction": -0.02},
            {"trade_date": "2025-01-08", "symbol": "DDD", "prediction": -0.20},
        ]
    ).to_parquet(scores_path, index=False)
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            min_hold_bars=1,
            keep_buffer=0,
            exit_policy="score_price_hybrid",
            score_reversal_confirm_days=3,
            score_reversal_threshold=0.0,
            hybrid_price_window=2,
            hybrid_price_threshold=0.0,
        )
    )
    strategy._hold_days = {"CCC": 4}
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC", "DDD"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.2]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
            "CCC": _history("CCC", [10.0, 9.9, 9.8]),
            "DDD": _history("DDD", [10.0, 10.0, 10.0]),
        },
        positions={"CCC": Position("CCC", 100, 10.0, 9.8)},
        cash=0.0,
    )

    selected = strategy.select(context)

    assert "CCC" not in selected


def test_score_price_hybrid_keeps_when_price_still_strong(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
            ("CCC", -0.02),
            ("DDD", -0.20),
        ],
    )
    pd.DataFrame(
        [
            {"trade_date": "2025-01-06", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-06", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-06", "symbol": "CCC", "prediction": -0.10},
            {"trade_date": "2025-01-06", "symbol": "DDD", "prediction": -0.20},
            {"trade_date": "2025-01-07", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-07", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-07", "symbol": "CCC", "prediction": -0.05},
            {"trade_date": "2025-01-07", "symbol": "DDD", "prediction": -0.20},
            {"trade_date": "2025-01-08", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-08", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-08", "symbol": "CCC", "prediction": -0.02},
            {"trade_date": "2025-01-08", "symbol": "DDD", "prediction": -0.20},
        ]
    ).to_parquet(scores_path, index=False)
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            min_hold_bars=1,
            keep_buffer=0,
            exit_policy="score_price_hybrid",
            score_reversal_confirm_days=3,
            score_reversal_threshold=0.0,
            hybrid_price_window=2,
            hybrid_price_threshold=0.0,
        )
    )
    strategy._hold_days = {"CCC": 4}
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC", "DDD"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.2]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
            "CCC": _history("CCC", [10.0, 10.3, 10.6]),
            "DDD": _history("DDD", [10.0, 10.0, 10.0]),
        },
        positions={"CCC": Position("CCC", 100, 10.0, 10.6)},
        cash=0.0,
    )

    selected = strategy.select(context)

    assert "CCC" in selected


def test_strong_keep_extension_retains_strong_position_beyond_buffer(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
            ("CCC", 0.70),
            ("DDD", 0.60),
            ("EEE", 0.50),
        ],
    )
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            min_hold_bars=1,
            keep_buffer=1,
            strong_keep_extra_buffer=1,
            strong_keep_momentum_window=2,
            strong_keep_min_return=0.05,
        )
    )
    strategy._hold_days = {"DDD": 3}
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC", "DDD", "EEE"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.2]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
            "CCC": _history("CCC", [10.0, 10.0, 10.0]),
            "DDD": _history("DDD", [10.0, 10.7, 11.4]),
            "EEE": _history("EEE", [10.0, 10.0, 10.0]),
        },
        positions={"DDD": Position("DDD", 100, 10.0, 11.4)},
        cash=0.0,
    )

    selected = strategy.select(context)

    assert "DDD" in selected


def test_strong_keep_extension_does_not_retain_weak_position_beyond_buffer(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
            ("CCC", 0.70),
            ("DDD", 0.60),
            ("EEE", 0.50),
        ],
    )
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            min_hold_bars=1,
            keep_buffer=1,
            strong_keep_extra_buffer=1,
            strong_keep_momentum_window=2,
            strong_keep_min_return=0.05,
        )
    )
    strategy._hold_days = {"DDD": 3}
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC", "DDD", "EEE"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.2]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
            "CCC": _history("CCC", [10.0, 10.0, 10.0]),
            "DDD": _history("DDD", [10.0, 10.1, 10.0]),
            "EEE": _history("EEE", [10.0, 10.0, 10.0]),
        },
        positions={"DDD": Position("DDD", 100, 10.0, 10.0)},
        cash=0.0,
    )

    selected = strategy.select(context)

    assert "DDD" not in selected


def test_strong_trim_slowdown_preserves_more_weight_for_strong_existing_position(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
        ],
    )
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            min_turnover_names=0,
            strong_trim_slowdown=0.5,
            strong_trim_momentum_window=2,
            strong_trim_min_return=0.05,
        )
    )
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB"),
        bars={
            "AAA": _history("AAA", [10.0, 10.8, 11.6]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
        },
        positions={
            "AAA": Position("AAA", 700, 10.0, 11.6),
            "BBB": Position("BBB", 300, 10.0, 10.0),
        },
        cash=0.0,
    )

    allocation = strategy.allocate(context, ["AAA", "BBB"])

    assert allocation.target_weights["AAA"] > 0.5
    assert allocation.target_weights["BBB"] < 0.5


def test_strong_trim_slowdown_keeps_equal_weight_when_signal_not_strong(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
        ],
    )
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=2,
            min_turnover_names=0,
            strong_trim_slowdown=0.5,
            strong_trim_momentum_window=2,
            strong_trim_min_return=0.20,
        )
    )
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB"),
        bars={
            "AAA": _history("AAA", [10.0, 10.1, 10.15]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
        },
        positions={
            "AAA": Position("AAA", 700, 10.0, 10.15),
            "BBB": Position("BBB", 300, 10.0, 10.0),
        },
        cash=0.0,
    )

    allocation = strategy.allocate(context, ["AAA", "BBB"])

    assert allocation.target_weights["AAA"] == 0.5
    assert allocation.target_weights["BBB"] == 0.5


def test_max_position_weight_caps_target_weights(tmp_path) -> None:
    scores_path, storage_root = _write_inputs(
        tmp_path,
        [
            ("AAA", 0.90),
            ("BBB", 0.80),
            ("CCC", 0.70),
        ],
    )
    strategy = ScoreTopKStrategy(
        ScoreStrategyConfig(
            scores_path=scores_path,
            storage_root=storage_root,
            top_k=3,
            min_hold_bars=1,
            keep_buffer=0,
            strong_trim_slowdown=1.0,
            strong_trim_momentum_window=2,
            strong_trim_min_return=0.01,
            max_position_weight=0.4,
        )
    )
    context = StrategyContext(
        trade_date=date(2025, 1, 13),
        universe=("AAA", "BBB", "CCC"),
        bars={
            "AAA": _history("AAA", [10.0, 11.0, 12.0]),
            "BBB": _history("BBB", [10.0, 10.0, 10.0]),
            "CCC": _history("CCC", [10.0, 10.0, 10.0]),
        },
        positions={"AAA": Position("AAA", 800, 10.0, 12.0)},
        cash=2000.0,
    )

    allocation = strategy.allocate(context, ["AAA", "BBB", "CCC"])

    assert allocation.target_weights["AAA"] <= 0.4
    assert round(sum(allocation.target_weights.values()), 6) == 1.0
