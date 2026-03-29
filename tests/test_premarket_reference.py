from __future__ import annotations

import json
from datetime import date

import pandas as pd

from ashare_backtest.data import InMemoryDataProvider
from ashare_backtest.protocol import Bar
from ashare_backtest.research.analysis import (
    PremarketReferenceConfig,
    StrategyStateConfig,
    generate_premarket_reference,
    generate_strategy_state,
)


def _bar(symbol: str, trade_date: date, close: float, amount: float = 1_000_000.0) -> Bar:
    return Bar(
        symbol=symbol,
        trade_date=trade_date,
        open=close,
        high=close,
        low=close,
        close=close,
        amount=amount,
    )


def test_generate_premarket_reference_outputs_hold_buy_sell_actions(tmp_path) -> None:
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-08", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-08", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-08", "symbol": "CCC", "prediction": 0.70},
            {"trade_date": "2025-01-09", "symbol": "AAA", "prediction": 0.95},
            {"trade_date": "2025-01-09", "symbol": "CCC", "prediction": 0.85},
            {"trade_date": "2025-01-09", "symbol": "BBB", "prediction": 0.10},
        ]
    ).to_parquet(scores_path, index=False)

    instruments_path = tmp_path / "storage" / "parquet" / "instruments" / "ashare_instruments.parquet"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "industry_level_1": "tech"},
            {"symbol": "BBB", "industry_level_1": "finance"},
            {"symbol": "CCC", "industry_level_1": "tech"},
        ]
    ).to_parquet(instruments_path, index=False)

    provider = InMemoryDataProvider(
        {
            "AAA": [
                _bar("AAA", date(2025, 1, 7), 10.0),
                _bar("AAA", date(2025, 1, 8), 10.1),
                _bar("AAA", date(2025, 1, 9), 10.2),
                _bar("AAA", date(2025, 1, 10), 10.3),
            ],
            "BBB": [
                _bar("BBB", date(2025, 1, 7), 20.0),
                _bar("BBB", date(2025, 1, 8), 20.1),
                _bar("BBB", date(2025, 1, 9), 20.2),
                _bar("BBB", date(2025, 1, 10), 20.3),
            ],
            "CCC": [
                _bar("CCC", date(2025, 1, 7), 30.0),
                _bar("CCC", date(2025, 1, 8), 30.1),
                _bar("CCC", date(2025, 1, 9), 30.2),
                _bar("CCC", date(2025, 1, 10), 30.3),
            ],
        }
    )

    output_path = tmp_path / "premarket.json"
    payload = generate_premarket_reference(
        PremarketReferenceConfig(
            scores_path=scores_path.as_posix(),
            storage_root=(tmp_path / "storage").as_posix(),
            output_path=output_path.as_posix(),
            trade_date="2025-01-10",
            top_k=2,
            rebalance_every=1,
            lookback_window=2,
            min_hold_bars=1,
            keep_buffer=0,
        ),
        provider=provider,
    )

    assert payload["summary"]["signal_date"] == "2025-01-09"
    assert payload["summary"]["execution_date"] == "2025-01-10"
    assert payload["selected_symbols"] == ["AAA", "CCC"]

    action_by_symbol = {item["symbol"]: item["action"] for item in payload["actions"]}
    assert action_by_symbol["AAA"] == "TRIM"
    assert action_by_symbol["BBB"] == "SELL"
    assert action_by_symbol["CCC"] == "BUY"

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["target_weights"]["AAA"] == 0.5
    assert persisted["target_weights"]["CCC"] == 0.5


def test_generate_premarket_reference_returns_hold_actions_on_non_rebalance_day(tmp_path) -> None:
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-07", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-07", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-08", "symbol": "AAA", "prediction": 0.92},
            {"trade_date": "2025-01-08", "symbol": "BBB", "prediction": 0.82},
            {"trade_date": "2025-01-09", "symbol": "AAA", "prediction": 0.95},
            {"trade_date": "2025-01-09", "symbol": "BBB", "prediction": 0.10},
        ]
    ).to_parquet(scores_path, index=False)

    instruments_path = tmp_path / "storage" / "parquet" / "instruments" / "ashare_instruments.parquet"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "industry_level_1": "tech"},
            {"symbol": "BBB", "industry_level_1": "finance"},
        ]
    ).to_parquet(instruments_path, index=False)

    provider = InMemoryDataProvider(
        {
            "AAA": [
                _bar("AAA", date(2025, 1, 6), 10.0),
                _bar("AAA", date(2025, 1, 7), 10.1),
                _bar("AAA", date(2025, 1, 8), 10.2),
                _bar("AAA", date(2025, 1, 9), 10.3),
                _bar("AAA", date(2025, 1, 10), 10.4),
            ],
            "BBB": [
                _bar("BBB", date(2025, 1, 6), 20.0),
                _bar("BBB", date(2025, 1, 7), 20.1),
                _bar("BBB", date(2025, 1, 8), 20.2),
                _bar("BBB", date(2025, 1, 9), 20.3),
                _bar("BBB", date(2025, 1, 10), 20.4),
            ],
        }
    )

    payload = generate_premarket_reference(
        PremarketReferenceConfig(
            scores_path=scores_path.as_posix(),
            storage_root=(tmp_path / "storage").as_posix(),
            output_path=(tmp_path / "premarket_hold.json").as_posix(),
            trade_date="2025-01-10",
            top_k=1,
            rebalance_every=5,
            lookback_window=2,
            min_hold_bars=1,
            keep_buffer=0,
        ),
        provider=provider,
    )

    assert payload["summary"]["decision_reason"] == "rebalance_schedule"
    assert payload["summary"]["current_position_count"] == 1
    assert payload["summary"]["target_position_count"] == 1
    assert payload["selected_symbols"] == ["AAA"]
    assert {item["symbol"]: item["action"] for item in payload["actions"]} == {"AAA": "HOLD"}


def test_generate_strategy_state_initial_entry_builds_first_portfolio(tmp_path) -> None:
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-08", "symbol": "AAA", "prediction": 0.60},
            {"trade_date": "2025-01-08", "symbol": "BBB", "prediction": 0.40},
            {"trade_date": "2025-01-09", "symbol": "AAA", "prediction": 0.95},
            {"trade_date": "2025-01-09", "symbol": "CCC", "prediction": 0.90},
            {"trade_date": "2025-01-09", "symbol": "BBB", "prediction": 0.10},
        ]
    ).to_parquet(scores_path, index=False)

    instruments_path = tmp_path / "storage" / "parquet" / "instruments" / "ashare_instruments.parquet"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "industry_level_1": "tech"},
            {"symbol": "BBB", "industry_level_1": "finance"},
            {"symbol": "CCC", "industry_level_1": "health"},
        ]
    ).to_parquet(instruments_path, index=False)

    provider = InMemoryDataProvider(
        {
            "AAA": [
                _bar("AAA", date(2025, 1, 7), 10.0),
                _bar("AAA", date(2025, 1, 8), 10.1),
                _bar("AAA", date(2025, 1, 9), 10.2),
                _bar("AAA", date(2025, 1, 10), 10.3),
            ],
            "BBB": [
                _bar("BBB", date(2025, 1, 7), 20.0),
                _bar("BBB", date(2025, 1, 8), 20.1),
                _bar("BBB", date(2025, 1, 9), 20.2),
                _bar("BBB", date(2025, 1, 10), 20.3),
            ],
            "CCC": [
                _bar("CCC", date(2025, 1, 7), 30.0),
                _bar("CCC", date(2025, 1, 8), 30.1),
                _bar("CCC", date(2025, 1, 9), 30.2),
                _bar("CCC", date(2025, 1, 10), 30.3),
            ],
        }
    )

    state_path = tmp_path / "strategy_state.json"
    payload = generate_strategy_state(
        StrategyStateConfig(
            scores_path=scores_path.as_posix(),
            storage_root=(tmp_path / "storage").as_posix(),
            output_path=state_path.as_posix(),
            trade_date="2025-01-10",
            mode="initial_entry",
            top_k=2,
            rebalance_every=5,
            lookback_window=2,
            min_hold_bars=1,
            keep_buffer=0,
        ),
        provider=provider,
    )

    assert payload["summary"]["state_mode"] == "initial_entry"
    assert payload["summary"]["decision_reason"] == "initial_entry"
    assert payload["plan"]["selected_symbols"] == ["AAA", "CCC"]
    assert {item["symbol"]: item["action"] for item in payload["plan"]["actions"]} == {
        "AAA": "BUY",
        "CCC": "BUY",
    }
    assert len(payload["next_state"]["positions"]) == 2
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["next_state"]["as_of_trade_date"] == "2025-01-10"
    trades_csv = state_path.with_name("trades.csv")
    assert trades_csv.exists()
    trades_text = trades_csv.read_text(encoding="utf-8")
    assert "AAA" in trades_text
    assert "BUY" in trades_text
    decision_log_csv = state_path.with_name("decision_log.csv")
    assert decision_log_csv.exists()
    decision_log_text = decision_log_csv.read_text(encoding="utf-8")
    assert "initial_entry" in decision_log_text


def test_generate_strategy_state_initial_entry_excludes_st_symbols(tmp_path) -> None:
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-08", "symbol": "AAA", "prediction": 0.60},
            {"trade_date": "2025-01-08", "symbol": "BBB", "prediction": 0.40},
            {"trade_date": "2025-01-09", "symbol": "AAA", "prediction": 0.99},
            {"trade_date": "2025-01-09", "symbol": "CCC", "prediction": 0.90},
            {"trade_date": "2025-01-09", "symbol": "BBB", "prediction": 0.80},
        ]
    ).to_parquet(scores_path, index=False)

    instruments_path = tmp_path / "storage" / "parquet" / "instruments" / "ashare_instruments.parquet"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "industry_level_1": "tech", "is_st": True},
            {"symbol": "BBB", "industry_level_1": "finance", "is_st": False},
            {"symbol": "CCC", "industry_level_1": "health", "is_st": False},
        ]
    ).to_parquet(instruments_path, index=False)

    provider = InMemoryDataProvider(
        {
            "AAA": [
                _bar("AAA", date(2025, 1, 7), 10.0),
                _bar("AAA", date(2025, 1, 8), 10.1),
                _bar("AAA", date(2025, 1, 9), 10.2),
                _bar("AAA", date(2025, 1, 10), 10.3),
            ],
            "BBB": [
                _bar("BBB", date(2025, 1, 7), 20.0),
                _bar("BBB", date(2025, 1, 8), 20.1),
                _bar("BBB", date(2025, 1, 9), 20.2),
                _bar("BBB", date(2025, 1, 10), 20.3),
            ],
            "CCC": [
                _bar("CCC", date(2025, 1, 7), 30.0),
                _bar("CCC", date(2025, 1, 8), 30.1),
                _bar("CCC", date(2025, 1, 9), 30.2),
                _bar("CCC", date(2025, 1, 10), 30.3),
            ],
        }
    )

    payload = generate_strategy_state(
        StrategyStateConfig(
            scores_path=scores_path.as_posix(),
            storage_root=(tmp_path / "storage").as_posix(),
            output_path=(tmp_path / "strategy_state.json").as_posix(),
            trade_date="2025-01-10",
            mode="initial_entry",
            top_k=2,
            rebalance_every=5,
            lookback_window=2,
            min_hold_bars=1,
            keep_buffer=0,
        ),
        provider=provider,
    )

    assert payload["plan"]["selected_symbols"] == ["CCC", "BBB"]
    assert all(symbol != "AAA" for symbol in payload["plan"]["selected_symbols"])


def test_generate_strategy_state_initial_entry_uses_universe_membership_filter(tmp_path) -> None:
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-08", "symbol": "AAA", "prediction": 0.60},
            {"trade_date": "2025-01-08", "symbol": "BBB", "prediction": 0.40},
            {"trade_date": "2025-01-09", "symbol": "AAA", "prediction": 0.99},
            {"trade_date": "2025-01-09", "symbol": "CCC", "prediction": 0.90},
            {"trade_date": "2025-01-09", "symbol": "BBB", "prediction": 0.80},
        ]
    ).to_parquet(scores_path, index=False)

    instruments_path = tmp_path / "storage" / "parquet" / "instruments" / "ashare_instruments.parquet"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "industry_level_1": "tech", "is_st": False},
            {"symbol": "BBB", "industry_level_1": "finance", "is_st": False},
            {"symbol": "CCC", "industry_level_1": "health", "is_st": False},
        ]
    ).to_parquet(instruments_path, index=False)

    memberships_path = tmp_path / "storage" / "parquet" / "universe" / "memberships.parquet"
    memberships_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"universe_name": "tradable_core", "symbol": "BBB", "effective_date": "2025-01-09", "expiry_date": None},
            {"universe_name": "tradable_core", "symbol": "CCC", "effective_date": "2025-01-09", "expiry_date": None},
        ]
    ).to_parquet(memberships_path, index=False)

    provider = InMemoryDataProvider(
        {
            "AAA": [
                _bar("AAA", date(2025, 1, 7), 10.0),
                _bar("AAA", date(2025, 1, 8), 10.1),
                _bar("AAA", date(2025, 1, 9), 10.2),
                _bar("AAA", date(2025, 1, 10), 10.3),
            ],
            "BBB": [
                _bar("BBB", date(2025, 1, 7), 20.0),
                _bar("BBB", date(2025, 1, 8), 20.1),
                _bar("BBB", date(2025, 1, 9), 20.2),
                _bar("BBB", date(2025, 1, 10), 20.3),
            ],
            "CCC": [
                _bar("CCC", date(2025, 1, 7), 30.0),
                _bar("CCC", date(2025, 1, 8), 30.1),
                _bar("CCC", date(2025, 1, 9), 30.2),
                _bar("CCC", date(2025, 1, 10), 30.3),
            ],
        }
    )

    payload = generate_strategy_state(
        StrategyStateConfig(
            scores_path=scores_path.as_posix(),
            storage_root=(tmp_path / "storage").as_posix(),
            output_path=(tmp_path / "strategy_state.json").as_posix(),
            trade_date="2025-01-10",
            universe_name="tradable_core",
            mode="initial_entry",
            top_k=2,
            rebalance_every=5,
            lookback_window=2,
            min_hold_bars=1,
            keep_buffer=0,
        ),
        provider=provider,
    )

    assert payload["plan"]["selected_symbols"] == ["CCC", "BBB"]
    assert "AAA" not in payload["plan"]["selected_symbols"]


def test_generate_strategy_state_marks_action_below_round_lot_when_no_trade_is_possible(tmp_path) -> None:
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-09", "symbol": "AAA", "prediction": 0.95},
        ]
    ).to_parquet(scores_path, index=False)

    instruments_path = tmp_path / "storage" / "parquet" / "instruments" / "ashare_instruments.parquet"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "industry_level_1": "tech", "is_st": False},
        ]
    ).to_parquet(instruments_path, index=False)

    provider = InMemoryDataProvider(
        {
            "AAA": [
                _bar("AAA", date(2025, 1, 7), 50.0),
                _bar("AAA", date(2025, 1, 8), 50.5),
                _bar("AAA", date(2025, 1, 9), 51.0),
                _bar("AAA", date(2025, 1, 10), 51.0),
            ],
        }
    )

    payload = generate_strategy_state(
        StrategyStateConfig(
            scores_path=scores_path.as_posix(),
            storage_root=(tmp_path / "storage").as_posix(),
            output_path=(tmp_path / "strategy_state.json").as_posix(),
            trade_date="2025-01-10",
            mode="initial_entry",
            top_k=1,
            rebalance_every=1,
            lookback_window=2,
            min_hold_bars=1,
            keep_buffer=0,
            initial_cash=5_000.0,
        ),
        provider=provider,
    )

    action = payload["plan"]["actions"][0]
    assert action["action"] == "BUY"
    assert action["planned_quantity"] == 0
    assert action["executed_quantity"] == 0
    assert action["would_trade"] is False
    assert action["execution_status"] == "below_round_lot"
    assert action["execution_reason"] == "below_100_share_lot"


def test_generate_strategy_state_continues_from_saved_state(tmp_path) -> None:
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-08", "symbol": "AAA", "prediction": 0.90},
            {"trade_date": "2025-01-08", "symbol": "BBB", "prediction": 0.80},
            {"trade_date": "2025-01-08", "symbol": "CCC", "prediction": 0.70},
            {"trade_date": "2025-01-09", "symbol": "AAA", "prediction": 0.95},
            {"trade_date": "2025-01-09", "symbol": "CCC", "prediction": 0.85},
            {"trade_date": "2025-01-09", "symbol": "BBB", "prediction": 0.10},
            {"trade_date": "2025-01-10", "symbol": "BBB", "prediction": 0.96},
            {"trade_date": "2025-01-10", "symbol": "CCC", "prediction": 0.90},
            {"trade_date": "2025-01-10", "symbol": "AAA", "prediction": 0.20},
        ]
    ).to_parquet(scores_path, index=False)

    instruments_path = tmp_path / "storage" / "parquet" / "instruments" / "ashare_instruments.parquet"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "industry_level_1": "tech"},
            {"symbol": "BBB", "industry_level_1": "finance"},
            {"symbol": "CCC", "industry_level_1": "health"},
        ]
    ).to_parquet(instruments_path, index=False)

    provider = InMemoryDataProvider(
        {
            "AAA": [
                _bar("AAA", date(2025, 1, 7), 10.0),
                _bar("AAA", date(2025, 1, 8), 10.1),
                _bar("AAA", date(2025, 1, 9), 10.2),
                _bar("AAA", date(2025, 1, 10), 10.3),
                _bar("AAA", date(2025, 1, 13), 10.4),
            ],
            "BBB": [
                _bar("BBB", date(2025, 1, 7), 20.0),
                _bar("BBB", date(2025, 1, 8), 20.1),
                _bar("BBB", date(2025, 1, 9), 20.2),
                _bar("BBB", date(2025, 1, 10), 20.3),
                _bar("BBB", date(2025, 1, 13), 20.4),
            ],
            "CCC": [
                _bar("CCC", date(2025, 1, 7), 30.0),
                _bar("CCC", date(2025, 1, 8), 30.1),
                _bar("CCC", date(2025, 1, 9), 30.2),
                _bar("CCC", date(2025, 1, 10), 30.3),
                _bar("CCC", date(2025, 1, 13), 30.4),
            ],
        }
    )

    state_path = tmp_path / "strategy_state.json"
    first_payload = generate_strategy_state(
        StrategyStateConfig(
            scores_path=scores_path.as_posix(),
            storage_root=(tmp_path / "storage").as_posix(),
            output_path=state_path.as_posix(),
            trade_date="2025-01-10",
            mode="initial_entry",
            top_k=2,
            rebalance_every=0,
            lookback_window=2,
            min_hold_bars=1,
            keep_buffer=0,
        ),
        provider=provider,
    )
    assert len(first_payload["next_state"]["positions"]) == 2

    continued = generate_strategy_state(
        StrategyStateConfig(
            scores_path=scores_path.as_posix(),
            storage_root=(tmp_path / "storage").as_posix(),
            output_path=state_path.as_posix(),
            trade_date="2025-01-13",
            mode="continue",
            previous_state_path=state_path.as_posix(),
            top_k=2,
            rebalance_every=0,
            lookback_window=2,
            min_hold_bars=1,
            keep_buffer=0,
        ),
        provider=provider,
    )

    assert continued["summary"]["state_mode"] == "continue"
    assert continued["summary"]["signal_date"] == "2025-01-10"
    assert set(continued["plan"]["selected_symbols"]) == {"BBB", "CCC"}
    action_by_symbol = {item["symbol"]: item["action"] for item in continued["plan"]["actions"]}
    assert action_by_symbol["AAA"] == "SELL"
    assert action_by_symbol["BBB"] == "BUY"


def test_generate_strategy_state_continue_accepts_single_day_scores_after_older_state(tmp_path) -> None:
    full_scores_path = tmp_path / "scores_full.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-07", "symbol": "AAA", "prediction": 0.95},
            {"trade_date": "2025-01-07", "symbol": "BBB", "prediction": 0.90},
            {"trade_date": "2025-01-07", "symbol": "CCC", "prediction": 0.10},
            {"trade_date": "2025-01-08", "symbol": "AAA", "prediction": 0.96},
            {"trade_date": "2025-01-08", "symbol": "BBB", "prediction": 0.85},
            {"trade_date": "2025-01-08", "symbol": "CCC", "prediction": 0.20},
            {"trade_date": "2025-01-10", "symbol": "BBB", "prediction": 0.97},
            {"trade_date": "2025-01-10", "symbol": "CCC", "prediction": 0.92},
            {"trade_date": "2025-01-10", "symbol": "AAA", "prediction": 0.15},
        ]
    ).to_parquet(full_scores_path, index=False)

    single_day_scores_path = tmp_path / "scores_2025-01-10.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-10", "symbol": "BBB", "prediction": 0.97},
            {"trade_date": "2025-01-10", "symbol": "CCC", "prediction": 0.92},
            {"trade_date": "2025-01-10", "symbol": "AAA", "prediction": 0.15},
        ]
    ).to_parquet(single_day_scores_path, index=False)

    instruments_path = tmp_path / "storage" / "parquet" / "instruments" / "ashare_instruments.parquet"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "industry_level_1": "tech"},
            {"symbol": "BBB", "industry_level_1": "finance"},
            {"symbol": "CCC", "industry_level_1": "health"},
        ]
    ).to_parquet(instruments_path, index=False)

    provider = InMemoryDataProvider(
        {
            "AAA": [
                _bar("AAA", date(2025, 1, 6), 10.0),
                _bar("AAA", date(2025, 1, 7), 10.1),
                _bar("AAA", date(2025, 1, 8), 10.2),
                _bar("AAA", date(2025, 1, 9), 10.3),
                _bar("AAA", date(2025, 1, 10), 10.4),
                _bar("AAA", date(2025, 1, 13), 10.5),
            ],
            "BBB": [
                _bar("BBB", date(2025, 1, 6), 20.0),
                _bar("BBB", date(2025, 1, 7), 20.1),
                _bar("BBB", date(2025, 1, 8), 20.2),
                _bar("BBB", date(2025, 1, 9), 20.3),
                _bar("BBB", date(2025, 1, 10), 20.4),
                _bar("BBB", date(2025, 1, 13), 20.5),
            ],
            "CCC": [
                _bar("CCC", date(2025, 1, 6), 30.0),
                _bar("CCC", date(2025, 1, 7), 30.1),
                _bar("CCC", date(2025, 1, 8), 30.2),
                _bar("CCC", date(2025, 1, 9), 30.3),
                _bar("CCC", date(2025, 1, 10), 30.4),
                _bar("CCC", date(2025, 1, 13), 30.5),
            ],
        }
    )

    state_path = tmp_path / "strategy_state.json"
    generate_strategy_state(
        StrategyStateConfig(
            scores_path=full_scores_path.as_posix(),
            storage_root=(tmp_path / "storage").as_posix(),
            output_path=state_path.as_posix(),
            trade_date="2025-01-09",
            mode="initial_entry",
            top_k=2,
            rebalance_every=0,
            lookback_window=2,
            min_hold_bars=1,
            keep_buffer=0,
        ),
        provider=provider,
    )

    continued = generate_strategy_state(
        StrategyStateConfig(
            scores_path=single_day_scores_path.as_posix(),
            storage_root=(tmp_path / "storage").as_posix(),
            output_path=state_path.as_posix(),
            trade_date="2025-01-13",
            mode="continue",
            previous_state_path=state_path.as_posix(),
            top_k=2,
            rebalance_every=0,
            lookback_window=2,
            min_hold_bars=1,
            keep_buffer=0,
        ),
        provider=provider,
    )

    assert continued["summary"]["state_mode"] == "continue"
    assert continued["summary"]["signal_date"] == "2025-01-10"
    assert set(continued["plan"]["selected_symbols"]) == {"BBB", "CCC"}
    action_by_symbol = {item["symbol"]: item["action"] for item in continued["plan"]["actions"]}
    assert action_by_symbol["AAA"] == "SELL"
    assert action_by_symbol["CCC"] == "BUY"


def test_generate_strategy_state_executes_saved_plan_only_once_on_target_date(tmp_path) -> None:
    scores_path = tmp_path / "scores.parquet"
    pd.DataFrame(
        [
            {"trade_date": "2025-01-09", "symbol": "AAA", "prediction": 0.95},
            {"trade_date": "2025-01-09", "symbol": "BBB", "prediction": 0.90},
        ]
    ).to_parquet(scores_path, index=False)

    instruments_path = tmp_path / "storage" / "parquet" / "instruments" / "ashare_instruments.parquet"
    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"symbol": "AAA", "industry_level_1": "tech"},
            {"symbol": "BBB", "industry_level_1": "finance"},
        ]
    ).to_parquet(instruments_path, index=False)

    provider = InMemoryDataProvider(
        {
            "AAA": [
                _bar("AAA", date(2025, 1, 8), 10.0),
                _bar("AAA", date(2025, 1, 9), 10.0),
                _bar("AAA", date(2025, 1, 10), 10.0),
            ],
            "BBB": [
                _bar("BBB", date(2025, 1, 8), 20.0),
                _bar("BBB", date(2025, 1, 9), 20.0),
                _bar("BBB", date(2025, 1, 10), 20.0),
            ],
        }
    )

    plan_state_path = tmp_path / "plan_strategy_state.json"
    plan_payload = generate_strategy_state(
        StrategyStateConfig(
            scores_path=scores_path.as_posix(),
            storage_root=(tmp_path / "storage").as_posix(),
            output_path=plan_state_path.as_posix(),
            trade_date="2025-01-10",
            mode="initial_entry",
            top_k=2,
            rebalance_every=0,
            lookback_window=2,
            min_hold_bars=1,
            keep_buffer=0,
            initial_cash=100000.0,
            simulate_trade_execution=False,
        ),
        provider=provider,
    )

    assert plan_payload["next_state"]["execution_pending"] is True

    execute_state_path = tmp_path / "execute_strategy_state.json"
    executed_payload = generate_strategy_state(
        StrategyStateConfig(
            scores_path=scores_path.as_posix(),
            storage_root=(tmp_path / "storage").as_posix(),
            output_path=execute_state_path.as_posix(),
            trade_date="2025-01-10",
            mode="continue",
            previous_state_path=plan_state_path.as_posix(),
            top_k=2,
            rebalance_every=0,
            lookback_window=2,
            min_hold_bars=1,
            keep_buffer=0,
            initial_cash=100000.0,
            simulate_trade_execution=True,
        ),
        provider=provider,
    )

    trades = pd.read_csv(execute_state_path.with_name("trades.csv"))
    filled = trades.loc[trades["status"] == "filled"].copy()

    assert len(filled) == 2
    assert filled["symbol"].tolist() == ["AAA", "BBB"]
    assert filled["quantity"].tolist() == [5000, 2400]
    assert executed_payload["next_state"]["execution_pending"] is False
