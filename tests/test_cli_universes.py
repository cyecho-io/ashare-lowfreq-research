from __future__ import annotations

import pandas as pd

from ashare_backtest.cli.main import list_universes


def test_list_universes_prints_summary(tmp_path, capsys) -> None:
    storage_root = tmp_path / "storage"
    memberships_path = storage_root / "parquet" / "universe" / "memberships.parquet"
    memberships_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "universe_name": "all_active",
                "symbol": "AAA",
                "effective_date": "2026-03-24",
                "expiry_date": None,
                "source": "test",
            },
            {
                "universe_name": "all_active",
                "symbol": "BBB",
                "effective_date": "2026-03-24",
                "expiry_date": None,
                "source": "test",
            },
            {
                "universe_name": "tradable_core",
                "symbol": "AAA",
                "effective_date": "2026-03-24",
                "expiry_date": None,
                "source": "test",
            },
        ]
    ).to_parquet(memberships_path, index=False)

    list_universes(storage_root.as_posix())

    output = capsys.readouterr().out.strip().splitlines()
    assert output == [
        "UNIVERSE name=all_active symbols=2 effective_from=2026-03-24 effective_to=-",
        "UNIVERSE name=tradable_core symbols=1 effective_from=2026-03-24 effective_to=-",
    ]
