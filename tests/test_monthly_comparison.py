from __future__ import annotations

import json
import pytest

from ashare_backtest.research.analysis import MonthlyComparisonConfig, compare_backtest_monthly_returns


def test_compare_backtest_monthly_returns_outputs_monthly_diff(tmp_path) -> None:
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    control_dir.joinpath("equity_curve.csv").write_text(
        "\n".join(
            [
                "trade_date,equity",
                "2025-01-02,100",
                "2025-01-31,110",
                "2025-02-03,110",
                "2025-02-28,121",
            ]
        ),
        encoding="utf-8",
    )

    capped_dir = tmp_path / "capped"
    capped_dir.mkdir()
    capped_dir.joinpath("equity_curve.csv").write_text(
        "\n".join(
            [
                "trade_date,equity",
                "2025-01-02,100",
                "2025-01-31,105",
                "2025-02-03,105",
                "2025-02-28,108",
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "monthly.json"
    payload = compare_backtest_monthly_returns(
        MonthlyComparisonConfig(
            result_dirs=(control_dir.as_posix(), capped_dir.as_posix()),
            labels=("control", "capped"),
            output_path=output_path.as_posix(),
        )
    )

    assert payload["summary"]["baseline_label"] == "control"
    assert payload["summary"]["by_label"][0]["best_month"] == "2025-01"
    assert payload["summary"]["by_label"][1]["worst_month"] == "2025-02"
    assert payload["by_month"][0]["control"] == pytest.approx(0.1)
    assert payload["by_month"][0]["capped"] == pytest.approx(0.05)
    assert payload["by_month"][0]["capped_minus_control"] == pytest.approx(-0.05)

    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["by_month"][1]["capped_minus_control"] < 0
