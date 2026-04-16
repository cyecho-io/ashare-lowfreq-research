from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ashare_backtest.cli.research_config import load_research_config, resolve_research_config_path
from ashare_backtest.cli.commands import research as research_commands
from ashare_backtest.web.app import _attach_score_config_paths


def _write_calendar(storage_root: Path) -> None:
    calendar_path = storage_root / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trade_date": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-03-28",
                    "2025-03-31",
                ]
            ),
            "is_open": [False, True, True, True, False],
        }
    ).to_parquet(calendar_path, index=False)


def _write_config(tmp_path: Path, content: str) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(content, encoding="utf-8")
    return config_path


def _minimal_qlib_config(storage_root: Path, extra: str = "") -> str:
    return f"""
[storage]
root = "{storage_root.as_posix()}"

[factor_spec]
id = "qlib_minimal"
universe_name = "tradable_core"

[training]
label_column = "raw_fwd_return_5"
train_window_months = 12
validation_window_months = 1
test_start_month = "2025-01"
test_end_month = "2025-03"
score_output_path = "research/models/scores.parquet"
metric_output_path = "research/models/metrics.json"

[analysis]
layer_output_path = "research/models/layers.json"

[qlib]
provider_uri = "~/.qlib/qlib_data/cn_data"
region = "cn"
market = "csi300"
model_name = "lgbm"
config_id = "qlib_minimal"

[model_backtest]
output_dir = "results/backtest"
top_k = 6
{extra}
"""


def test_qlib_minimal_config_derives_dates_without_factor_panel(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    _write_calendar(storage_root)
    config = load_research_config(_write_config(tmp_path, _minimal_qlib_config(storage_root)))

    assert config.factor_snapshot_path == ""
    assert config.factor_start_date == "2023-12-01"
    assert config.factor_as_of_date == "2025-03-31"
    assert config.backtest_start_date == "2025-01-02"
    assert config.backtest_end_date == "2025-03-28"


def test_qlib_legacy_fields_can_override_when_consistent(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    _write_calendar(storage_root)
    config_path = _write_config(
        tmp_path,
        f"""
[storage]
root = "{storage_root.as_posix()}"

[factor_spec]
id = "qlib_legacy"
universe_name = "tradable_core"
start_date = "2023-01-02"

[research_snapshot]
as_of_date = "2025-03-28"

[factors]
output_path = "research/factors/legacy.parquet"

[training]
label_column = "raw_fwd_return_5"
train_window_months = 12
validation_window_months = 1
test_start_month = "2025-01"
test_end_month = "2025-03"
score_output_path = "research/models/scores.parquet"
metric_output_path = "research/models/metrics.json"

[analysis]
layer_output_path = "research/models/layers.json"

[qlib]
market = "csi300"

[model_backtest]
output_dir = "results/backtest"
start_date = "2025-01-03"
end_date = "2025-03-28"
""",
    )

    config = load_research_config(config_path)

    assert config.factor_start_date == "2023-01-02"
    assert config.factor_as_of_date == "2025-03-28"
    assert config.factor_snapshot_path == "research/factors/legacy.parquet"
    assert config.backtest_start_date == "2025-01-03"
    assert config.backtest_end_date == "2025-03-28"


def test_qlib_rejects_late_factor_start_date(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    config_path = _write_config(
        tmp_path,
        _minimal_qlib_config(storage_root).replace(
            'universe_name = "tradable_core"',
            'universe_name = "tradable_core"\nstart_date = "2024-01-02"',
        ),
    )

    with pytest.raises(ValueError, match="factor_spec.start_date is later"):
        load_research_config(config_path)


def test_qlib_rejects_as_of_date_outside_test_end_month(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    config_path = _write_config(
        tmp_path,
        _minimal_qlib_config(storage_root).replace(
            "[training]",
            '[research_snapshot]\nas_of_date = "2025-04-01"\n\n[training]',
        ),
    )

    with pytest.raises(ValueError, match="as_of_date must fall within"):
        load_research_config(config_path)


def test_qlib_rejects_backtest_range_outside_score_coverage(tmp_path: Path) -> None:
    storage_root = tmp_path / "storage"
    config_path = _write_config(
        tmp_path,
        _minimal_qlib_config(
            storage_root,
            extra='start_date = "2025-01-02"\nend_date = "2025-04-01"',
        ),
    )

    with pytest.raises(ValueError, match="backtest range exceeds score coverage"):
        load_research_config(config_path)


def test_native_config_still_requires_factor_dates(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
        """
[storage]
root = "storage/demo"

[factor_spec]
id = "native"
universe_name = "tradable_core"
start_date = "2024-01-02"

[research_snapshot]
as_of_date = "2025-03-28"

[factors]
output_path = "research/factors/native.parquet"

[training]
label_column = "raw_fwd_return_5"
train_window_months = 12
validation_window_months = 1
test_start_month = "2025-01"
test_end_month = "2025-03"
score_output_path = "research/models/scores.parquet"
metric_output_path = "research/models/metrics.json"

[analysis]
layer_output_path = "research/models/layers.json"

[model_backtest]
output_dir = "results/backtest"
start_date = "2025-01-02"
end_date = "2025-03-28"
""",
    )

    config = load_research_config(config_path)

    assert config.factor_start_date == "2024-01-02"
    assert config.factor_as_of_date == "2025-03-28"
    assert config.factor_snapshot_path == "research/factors/native.parquet"
    assert config.backtest_start_date == "2025-01-02"
    assert config.backtest_end_date == "2025-03-28"


def test_qlib_pipeline_passes_derived_data_window_without_factor_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage_root = tmp_path / "storage"
    _write_calendar(storage_root)
    config_path = _write_config(tmp_path, _minimal_qlib_config(storage_root))
    captured = {}

    def fake_train_qlib_walk_forward(config):
        captured["config"] = config
        return {"window_count": 3, "mean_spearman_ic": 0.1}

    monkeypatch.setattr(research_commands, "train_qlib_walk_forward", fake_train_qlib_walk_forward)
    monkeypatch.setattr(
        research_commands,
        "analyze_score_layers",
        lambda config: {"summary": {"rows": 1, "mean_top_bottom_spread": 0.2}},
    )

    result = research_commands.run_qlib_research_pipeline(config_path.as_posix())

    assert result["factor_path"] == ""
    qlib_config = captured["config"]
    assert qlib_config.data_start_date == "2023-12-01"
    assert qlib_config.data_end_date == "2025-03-31"
    assert qlib_config.test_start_month == "2025-01"
    assert qlib_config.test_end_month == "2025-03"


def test_resolve_research_config_path_finds_qlib_subdirectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "configs" / "qlib"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "qlib_strategy.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert resolve_research_config_path(factor_spec_id="qlib_strategy") == config_path.resolve()


def test_score_file_payload_attaches_config_path_from_score_metadata() -> None:
    scores = [
        {
            "path": "research/qlib/models/walk_forward_scores.parquet",
            "config_id": "demo_strategy",
        }
    ]
    presets = [
        {
            "id": "demo_strategy",
            "factor_spec_id": "demo_strategy",
            "score_output_path": "research/qlib/models/other_scores.parquet",
            "config_path": "configs/qlib/demo_strategy.toml",
        }
    ]

    enriched = _attach_score_config_paths(scores, presets)

    assert enriched[0]["config_path"] == "configs/qlib/demo_strategy.toml"
