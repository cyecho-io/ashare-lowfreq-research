from __future__ import annotations

import csv
import json
import logging
import os
import re
import sqlite3
import threading
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from ashare_backtest.data import ParquetDataProvider
from ashare_backtest.data import DEFAULT_SQLITE_SOURCE
from ashare_backtest.cli.main import (
    generate_strategy_state_from_config,
    run_model_backtest,
    train_walk_forward_single_date_from_config,
)
from ashare_backtest.cli.research_config import ResearchRunConfig, load_research_config, resolve_dated_output_path
from ashare_backtest.factors import FactorBuildConfig, FactorBuilder, resolve_factor_snapshot_path
from ashare_backtest.logging_utils import configure_file_logging, get_logger
from ashare_backtest.research import StrategyStateConfig, generate_strategy_state
from ashare_backtest.research.trainer import WalkForwardAsOfDateConfig, train_lightgbm_walk_forward_as_of_date

REPO_ROOT = Path(__file__).resolve().parents[3]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
RESULTS_ROOT = REPO_ROOT / "results"
WEB_RUNS_ROOT = RESULTS_ROOT / "web_runs"
PAPER_RUNS_ROOT = RESULTS_ROOT / "paper_runs"
SIMULATION_ACCOUNTS_ROOT = RESULTS_ROOT / "simulation_accounts"
SIMULATION_RUNS_ROOT = SIMULATION_ACCOUNTS_ROOT
SIMULATION_PLANS_ROOT = SIMULATION_ACCOUNTS_ROOT
SIMULATION_LATEST_ROOT = RESULTS_ROOT / "simulation_latest"
CONFIG_ROOT = REPO_ROOT / "configs"
BARS_PATH = REPO_ROOT / "storage" / "parquet" / "bars" / "daily.parquet"
BENCHMARK_PATH = REPO_ROOT / "storage" / "parquet" / "benchmarks" / "000300.SH.parquet"
SCORE_SOURCE_MANIFEST_PATH = REPO_ROOT / "research" / "models" / "score_source_manifest.json"
CATALOG_PATH = REPO_ROOT / "storage" / "catalog.json"
LOGGER = get_logger("web.app")


@dataclass(frozen=True)
class StrategyPreset:
    id: str
    name: str
    config_path: str
    factor_spec_id: str
    score_output_path: str
    paper_score_output_path: str
    paper_source_kind: str
    paper_score_start_date: str
    paper_score_end_date: str
    latest_signal_date: str
    latest_execution_date: str
    model_backtest_output_dir: str
    default_start_date: str
    default_end_date: str
    initial_cash: float
    top_k: int
    rebalance_every: int
    min_hold_bars: int
    keep_buffer: int


def _previous_open_trade_date(trade_date_text: str, storage_root: str = "storage") -> str:
    target_trade_date = date.fromisoformat(trade_date_text)
    calendar_path = _resolve_repo_path(storage_root) / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    frame = pd.read_parquet(calendar_path, columns=["trade_date", "is_open"])
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["is_open"] = frame["is_open"].astype(bool)
    trade_dates = [
        item.date()
        for item in frame.loc[
            (frame["trade_date"].notna())
            & (frame["is_open"])
            & (frame["trade_date"] <= pd.Timestamp(target_trade_date)),
            "trade_date",
        ].sort_values()
    ]
    if target_trade_date not in trade_dates:
        raise ValueError(f"trade date is not an open trading day: {trade_date_text}")
    target_index = trade_dates.index(target_trade_date)
    if target_index == 0:
        raise ValueError("cannot derive signal date for the first available trade date")
    return trade_dates[target_index - 1].isoformat()


def _next_open_trade_date(trade_date_text: str, storage_root: str = "storage") -> str:
    target_trade_date = date.fromisoformat(trade_date_text)
    calendar_path = _resolve_repo_path(storage_root) / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    frame = pd.read_parquet(calendar_path, columns=["trade_date", "is_open"])
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["is_open"] = frame["is_open"].astype(bool)
    trade_dates = [
        item.date()
        for item in frame.loc[
            (frame["trade_date"].notna())
            & (frame["is_open"])
            & (frame["trade_date"] >= pd.Timestamp(target_trade_date)),
            "trade_date",
        ].sort_values()
    ]
    if target_trade_date not in trade_dates:
        raise ValueError(f"trade date is not an open trading day: {trade_date_text}")
    target_index = trade_dates.index(target_trade_date)
    if target_index + 1 >= len(trade_dates):
        raise ValueError(f"next open trade date is unavailable after: {trade_date_text}")
    return trade_dates[target_index + 1].isoformat()


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(self, job_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._jobs[job_id] = payload

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(changes)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else dict(job)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(job) for _, job in sorted(self._jobs.items(), key=lambda item: item[0], reverse=True)]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "run"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_equity_curve(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "trade_date": row["trade_date"],
                    "equity": float(row["equity"]),
                }
            )
    return rows


def _read_trades(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "trade_date": row["trade_date"],
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "quantity": int(float(row["quantity"] or 0)),
                    "price": float(row["price"] or 0.0),
                    "amount": float(row["amount"] or 0.0),
                    "commission": float(row["commission"] or 0.0),
                    "tax": float(row["tax"] or 0.0),
                    "slippage": float(row["slippage"] or 0.0),
                    "status": row["status"],
                    "reason": row["reason"],
                }
            )
    return rows


def _read_decision_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "trade_date": row["trade_date"],
                    "signal_date": row["signal_date"],
                    "decision_reason": row["decision_reason"],
                    "should_rebalance": str(row["should_rebalance"]).strip().lower() == "true",
                    "selected_symbols": row["selected_symbols"],
                    "current_position_count": int(float(row["current_position_count"] or 0)),
                    "target_position_count": int(float(row["target_position_count"] or 0)),
                    "cash_pre_decision": float(row["cash_pre_decision"] or 0.0),
                }
            )
    return rows


def _append_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_simulation_account_event(account_id: str, payload: dict[str, Any], results_root: Path = SIMULATION_ACCOUNTS_ROOT) -> None:
    _append_csv_rows(
        _simulation_account_events_path(account_id, results_root=results_root),
        [
            "event_at",
            "event_type",
            "account_id",
            "node_id",
            "strategy_id",
            "signal_date",
            "execution_date",
            "source_kind",
        ],
        [payload],
    )


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _sync_simulation_account_files(
    *,
    account_id: str,
    meta_payload: dict[str, Any],
    strategy_state: dict[str, Any],
    node_dir: Path,
    append_account_trades: bool = True,
    results_root: Path = SIMULATION_ACCOUNTS_ROOT,
) -> None:
    account_dir = _simulation_account_dir(account_id, results_root=results_root)
    account_dir.mkdir(parents=True, exist_ok=True)
    _simulation_account_meta_path(account_id, results_root=results_root).write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _simulation_account_state_path(account_id, results_root=results_root).write_text(
        json.dumps(strategy_state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    trades_path = node_dir / "trades.csv"
    if append_account_trades and trades_path.exists():
        trades = _read_trades(trades_path)
        _append_csv_rows(
            _simulation_account_trades_path(account_id, results_root=results_root),
            ["trade_date", "symbol", "side", "quantity", "price", "amount", "commission", "tax", "slippage", "status", "reason"],
            [
                {
                    "trade_date": row["trade_date"],
                    "symbol": row["symbol"],
                    "side": row["side"],
                    "quantity": int(row["quantity"]),
                    "price": float(row["price"]),
                    "amount": float(row["amount"]),
                    "commission": float(row["commission"]),
                    "tax": float(row["tax"]),
                    "slippage": float(row["slippage"]),
                    "status": row["status"],
                    "reason": row["reason"],
                }
                for row in trades
            ],
        )

    decision_log_path = node_dir / "decision_log.csv"
    if decision_log_path.exists():
        decisions = _read_decision_log(decision_log_path)
        _append_csv_rows(
            _simulation_account_decision_log_path(account_id, results_root=results_root),
            [
                "trade_date",
                "signal_date",
                "decision_reason",
                "should_rebalance",
                "selected_symbols",
                "current_position_count",
                "target_position_count",
                "cash_pre_decision",
            ],
            [
                {
                    "trade_date": row["trade_date"],
                    "signal_date": row["signal_date"],
                    "decision_reason": row["decision_reason"],
                    "should_rebalance": row["should_rebalance"],
                    "selected_symbols": row["selected_symbols"],
                    "current_position_count": int(row["current_position_count"]),
                    "target_position_count": int(row["target_position_count"]),
                    "cash_pre_decision": float(row["cash_pre_decision"]),
                }
                for row in decisions
            ],
        )


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_catalog(catalog_path: Path = CATALOG_PATH) -> dict[str, Any]:
    if not catalog_path.exists():
        return {}
    return _read_json(catalog_path)


def _resolve_dashboard_sqlite_path(repo_root: Path = REPO_ROOT) -> Path:
    catalog = _read_catalog(repo_root / "storage" / "catalog.json")
    catalog_source = str(catalog.get("source_path") or "").strip()
    if catalog_source:
        candidate = catalog_source if Path(catalog_source).is_absolute() else (repo_root / catalog_source).as_posix()
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path
    repo_default = repo_root / "storage" / "source" / "ashare_arena_sync.db"
    if repo_default.exists():
        return repo_default
    fallback = Path(DEFAULT_SQLITE_SOURCE)
    return fallback


def _latest_strategy_dir(strategy_id: str) -> Path:
    return REPO_ROOT / "research" / "models" / "latest" / strategy_id


def _read_latest_manifest(strategy_id: str) -> dict[str, Any]:
    manifest_path = _latest_strategy_dir(strategy_id) / "manifest.json"
    if not manifest_path.exists():
        return {}
    payload = _read_json(manifest_path)
    payload["manifest_path"] = _display_path(manifest_path)
    return payload


def _paper_score_candidates(strategy_id: str, fallback_scores_path: str) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    manifest = _read_latest_manifest(strategy_id)
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for kind, path_text in (
        ("config_default", fallback_scores_path),
        ("latest_manifest_source", str(manifest.get("source_scores_path") or "").strip()),
        ("latest_manifest", str(manifest.get("scores_path") or "").strip()),
    ):
        normalized = str(path_text).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        candidates.append((kind, normalized))
    return manifest, candidates


def _score_date_range(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    frame = pd.read_parquet(path, columns=["trade_date"])
    if frame.empty:
        return "", ""
    trade_dates = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
    if trade_dates.empty:
        return "", ""
    return trade_dates.min().date().isoformat(), trade_dates.max().date().isoformat()


def _equity_curve_date_range(points: list[dict[str, Any]]) -> tuple[str, str]:
    trade_dates = [str(item.get("trade_date") or "").strip() for item in points if str(item.get("trade_date") or "").strip()]
    if not trade_dates:
        return "", ""
    return min(trade_dates), max(trade_dates)


def _open_trade_dates(storage_root: str) -> list[str]:
    calendar_path = _resolve_repo_path(storage_root) / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    frame = pd.read_parquet(calendar_path, columns=["trade_date", "is_open"])
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    return [
        item.date().isoformat()
        for item in frame.loc[
            frame["trade_date"].notna() & frame["is_open"].astype(bool),
            "trade_date",
        ].sort_values()
    ]


def _latest_simulatable_signal_date(storage_root: str) -> str:
    open_trade_dates = _open_trade_dates(storage_root)
    if len(open_trade_dates) < 2:
        raise ValueError("at least two open trade dates are required for simulation")
    return open_trade_dates[-2]


def load_score_source_manifest(manifest_path: Path | None = None) -> dict[str, dict[str, Any]]:
    manifest_path = manifest_path or SCORE_SOURCE_MANIFEST_PATH
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = payload.get("sources", [])
    if not isinstance(items, list):
        return {}
    manifest: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        path_text = str(item.get("scores_path") or "").strip()
        if not path_text:
            continue
        normalized = _display_path(_resolve_repo_path(path_text))
        manifest[normalized] = dict(item)
    return manifest


def _materialize_merged_paper_scores(strategy_id: str, candidate_paths: list[str]) -> str:
    target = _latest_strategy_dir(strategy_id) / "paper_history_scores.parquet"
    source_paths = [_resolve_repo_path(path_text) for path_text in candidate_paths if _resolve_repo_path(path_text).exists()]
    if not source_paths:
        return candidate_paths[0]
    latest_source_mtime = max(path.stat().st_mtime for path in source_paths)
    if target.exists() and target.stat().st_mtime >= latest_source_mtime:
        return _display_path(target)

    merged = pd.concat([pd.read_parquet(path) for path in source_paths], ignore_index=True)
    if {"trade_date", "symbol"}.issubset(merged.columns):
        merged = merged.sort_values(["trade_date", "symbol"]).drop_duplicates(["trade_date", "symbol"], keep="last")
    target.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(target, index=False)
    return _display_path(target)


def _resolve_paper_scores_path(strategy_id: str, fallback_scores_path: str) -> tuple[str, dict[str, Any], str]:
    manifest, candidates = _paper_score_candidates(strategy_id, fallback_scores_path)
    if not candidates:
        return fallback_scores_path, manifest, "config_default"
    existing_candidates = [(kind, path) for kind, path in candidates if _resolve_repo_path(path).exists()]
    if len(existing_candidates) == 1:
        return existing_candidates[0][1], manifest, existing_candidates[0][0]
    if len(existing_candidates) >= 2:
        merged_path = _materialize_merged_paper_scores(strategy_id, [path for _, path in existing_candidates])
        return merged_path, manifest, "merged_history"

    preferred_candidate = next(
        (item for item in candidates if item[0] == "latest_manifest_source"),
        next((item for item in candidates if item[0] == "latest_manifest"), candidates[0]),
    )
    return preferred_candidate[1], manifest, preferred_candidate[0]


def _iter_result_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(
        [
            path
            for path in results_root.rglob("*")
            if path.is_dir()
            and (path / "summary.json").exists()
            and (path / "equity_curve.csv").exists()
            and (path / "trades.csv").exists()
        ],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def list_score_parquet_files(
    models_root: Path = REPO_ROOT / "research" / "models",
    *,
    include_single_day: bool = True,
) -> list[dict[str, str]]:
    if not models_root.exists():
        return []
    manifest = load_score_source_manifest()
    files: list[dict[str, str]] = []
    for path in models_root.rglob("*.parquet"):
        if not path.is_file() or "scores" not in path.name:
            continue
        display_path = _display_path(path)
        start_date, end_date = _score_date_range(path)
        item = {
            "path": display_path,
            "start_date": start_date,
            "end_date": end_date,
        }
        if not include_single_day and start_date and end_date and start_date == end_date:
            continue
        metadata = manifest.get(display_path, {})
        if metadata:
            item.update(
                {
                    "score_source_id": str(metadata.get("score_source_id") or ""),
                    "strategy_id": str(metadata.get("strategy_id") or ""),
                    "factor_spec_id": str(metadata.get("factor_spec_id") or ""),
                    "factor_snapshot_date": str(metadata.get("factor_snapshot_date") or ""),
                    "factor_snapshot_path": str(
                        metadata.get("factor_snapshot_path") or metadata.get("factor_panel_path") or ""
                    ),
                    "factor_panel_path": str(metadata.get("factor_panel_path") or ""),
                    "config_path": str(metadata.get("config_path") or ""),
                    "source_kind": str(metadata.get("source_kind") or ""),
                    "description": str(metadata.get("description") or ""),
                    "supports_incremental_update": bool(metadata.get("supports_incremental_update", False)),
                }
            )
        files.append(item)
    deduped: dict[str, dict[str, str]] = {}
    for item in files:
        deduped[item["path"]] = item
    return [deduped[path] for path in sorted(deduped)]


def _resolve_selected_scores_path(scores_path: str, strategy_id: str, fallback_scores_path: str) -> tuple[str, dict[str, Any], str]:
    normalized = str(scores_path).strip()
    if normalized:
        absolute_scores = _resolve_repo_path(normalized)
        if not absolute_scores.exists():
            raise FileNotFoundError(f"score file not found: {normalized}")
        return _display_path(absolute_scores), {}, "user_selected"
    return _resolve_paper_scores_path(strategy_id, fallback_scores_path)


def build_paper_readiness(
    config_path: str,
    trade_date: str,
    scores_path: str = "",
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    absolute_config = (repo_root / config_path).resolve()
    if not absolute_config.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    config = load_research_config(absolute_config)
    resolved_scores_path, _, source_kind = _resolve_selected_scores_path(
        scores_path,
        absolute_config.stem,
        config.score_output_path,
    )
    score_start_date, score_end_date = _score_date_range(_resolve_repo_path(resolved_scores_path))
    manifest = load_score_source_manifest().get(resolved_scores_path, {})
    factor_snapshot_path = str(manifest.get("factor_snapshot_path") or manifest.get("factor_panel_path") or "")
    open_trade_dates = _open_trade_dates(config.storage_root)
    if trade_date not in open_trade_dates:
        return {
            "config_path": config_path,
            "trade_date": trade_date,
            "required_signal_date": "",
            "score_start_date": score_start_date,
            "score_end_date": score_end_date,
            "scores_path": resolved_scores_path,
            "source_kind": source_kind,
            "factor_snapshot_path": factor_snapshot_path,
            "factor_panel_path": str(manifest.get("factor_panel_path") or ""),
            "strategy_id": str(manifest.get("strategy_id") or absolute_config.stem),
            "is_ready": False,
            "missing_score_days": 0,
            "supports_incremental_update": bool(manifest.get("supports_incremental_update", False)),
            "message": f"执行日期 {trade_date} 不是交易日，或当前交易日历尚未覆盖该日期。",
        }
    target_index = open_trade_dates.index(trade_date)
    if target_index == 0:
        return {
            "config_path": config_path,
            "trade_date": trade_date,
            "required_signal_date": "",
            "score_start_date": score_start_date,
            "score_end_date": score_end_date,
            "scores_path": resolved_scores_path,
            "source_kind": source_kind,
            "factor_snapshot_path": factor_snapshot_path,
            "factor_panel_path": str(manifest.get("factor_panel_path") or ""),
            "strategy_id": str(manifest.get("strategy_id") or absolute_config.stem),
            "is_ready": False,
            "missing_score_days": 0,
            "supports_incremental_update": bool(manifest.get("supports_incremental_update", False)),
            "message": f"执行日期 {trade_date} 是当前交易日历中的首个交易日，无法推导上一交易日信号。",
        }
    signal_date = open_trade_dates[target_index - 1]
    is_ready = bool(score_end_date) and score_end_date >= signal_date
    missing_score_days = 0
    if not is_ready:
        lower_bound = score_end_date if score_end_date else ""
        missing_score_days = sum(1 for item in open_trade_dates if lower_bound < item <= signal_date)
    if is_ready:
        message = f"所选数据源已覆盖信号日 {signal_date}，可以生成 {trade_date} 盘前快照。"
    else:
        message = (
            f"所选数据源最新只到 {score_end_date or '无数据'}，"
            f"生成 {trade_date} 需要至少覆盖到信号日 {signal_date}，还差 {missing_score_days} 个交易日。"
        )
    return {
        "config_path": config_path,
        "trade_date": trade_date,
        "required_signal_date": signal_date,
        "score_start_date": score_start_date,
        "score_end_date": score_end_date,
        "scores_path": resolved_scores_path,
        "source_kind": source_kind,
        "factor_snapshot_path": factor_snapshot_path,
        "factor_panel_path": str(manifest.get("factor_panel_path") or ""),
        "strategy_id": str(manifest.get("strategy_id") or absolute_config.stem),
        "is_ready": is_ready,
        "missing_score_days": missing_score_days,
        "supports_incremental_update": bool(manifest.get("supports_incremental_update", False)),
        "message": message,
    }


def fill_scores_to_signal_date(
    config_path: str,
    trade_date: str,
    scores_path: str = "",
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    absolute_config = (repo_root / config_path).resolve()
    if not absolute_config.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    config = load_research_config(absolute_config)
    resolved_scores_path, manifest, _ = _resolve_selected_scores_path(
        scores_path,
        absolute_config.stem,
        config.score_output_path,
    )
    if not manifest:
        manifest = load_score_source_manifest().get(resolved_scores_path, {})
    if not bool(manifest.get("supports_incremental_update", False)):
        raise ValueError(f"selected scores source does not support incremental update: {resolved_scores_path}")

    factor_snapshot_path = str(manifest.get("factor_snapshot_path") or manifest.get("factor_panel_path") or "").strip()
    if not factor_snapshot_path:
        raise ValueError(f"incremental update requires factor_snapshot_path: {resolved_scores_path}")

    open_trade_dates = _open_trade_dates(config.storage_root)
    if trade_date not in open_trade_dates:
        raise ValueError(f"trade date is not an open trading day: {trade_date}")
    target_index = open_trade_dates.index(trade_date)
    if target_index == 0:
        raise ValueError("cannot derive signal date for the first available trade date")
    signal_date = open_trade_dates[target_index - 1]

    target_scores_path = _resolve_repo_path(resolved_scores_path)
    score_start_date, score_end_date = _score_date_range(target_scores_path)
    missing_dates = [item for item in open_trade_dates if (score_end_date or "") < item <= signal_date]
    if not missing_dates:
        return {
            "scores_path": resolved_scores_path,
            "filled_dates": [],
            "required_signal_date": signal_date,
            "score_start_date": score_start_date,
            "score_end_date": score_end_date,
            "filled_count": 0,
        }

    existing = pd.read_parquet(target_scores_path) if target_scores_path.exists() else pd.DataFrame()
    scored_parts: list[pd.DataFrame] = []
    temp_dir = Path(tempfile.mkdtemp(prefix="paper_fill_scores_"))
    try:
        for missing_date in missing_dates:
            temp_scores_path = temp_dir / f"scores_{missing_date}.parquet"
            temp_metrics_path = temp_dir / f"metrics_{missing_date}.json"
            train_lightgbm_walk_forward_as_of_date(
                WalkForwardAsOfDateConfig(
                    factor_panel_path=factor_snapshot_path,
                    output_scores_path=temp_scores_path.as_posix(),
                    output_metrics_path=temp_metrics_path.as_posix(),
                    label_column=str(manifest.get("label_column") or config.label_column).strip(),
                    as_of_date=missing_date,
                    train_window_months=int(manifest.get("train_window_months") or config.train_window_months),
                )
            )
            scored_parts.append(pd.read_parquet(temp_scores_path))
        merged = pd.concat([frame for frame in [existing, *scored_parts] if not frame.empty], ignore_index=True)
        if {"trade_date", "symbol"}.issubset(merged.columns):
            merged["trade_date"] = pd.to_datetime(merged["trade_date"], errors="coerce")
            merged = merged.sort_values(["trade_date", "symbol"]).drop_duplicates(["trade_date", "symbol"], keep="last")
        target_scores_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(target_scores_path, index=False)
    finally:
        for path in temp_dir.glob("*"):
            path.unlink(missing_ok=True)
        temp_dir.rmdir()

    new_start_date, new_end_date = _score_date_range(target_scores_path)
    return {
        "scores_path": resolved_scores_path,
        "filled_dates": missing_dates,
        "required_signal_date": signal_date,
        "score_start_date": new_start_date,
        "score_end_date": new_end_date,
        "filled_count": len(missing_dates),
    }


def _resolve_simulation_factor_panel_path(config: ResearchRunConfig, signal_date: str) -> str:
    return resolve_factor_snapshot_path(
        factor_spec_id=config.factor_spec_id,
        as_of_date=signal_date,
        universe_name=config.factor_universe_name,
        start_date=config.factor_start_date,
    )


def _ensure_simulation_factor_panel(config: ResearchRunConfig, signal_date: str) -> tuple[str, bool]:
    relative_factor_panel_path = _resolve_simulation_factor_panel_path(config, signal_date)
    absolute_factor_panel_path = _resolve_repo_path(relative_factor_panel_path)
    if absolute_factor_panel_path.exists():
        LOGGER.info(
            "reuse simulation factor panel signal_date=%s path=%s",
            signal_date,
            absolute_factor_panel_path.as_posix(),
        )
        return _display_path(absolute_factor_panel_path), False
    LOGGER.info(
        "build simulation factor panel signal_date=%s path=%s universe_name=%s",
        signal_date,
        absolute_factor_panel_path.as_posix(),
        config.factor_universe_name,
    )
    FactorBuilder(
        FactorBuildConfig(
            storage_root=config.storage_root,
            output_path=absolute_factor_panel_path.as_posix(),
            universe_name=config.factor_universe_name,
            start_date=config.factor_start_date,
            as_of_date=signal_date,
        )
    ).build()
    return _display_path(absolute_factor_panel_path), True


def build_simulation_readiness(
    config_path: str,
    signal_date: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    absolute_config = (repo_root / config_path).resolve()
    if not absolute_config.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    config = load_research_config(absolute_config)
    open_trade_dates = _open_trade_dates(config.storage_root)
    latest_signal_date = _latest_simulatable_signal_date(config.storage_root)
    factor_panel_path = _resolve_simulation_factor_panel_path(config, signal_date)
    score_output_path = resolve_dated_output_path(config.score_output_path, signal_date)
    factor_exists = _resolve_repo_path(factor_panel_path).exists()
    if signal_date not in open_trade_dates:
        return {
            "config_path": config_path,
            "signal_date": signal_date,
            "execution_date": "",
            "factor_panel_path": factor_panel_path,
            "factor_exists": factor_exists,
            "scores_path": score_output_path,
            "strategy_id": absolute_config.stem,
            "latest_signal_date": latest_signal_date,
            "is_ready": False,
            "message": f"信号截止日期 {signal_date} 不是交易日，或当前交易日历尚未覆盖该日期。",
        }
    if signal_date >= open_trade_dates[-1]:
        return {
            "config_path": config_path,
            "signal_date": signal_date,
            "execution_date": "",
            "factor_panel_path": factor_panel_path,
            "factor_exists": factor_exists,
            "scores_path": score_output_path,
            "strategy_id": absolute_config.stem,
            "latest_signal_date": latest_signal_date,
            "is_ready": False,
            "message": f"当前交易日历最新开市日是 {open_trade_dates[-1]}，请将信号截止日期选择到 {latest_signal_date} 或更早。",
        }
    try:
        execution_date = _next_open_trade_date(signal_date, storage_root=config.storage_root)
    except ValueError as exc:
        return {
            "config_path": config_path,
            "signal_date": signal_date,
            "execution_date": "",
            "factor_panel_path": factor_panel_path,
            "factor_exists": factor_exists,
            "scores_path": score_output_path,
            "strategy_id": absolute_config.stem,
            "latest_signal_date": latest_signal_date,
            "is_ready": False,
            "message": str(exc),
        }
    factor_status = "已存在" if factor_exists else "不存在，将按策略自动生成"
    return {
        "config_path": config_path,
        "signal_date": signal_date,
        "execution_date": execution_date,
        "factor_panel_path": factor_panel_path,
        "factor_exists": factor_exists,
        "scores_path": score_output_path,
        "strategy_id": absolute_config.stem,
        "latest_signal_date": latest_signal_date,
        "is_ready": True,
        "message": f"将基于 {signal_date} 收盘后信号生成 {execution_date} 盘前调仓方案。factor 快照{factor_status}。",
    }


def _iter_paper_dirs(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(
        [path for path in results_root.rglob("*") if path.is_dir() and (path / "strategy_state.json").exists()],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def _resolve_run_dir(run_id: str, results_root: Path) -> Path:
    safe_run_id = Path(run_id).name
    direct = results_root / safe_run_id
    if (
        direct.exists()
        and direct.is_dir()
        and (direct / "summary.json").exists()
        and (direct / "equity_curve.csv").exists()
        and (direct / "trades.csv").exists()
    ):
        return direct

    matches = [path for path in _iter_result_dirs(results_root) if path.name == safe_run_id]
    if not matches:
        raise FileNotFoundError(f"run not found: {run_id}")
    return matches[0]


def _resolve_paper_run_dir(run_id: str, results_root: Path) -> Path:
    safe_run_id = Path(run_id).name
    direct = results_root / safe_run_id
    if direct.exists() and direct.is_dir() and (direct / "strategy_state.json").exists():
        return direct

    matches = [path for path in _iter_paper_dirs(results_root) if path.name == safe_run_id]
    if not matches:
        raise FileNotFoundError(f"paper run not found: {run_id}")
    return matches[0]


def _simulation_latest_dir(strategy_id: str) -> Path:
    return SIMULATION_LATEST_ROOT / Path(strategy_id).name


def _simulation_account_dir(account_id: str, results_root: Path = SIMULATION_ACCOUNTS_ROOT) -> Path:
    return results_root / Path(account_id).name


def _simulation_account_meta_path(account_id: str, results_root: Path = SIMULATION_ACCOUNTS_ROOT) -> Path:
    return _simulation_account_dir(account_id, results_root=results_root) / "meta.json"


def _simulation_account_state_path(account_id: str, results_root: Path = SIMULATION_ACCOUNTS_ROOT) -> Path:
    return _simulation_account_dir(account_id, results_root=results_root) / "strategy_state.json"


def _simulation_account_trades_path(account_id: str, results_root: Path = SIMULATION_ACCOUNTS_ROOT) -> Path:
    return _simulation_account_dir(account_id, results_root=results_root) / "trades.csv"


def _simulation_account_decision_log_path(account_id: str, results_root: Path = SIMULATION_ACCOUNTS_ROOT) -> Path:
    return _simulation_account_dir(account_id, results_root=results_root) / "decision_log.csv"


def _simulation_account_events_path(account_id: str, results_root: Path = SIMULATION_ACCOUNTS_ROOT) -> Path:
    return _simulation_account_dir(account_id, results_root=results_root) / "events.csv"


def _simulation_plan_dir(account_id: str, plan_id: str, results_root: Path = SIMULATION_PLANS_ROOT) -> Path:
    return _simulation_account_dir(account_id, results_root=results_root) / "plans" / Path(plan_id).name


def _simulation_run_dir(account_id: str, run_id: str, results_root: Path = SIMULATION_RUNS_ROOT) -> Path:
    return _simulation_account_dir(account_id, results_root=results_root) / "runs" / Path(run_id).name


def _iter_simulation_plan_dirs(results_root: Path = SIMULATION_PLANS_ROOT) -> list[Path]:
    if not results_root.exists():
        return []
    nested = [
        path
        for path in results_root.rglob("plans/*")
        if path.is_dir() and (path / "strategy_state.json").exists()
    ]
    flat = [
        path
        for path in results_root.iterdir()
        if path.is_dir()
        and (path / "strategy_state.json").exists()
        and not (path / "plans").exists()
        and not (path / "runs").exists()
    ]
    return sorted(
        list({path.resolve(): path for path in [*nested, *flat]}.values()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def _iter_simulation_run_dirs(results_root: Path = SIMULATION_RUNS_ROOT) -> list[Path]:
    if not results_root.exists():
        return []
    nested = [
        path
        for path in results_root.rglob("runs/*")
        if path.is_dir() and (path / "strategy_state.json").exists()
    ]
    flat = [
        path
        for path in results_root.iterdir()
        if path.is_dir()
        and (path / "strategy_state.json").exists()
        and not (path / "plans").exists()
        and not (path / "runs").exists()
    ]
    return sorted(
        list({path.resolve(): path for path in [*nested, *flat]}.values()),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def _resolve_simulation_plan_dir(plan_id: str, results_root: Path = SIMULATION_PLANS_ROOT) -> Path:
    safe_plan_id = Path(plan_id).name
    matches = [path for path in _iter_simulation_plan_dirs(results_root) if path.name == safe_plan_id]
    if not matches:
        raise FileNotFoundError(f"simulation plan not found: {plan_id}")
    return matches[0]


def _resolve_simulation_run_dir(run_id: str, results_root: Path = SIMULATION_RUNS_ROOT) -> Path:
    safe_run_id = Path(run_id).name
    matches = [path for path in _iter_simulation_run_dirs(results_root) if path.name == safe_run_id]
    if not matches:
        raise FileNotFoundError(f"simulation run not found: {run_id}")
    return matches[0]


def _simulation_bars_ready(storage_root: str, trade_date: str) -> bool:
    bars_path = _resolve_repo_path(storage_root) / "parquet" / "bars" / "daily.parquet"
    if not bars_path.exists():
        return False
    try:
        frame = pd.read_parquet(
            bars_path,
            columns=["trade_date"],
            filters=[("trade_date", "==", pd.Timestamp(trade_date))],
        )
    except Exception:
        frame = pd.read_parquet(bars_path, columns=["trade_date"])
        frame = frame.loc[frame["trade_date"] == pd.Timestamp(trade_date)]
    return not frame.empty


def _read_latest_simulation_manifest(strategy_id: str) -> dict[str, Any]:
    manifest_path = _simulation_latest_dir(strategy_id) / "manifest.json"
    if not manifest_path.exists():
        return {}
    payload = _read_json(manifest_path)
    payload["manifest_path"] = _display_path(manifest_path)
    return payload


def _write_latest_simulation_manifest(strategy_id: str, payload: dict[str, Any]) -> None:
    latest_dir = _simulation_latest_dir(strategy_id)
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_equal_weight_benchmark_curve(
    equity_curve: list[dict[str, Any]],
    bars_path: Path = BARS_PATH,
) -> tuple[str, list[dict[str, Any]]]:
    if not equity_curve or not bars_path.exists():
        return "A股等权基准", []

    trade_dates = [item["trade_date"] for item in equity_curve]
    start_date = min(trade_dates)
    end_date = max(trade_dates)
    frame = pd.read_parquet(bars_path, columns=["trade_date", "symbol", "close_adj", "close", "is_suspended"])
    frame = frame.loc[
        (frame["trade_date"] >= pd.Timestamp(start_date))
        & (frame["trade_date"] <= pd.Timestamp(end_date))
        & (~frame["is_suspended"].fillna(False))
    ].copy()
    if frame.empty:
        return "A股等权基准", []

    frame["price"] = pd.to_numeric(frame["close_adj"], errors="coerce").fillna(pd.to_numeric(frame["close"], errors="coerce"))
    frame = frame.dropna(subset=["price"]).sort_values(["symbol", "trade_date"])
    if frame.empty:
        return "A股等权基准", []

    frame["daily_return"] = frame.groupby("symbol")["price"].pct_change()
    daily = (
        frame.groupby("trade_date", as_index=False)["daily_return"]
        .mean()
        .sort_values("trade_date")
    )
    daily["daily_return"] = daily["daily_return"].fillna(0.0)
    initial_equity = float(equity_curve[0]["equity"])
    daily["benchmark_equity"] = initial_equity * (1.0 + daily["daily_return"]).cumprod()
    benchmark_map = {
        row["trade_date"].date().isoformat(): float(row["benchmark_equity"])
        for _, row in daily.iterrows()
    }
    curve = [
        {
            "trade_date": item["trade_date"],
            "equity": benchmark_map.get(item["trade_date"], initial_equity),
        }
        for item in equity_curve
    ]
    return "A股等权基准", curve


def _build_cached_benchmark_curve(
    equity_curve: list[dict[str, Any]],
    benchmark_path: Path = BENCHMARK_PATH,
    label: str = "沪深300",
) -> tuple[str, list[dict[str, Any]]]:
    if not equity_curve or not benchmark_path.exists():
        return label, []
    frame = pd.read_parquet(benchmark_path)
    if frame.empty or "trade_date" not in frame.columns or "close" not in frame.columns:
        return label, []
    frame = frame.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "close"]).sort_values("trade_date")
    if frame.empty:
        return label, []

    trade_dates = [item["trade_date"] for item in equity_curve]
    start = pd.Timestamp(min(trade_dates))
    end = pd.Timestamp(max(trade_dates))
    frame = frame.loc[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)].copy()
    if frame.empty:
        return label, []

    base_close = float(frame["close"].iloc[0])
    initial_equity = float(equity_curve[0]["equity"])
    if base_close <= 0:
        return label, []
    frame["benchmark_equity"] = initial_equity * (frame["close"] / base_close)
    benchmark_map = {
        row["trade_date"].date().isoformat(): float(row["benchmark_equity"])
        for _, row in frame.iterrows()
    }
    curve = [
        {
            "trade_date": item["trade_date"],
            "equity": benchmark_map.get(item["trade_date"]),
        }
        for item in equity_curve
        if benchmark_map.get(item["trade_date"]) is not None
    ]
    return label, curve


def list_strategy_presets(config_root: Path = CONFIG_ROOT) -> list[StrategyPreset]:
    presets: list[StrategyPreset] = []
    for path in sorted(config_root.glob("*.toml")):
        try:
            config = load_research_config(path)
        except Exception:
            continue
        presets.append(_preset_from_config(path, config))
    return presets


def build_dashboard_summary(
    *,
    repo_root: Path = REPO_ROOT,
    config_root: Path = CONFIG_ROOT,
) -> dict[str, Any]:
    catalog = _read_catalog(repo_root / "storage" / "catalog.json")
    sqlite_path = _resolve_dashboard_sqlite_path(repo_root=repo_root)
    strategy_count = len(list_strategy_presets(config_root=config_root))

    calendar_path = repo_root / "storage" / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
    calendar_cells: list[dict[str, Any]] = []
    latest_open_date = ""
    latest_calendar_date = ""
    open_days = 0
    closed_days = 0
    if calendar_path.exists():
        frame = pd.read_parquet(calendar_path, columns=["trade_date", "is_open"])
        frame = frame.copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
        frame = frame.loc[frame["trade_date"].notna()].sort_values("trade_date")
        latest_calendar_date = frame["trade_date"].max().date().isoformat() if not frame.empty else ""
        open_frame = frame.loc[frame["is_open"].astype(bool)]
        latest_open_date = open_frame["trade_date"].max().date().isoformat() if not open_frame.empty else ""
        open_days = int(open_frame.shape[0])
        closed_days = int(frame.shape[0] - open_days)
        tail = frame.tail(140)
        calendar_cells = [
            {
                "date": row.trade_date.date().isoformat(),
                "is_open": bool(row.is_open),
                "weekday": int(row.trade_date.weekday()),
            }
            for row in tail.itertuples(index=False)
        ]

    equity_symbol_count = 0
    equity_date_min = ""
    equity_date_max = ""
    instrument_count = 0
    with sqlite3.connect(sqlite_path) as conn:
        bars_row = conn.execute(
            """
            select
                count(distinct symbol) as symbol_count,
                min(trade_date) as min_trade_date,
                max(trade_date) as max_trade_date
            from equity_daily_bars
            """
        ).fetchone()
        if bars_row is not None:
            equity_symbol_count = int(bars_row[0] or 0)
            equity_date_min = str(bars_row[1] or "")
            equity_date_max = str(bars_row[2] or "")
        instruments_row = conn.execute("select count(*) from equity_instruments").fetchone()
        instrument_count = int((instruments_row or [0])[0] or 0)

    datasets = {str(item.get("name") or ""): item for item in catalog.get("datasets", []) if isinstance(item, dict)}
    return {
        "calendar": {
            "latest_open_date": latest_open_date,
            "latest_calendar_date": latest_calendar_date,
            "open_days": open_days,
            "closed_days": closed_days,
            "cells": calendar_cells,
        },
        "strategies": {
            "count": strategy_count,
        },
        "sqlite": {
            "path": _display_path(sqlite_path),
            "equity_symbol_count": equity_symbol_count,
            "instrument_count": instrument_count,
            "date_min": equity_date_min,
            "date_max": equity_date_max,
        },
        "catalog": {
            "imported_at": str(catalog.get("imported_at") or ""),
            "bars_daily": datasets.get("bars.daily", {}),
            "calendar_ashare": datasets.get("calendar.ashare", {}),
            "instruments_ashare": datasets.get("instruments.ashare", {}),
        },
    }


def _preset_from_config(path: Path, config: ResearchRunConfig) -> StrategyPreset:
    display_name = path.stem.replace("_", " ")
    resolved_paper_scores_path, manifest, paper_source_kind = _resolve_paper_scores_path(path.stem, config.score_output_path)
    paper_score_start_date, paper_score_end_date = _score_date_range(_resolve_repo_path(resolved_paper_scores_path))
    return StrategyPreset(
        id=path.stem,
        name=display_name,
        config_path=path.relative_to(REPO_ROOT).as_posix(),
        factor_spec_id=config.factor_spec_id,
        score_output_path=config.score_output_path,
        paper_score_output_path=resolved_paper_scores_path,
        paper_source_kind=paper_source_kind,
        paper_score_start_date=paper_score_start_date,
        paper_score_end_date=paper_score_end_date,
        latest_signal_date=str(manifest.get("signal_date") or ""),
        latest_execution_date=str(manifest.get("execution_date") or ""),
        model_backtest_output_dir=config.model_backtest_output_dir,
        default_start_date=config.backtest_start_date,
        default_end_date=config.backtest_end_date,
        initial_cash=config.initial_cash,
        top_k=config.top_k,
        rebalance_every=config.rebalance_every,
        min_hold_bars=config.min_hold_bars,
        keep_buffer=config.keep_buffer,
    )


def list_run_summaries(results_root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for entry in _iter_result_dirs(results_root):
        summary_path = entry / "summary.json"
        equity_path = entry / "equity_curve.csv"
        trades_path = entry / "trades.csv"
        try:
            summary = _read_json(summary_path)
        except Exception:
            continue
        runs.append(
            {
                "id": entry.name,
                "name": entry.name,
                "result_dir": _display_path(entry),
                "updated_at": datetime.fromtimestamp(entry.stat().st_mtime).isoformat(timespec="seconds"),
                "summary": summary,
            }
        )
    return runs


def load_run_detail(
    run_id: str,
    results_root: Path = RESULTS_ROOT,
    bars_path: Path = BARS_PATH,
    benchmark_path: Path = BENCHMARK_PATH,
) -> dict[str, Any]:
    safe_run_id = Path(run_id).name
    target = _resolve_run_dir(safe_run_id, results_root)
    summary_path = target / "summary.json"
    equity_path = target / "equity_curve.csv"
    trades_path = target / "trades.csv"
    if not (summary_path.exists() and equity_path.exists() and trades_path.exists()):
        raise FileNotFoundError(f"run not found: {run_id}")
    try:
        summary = _read_json(summary_path)
        equity_curve = _read_equity_curve(equity_path)
        trades = _read_trades(trades_path)
    except Exception as exc:
        raise FileNotFoundError(f"run is unreadable: {run_id}") from exc
    benchmark_label, benchmark_curve = _build_cached_benchmark_curve(equity_curve, benchmark_path=benchmark_path)
    if not benchmark_curve:
        benchmark_label, benchmark_curve = _build_equal_weight_benchmark_curve(equity_curve, bars_path=bars_path)
    backtest_start_date, backtest_end_date = _equity_curve_date_range(equity_curve)
    strategy_state_path = target / "strategy_state_latest.json"
    strategy_state = _read_json(strategy_state_path) if strategy_state_path.exists() else None
    scores_path = str((strategy_state or {}).get("strategy_config", {}).get("scores_path") or "")
    score_start_date = ""
    score_end_date = ""
    if scores_path:
        score_start_date, score_end_date = _score_date_range(_resolve_repo_path(scores_path))
    return {
        "id": safe_run_id,
        "name": safe_run_id,
        "result_dir": _display_path(target),
        "summary": summary,
        "scores_path": scores_path,
        "backtest_start_date": backtest_start_date,
        "backtest_end_date": backtest_end_date,
        "score_start_date": score_start_date,
        "score_end_date": score_end_date,
        "equity_curve": equity_curve,
        "benchmark_label": benchmark_label,
        "benchmark_curve": benchmark_curve,
        "strategy_state": strategy_state,
        "trades": trades,
    }


def _read_optional_json(path: Path) -> dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def list_paper_trade_summaries(results_root: Path = PAPER_RUNS_ROOT) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for entry in _iter_paper_dirs(results_root):
        strategy_state_path = entry / "strategy_state.json"
        meta_path = entry / "meta.json"
        try:
            strategy_state = _read_json(strategy_state_path)
            meta = _read_optional_json(meta_path)
        except Exception:
            continue
        runs.append(
            {
                "id": entry.name,
                "name": str(meta.get("name") or entry.name),
                "result_dir": _display_path(entry),
                "updated_at": datetime.fromtimestamp(entry.stat().st_mtime).isoformat(timespec="seconds"),
                "config_path": str(meta.get("config_path") or ""),
                "trade_date": str(strategy_state.get("summary", {}).get("execution_date", "")),
                "summary": strategy_state.get("summary", {}),
            }
        )
    return runs


def load_paper_trade_detail(run_id: str, results_root: Path = PAPER_RUNS_ROOT) -> dict[str, Any]:
    safe_run_id = Path(run_id).name
    target = _resolve_paper_run_dir(safe_run_id, results_root)
    strategy_state_path = target / "strategy_state.json"
    meta_path = target / "meta.json"
    if not strategy_state_path.exists():
        raise FileNotFoundError(f"paper run not found: {run_id}")
    try:
        strategy_state = _read_json(strategy_state_path)
        meta = _read_optional_json(meta_path)
    except Exception as exc:
        raise FileNotFoundError(f"paper run is unreadable: {run_id}") from exc
    return {
        "id": safe_run_id,
        "name": str(meta.get("name") or safe_run_id),
        "result_dir": _display_path(target),
        "config_path": str(meta.get("config_path") or ""),
        "created_at": str(meta.get("created_at") or ""),
        "scores_path": str(meta.get("scores_path") or ""),
        "paper_source_kind": str(meta.get("paper_source_kind") or ""),
        "latest_signal_date": str(meta.get("latest_signal_date") or ""),
        "latest_execution_date": str(meta.get("latest_execution_date") or ""),
        "latest_manifest_path": str(meta.get("latest_manifest_path") or ""),
        "strategy_state": strategy_state,
    }


def load_latest_paper_snapshot(strategy_id: str) -> dict[str, Any]:
    manifest = _read_latest_manifest(strategy_id)
    if not manifest:
        raise FileNotFoundError(f"latest manifest not found: {strategy_id}")

    strategy_state_path = str(manifest.get("strategy_state_path") or "").strip()
    if not strategy_state_path:
        raise FileNotFoundError(f"latest strategy state path missing: {strategy_id}")

    absolute_state_path = REPO_ROOT / strategy_state_path
    if not absolute_state_path.exists():
        raise FileNotFoundError(f"latest strategy state not found: {strategy_id}")

    strategy_state = _read_json(absolute_state_path)
    return {
        "id": strategy_id,
        "name": strategy_id,
        "result_dir": _display_path(absolute_state_path.parent),
        "scores_path": str(manifest.get("scores_path") or ""),
        "trades_path": str(manifest.get("trades_path") or ""),
        "paper_source_kind": "latest_manifest",
        "latest_signal_date": str(manifest.get("signal_date") or ""),
        "latest_execution_date": str(manifest.get("execution_date") or ""),
        "latest_manifest_path": str(manifest.get("manifest_path") or ""),
        "strategy_state": strategy_state,
    }


def load_paper_history_detail(
    strategy_id: str,
    config_root: Path | None = None,
    results_root: Path | None = None,
) -> dict[str, Any]:
    manifest = _read_latest_manifest(strategy_id)
    latest_trades_path = str(manifest.get("trades_path") or "").strip()
    latest_state_path = str(manifest.get("strategy_state_path") or "").strip()
    if latest_trades_path and latest_state_path:
        absolute_trades_path = REPO_ROOT / latest_trades_path
        absolute_state_path = REPO_ROOT / latest_state_path
        if absolute_trades_path.exists() and absolute_state_path.exists():
            trades = _read_trades(absolute_trades_path)
            strategy_state = _read_json(absolute_state_path)
            summary = dict(strategy_state.get("summary", {}))
            summary["trade_count"] = len(trades)
            summary["filled_trade_count"] = sum(1 for trade in trades if trade["status"] == "filled")
            summary["rejected_trade_count"] = sum(1 for trade in trades if trade["status"] == "rejected")
            return {
                "strategy_id": strategy_id,
                "run_id": "latest",
                "result_dir": _display_path(absolute_trades_path.parent),
                "summary": summary,
                "equity_curve": [],
                "benchmark_label": "",
                "benchmark_curve": [],
                "trades": trades,
                "strategy_state": strategy_state,
                "source_kind": "latest_trade_log",
            }
    raise FileNotFoundError(f"latest trade log not found: {strategy_id}")


def load_latest_paper_lineage(strategy_id: str) -> dict[str, Any]:
    manifest = _read_latest_manifest(strategy_id)
    if not manifest:
        raise FileNotFoundError(f"latest manifest not found: {strategy_id}")

    decision_log_path = str(manifest.get("decision_log_path") or "").strip()
    trades_path = str(manifest.get("trades_path") or "").strip()
    state_path = str(manifest.get("strategy_state_path") or "").strip()
    if not decision_log_path or not state_path:
        raise FileNotFoundError(f"latest lineage not found: {strategy_id}")

    absolute_decision_log = REPO_ROOT / decision_log_path
    absolute_state_path = REPO_ROOT / state_path
    absolute_trades_path = REPO_ROOT / trades_path if trades_path else None
    if not absolute_decision_log.exists() or not absolute_state_path.exists():
        raise FileNotFoundError(f"latest lineage not found: {strategy_id}")

    return {
        "strategy_id": strategy_id,
        "decision_log": _read_decision_log(absolute_decision_log),
        "trades": _read_trades(absolute_trades_path) if absolute_trades_path and absolute_trades_path.exists() else [],
        "strategy_state": _read_json(absolute_state_path),
        "latest_signal_date": str(manifest.get("signal_date") or ""),
        "latest_execution_date": str(manifest.get("execution_date") or ""),
        "source_kind": "latest_lineage",
    }


def _find_previous_simulation_run(strategy_id: str, trade_date: str, results_root: Path = SIMULATION_RUNS_ROOT) -> dict[str, Any] | None:
    target_date = str(trade_date).strip()
    candidates: list[dict[str, Any]] = []
    for entry in _iter_simulation_run_dirs(results_root):
        strategy_state_path = entry / "strategy_state.json"
        meta_path = entry / "meta.json"
        if not strategy_state_path.exists():
            continue
        try:
            strategy_state = _read_json(strategy_state_path)
            meta = _read_optional_json(meta_path)
        except Exception:
            continue
        if str(meta.get("strategy_id") or "").strip() != strategy_id:
            continue
        execution_date = str(strategy_state.get("summary", {}).get("execution_date", "")).strip()
        if not execution_date or execution_date >= target_date:
            continue
        candidates.append(
            {
                "run_id": entry.name,
                "result_dir": _display_path(entry),
                "execution_date": execution_date,
                "strategy_state_path": _display_path(strategy_state_path),
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item["execution_date"], item["run_id"]))


def _find_simulation_run_for_plan(plan_id: str, results_root: Path = SIMULATION_RUNS_ROOT) -> dict[str, Any] | None:
    target_plan_id = Path(plan_id).name
    matches: list[dict[str, Any]] = []
    for entry in _iter_simulation_run_dirs(results_root):
        meta_path = entry / "meta.json"
        strategy_state_path = entry / "strategy_state.json"
        if not strategy_state_path.exists():
            continue
        try:
            meta = _read_optional_json(meta_path)
            strategy_state = _read_json(strategy_state_path)
        except Exception:
            continue
        if str(meta.get("plan_id") or "").strip() != target_plan_id:
            continue
        matches.append(
            {
                "run_id": entry.name,
                "execution_date": str(strategy_state.get("summary", {}).get("execution_date", "") or ""),
            }
        )
    if not matches:
        return None
    return max(matches, key=lambda item: (item["execution_date"], item["run_id"]))


def _simulation_account_id(meta: dict[str, Any], run_id: str) -> str:
    account_id = str(meta.get("account_id") or "").strip()
    return account_id or run_id


def _find_latest_simulation_plan_for_account(account_id: str, results_root: Path = SIMULATION_PLANS_ROOT) -> dict[str, Any] | None:
    target_account_id = str(account_id).strip()
    if not target_account_id:
        return None
    latest_match: dict[str, Any] | None = None
    latest_key: tuple[str, str] | None = None
    for entry in _iter_simulation_plan_dirs(results_root):
        meta = _read_optional_json(entry / "meta.json")
        current_account_id = str(meta.get("account_id") or entry.name).strip()
        if current_account_id != target_account_id:
            continue
        try:
            strategy_state = _read_json(entry / "strategy_state.json")
        except Exception:
            continue
        summary = strategy_state.get("summary", {})
        current_key = (str(summary.get("execution_date") or ""), entry.name)
        if latest_key is None or current_key > latest_key:
            latest_key = current_key
            latest_match = {
                "plan_id": entry.name,
                "account_id": current_account_id,
                "execution_date": current_key[0],
            }
    return latest_match


def _load_simulation_account_snapshot(account_id: str, results_root: Path = SIMULATION_ACCOUNTS_ROOT) -> dict[str, Any]:
    safe_account_id = Path(account_id).name
    account_dir = _simulation_account_dir(safe_account_id, results_root=results_root)
    meta_path = account_dir / "meta.json"
    state_path = account_dir / "strategy_state.json"
    trades_path = account_dir / "trades.csv"
    decision_log_path = account_dir / "decision_log.csv"
    events_path = account_dir / "events.csv"
    if not state_path.exists():
        raise FileNotFoundError(f"simulation account not found: {account_id}")
    meta = _read_optional_json(meta_path)
    strategy_state = _read_json(state_path)
    return {
        "account_id": safe_account_id,
        "meta": meta,
        "strategy_state": strategy_state,
        "trades": _read_trades(trades_path) if trades_path.exists() else [],
        "decision_log": _read_decision_log(decision_log_path) if decision_log_path.exists() else [],
        "events": list(csv.DictReader(events_path.open("r", encoding="utf-8", newline=""))) if events_path.exists() else [],
        "result_dir": _display_path(account_dir),
    }


def list_simulation_plan_summaries(results_root: Path = SIMULATION_PLANS_ROOT) -> list[dict[str, Any]]:
    latest_by_account: dict[str, dict[str, Any]] = {}
    for entry in _iter_simulation_plan_dirs(results_root):
        strategy_state_path = entry / "strategy_state.json"
        meta_path = entry / "meta.json"
        try:
            strategy_state = _read_json(strategy_state_path)
            meta = _read_optional_json(meta_path)
        except Exception:
            continue
        summary = strategy_state.get("summary", {})
        execution_date = str(summary.get("execution_date", "") or "")
        storage_root = str(strategy_state.get("strategy_config", {}).get("storage_root") or "storage")
        executed_run_id = str(meta.get("executed_run_id") or "").strip()
        if not executed_run_id:
            match = _find_simulation_run_for_plan(entry.name)
            executed_run_id = str(match.get("run_id") or "") if match else ""
        item = {
            "id": entry.name,
            "account_id": str(meta.get("account_id") or entry.name),
            "name": str(meta.get("name") or entry.name),
            "strategy_id": str(meta.get("strategy_id") or ""),
            "result_dir": _display_path(entry),
            "updated_at": datetime.fromtimestamp(entry.stat().st_mtime).isoformat(timespec="seconds"),
            "config_path": str(meta.get("config_path") or ""),
            "signal_date": str(summary.get("signal_date", "") or ""),
            "trade_date": execution_date,
            "execution_ready": _simulation_bars_ready(storage_root, execution_date) if execution_date else False,
            "executed_run_id": executed_run_id,
            "status": "executed" if executed_run_id else "pending_execution",
            "current_node_id": entry.name,
            "current_node_type": "executed" if executed_run_id else "planned",
            "summary": summary,
        }
        existing = latest_by_account.get(item["account_id"])
        if existing is None or (item["trade_date"], item["id"]) > (str(existing.get("trade_date") or ""), str(existing.get("id") or "")):
            latest_by_account[item["account_id"]] = item
    return sorted(latest_by_account.values(), key=lambda item: (str(item.get("trade_date") or ""), str(item.get("id") or "")), reverse=True)


def load_simulation_plan_detail(plan_id: str, results_root: Path = SIMULATION_PLANS_ROOT) -> dict[str, Any]:
    safe_plan_id = Path(plan_id).name
    target = _resolve_simulation_plan_dir(safe_plan_id, results_root)
    strategy_state_path = target / "strategy_state.json"
    meta_path = target / "meta.json"
    if not strategy_state_path.exists():
        raise FileNotFoundError(f"simulation plan not found: {plan_id}")
    try:
        strategy_state = _read_json(strategy_state_path)
        meta = _read_optional_json(meta_path)
    except Exception as exc:
        raise FileNotFoundError(f"simulation plan is unreadable: {plan_id}") from exc
    plan_strategy_state = dict(strategy_state or {})
    execution_date = str(plan_strategy_state.get("summary", {}).get("execution_date", "") or "")
    storage_root = str(plan_strategy_state.get("strategy_config", {}).get("storage_root") or "storage")
    execution_pending = bool(plan_strategy_state.get("next_state", {}).get("execution_pending", False))
    executed_run_id = str(meta.get("executed_run_id") or "").strip()
    if not executed_run_id:
        match = _find_simulation_run_for_plan(safe_plan_id)
        executed_run_id = str(match.get("run_id") or "") if match else ""
    bars_ready = _simulation_bars_ready(storage_root, execution_date) if execution_date else False
    execution_ready = bool(bars_ready and execution_pending and not executed_run_id)
    account_id = str(meta.get("account_id") or safe_plan_id)
    account_snapshot: dict[str, Any] | None = None
    try:
        account_snapshot = _load_simulation_account_snapshot(account_id)
    except FileNotFoundError:
        account_snapshot = None
    latest_plan = _find_latest_simulation_plan_for_account(account_id, results_root=results_root)
    is_latest_plan = bool(latest_plan and str(latest_plan.get("plan_id") or "") == safe_plan_id)
    next_plan_ready = bool(is_latest_plan and (executed_run_id or not execution_pending))
    next_plan_message = (
        f"将基于账户 {account_id} 在 {execution_date} 收盘后的执行结果生成下一交易日盘前计划。"
        if next_plan_ready
        else (
            "请先完成当前计划的模拟下单，再继续生成下一交易日盘前计划。"
            if not executed_run_id
            else "当前账户已经存在更新的盘前计划，请在最新计划详情页继续往后滚动。"
        )
    )
    if not executed_run_id and not execution_pending:
        next_plan_message = f"当前计划无待执行订单，可直接基于 {execution_date} 收盘后的账户状态生成下一交易日盘前计划。"
    return {
        "id": safe_plan_id,
        "account_id": account_id,
        "name": str(meta.get("name") or safe_plan_id),
        "strategy_id": str(meta.get("strategy_id") or ""),
        "result_dir": str((account_snapshot or {}).get("result_dir") or _display_path(target)),
        "config_path": str(meta.get("config_path") or ""),
        "created_at": str(meta.get("created_at") or ""),
        "scores_path": str(meta.get("scores_path") or ""),
        "source_kind": str(meta.get("source_kind") or "simulation_plan"),
        "previous_run_id": str(meta.get("previous_run_id") or ""),
        "executed_run_id": executed_run_id,
        "current_node_id": safe_plan_id,
        "current_node_type": "executed" if executed_run_id else "planned",
        "execution_pending": execution_pending,
        "execution_ready": execution_ready,
        "execution_status": (
            "executed"
            if executed_run_id
            else ("ready" if execution_ready else ("not_needed" if not execution_pending else "waiting_market_data"))
        ),
        "next_plan_ready": next_plan_ready,
        "next_plan_message": next_plan_message,
        "execution_message": (
            f"执行日 {execution_date} 行情已就绪，可模拟下单。"
            if execution_ready
            else (
                f"计划已执行，关联 run: {executed_run_id}"
                if executed_run_id
                else (
                    f"当前计划在 {execution_date} 无待执行订单，无需模拟下单。"
                    if not execution_pending
                    else f"执行日 {execution_date} 行情尚未落地，当前只能查看盘前计划。"
                )
            )
        ),
        "account_status_message": (
            f"账户已执行到 {execution_date}，可继续生成下一交易日盘前计划。"
            if executed_run_id
            else (
                f"当前计划在 {execution_date} 无待执行订单，可直接继续生成下一交易日盘前计划。"
                if not execution_pending
                else (f"执行日 {execution_date} 行情已就绪，可更新账户状态。" if execution_ready else f"执行日 {execution_date} 行情尚未落地。")
            )
        ),
        "strategy_state": dict((account_snapshot or {}).get("strategy_state") or strategy_state),
    }


def list_simulation_summaries(results_root: Path = SIMULATION_RUNS_ROOT) -> list[dict[str, Any]]:
    latest_by_account: dict[str, dict[str, Any]] = {}
    for entry in _iter_simulation_run_dirs(results_root):
        strategy_state_path = entry / "strategy_state.json"
        meta_path = entry / "meta.json"
        try:
            strategy_state = _read_json(strategy_state_path)
            meta = _read_optional_json(meta_path)
        except Exception:
            continue
        summary = strategy_state.get("summary", {})
        item = {
            "id": entry.name,
            "account_id": _simulation_account_id(meta, entry.name),
            "name": str(meta.get("name") or entry.name),
            "strategy_id": str(meta.get("strategy_id") or ""),
            "result_dir": _display_path(entry),
            "updated_at": datetime.fromtimestamp(entry.stat().st_mtime).isoformat(timespec="seconds"),
            "config_path": str(meta.get("config_path") or ""),
            "trade_date": str(summary.get("execution_date", "")),
            "summary": summary,
        }
        existing = latest_by_account.get(item["account_id"])
        if existing is None or (item["trade_date"], item["id"]) > (str(existing.get("trade_date") or ""), str(existing.get("id") or "")):
            latest_by_account[item["account_id"]] = item
    return sorted(latest_by_account.values(), key=lambda item: (str(item.get("trade_date") or ""), str(item.get("id") or "")), reverse=True)


def load_simulation_detail(run_id: str, results_root: Path = SIMULATION_RUNS_ROOT) -> dict[str, Any]:
    safe_run_id = Path(run_id).name
    target = _resolve_simulation_run_dir(safe_run_id, results_root)
    strategy_state_path = target / "strategy_state.json"
    meta_path = target / "meta.json"
    if not strategy_state_path.exists():
        raise FileNotFoundError(f"simulation run not found: {run_id}")
    try:
        strategy_state = _read_json(strategy_state_path)
        meta = _read_optional_json(meta_path)
    except Exception as exc:
        raise FileNotFoundError(f"simulation run is unreadable: {run_id}") from exc
    return {
        "id": safe_run_id,
        "account_id": _simulation_account_id(meta, safe_run_id),
        "name": str(meta.get("name") or safe_run_id),
        "strategy_id": str(meta.get("strategy_id") or ""),
        "result_dir": _display_path(target),
        "config_path": str(meta.get("config_path") or ""),
        "created_at": str(meta.get("created_at") or ""),
        "scores_path": str(meta.get("scores_path") or ""),
        "source_kind": str(meta.get("source_kind") or ""),
        "previous_state_path": str(meta.get("previous_state_path") or ""),
        "previous_run_id": str(meta.get("previous_run_id") or ""),
        "current_node_id": safe_run_id,
        "current_node_type": "executed",
        "account_status_message": "账户已完成模拟下单并更新到当前状态。",
        "strategy_state": strategy_state,
        "trades": _read_trades(target / "trades.csv") if (target / "trades.csv").exists() else [],
        "decision_log": _read_decision_log(target / "decision_log.csv") if (target / "decision_log.csv").exists() else [],
    }


def load_latest_simulation_snapshot(strategy_id: str) -> dict[str, Any]:
    manifest = _read_latest_simulation_manifest(strategy_id)
    if not manifest:
        raise FileNotFoundError(f"latest simulation manifest not found: {strategy_id}")
    strategy_state_path = str(manifest.get("strategy_state_path") or "").strip()
    if not strategy_state_path:
        raise FileNotFoundError(f"latest simulation state path missing: {strategy_id}")
    absolute_state_path = REPO_ROOT / strategy_state_path
    if not absolute_state_path.exists():
        raise FileNotFoundError(f"latest simulation state not found: {strategy_id}")
    return {
        "id": str(manifest.get("run_id") or strategy_id),
        "account_id": str(manifest.get("account_id") or manifest.get("run_id") or strategy_id),
        "name": str(manifest.get("name") or strategy_id),
        "strategy_id": strategy_id,
        "result_dir": _display_path(absolute_state_path.parent),
        "scores_path": str(manifest.get("scores_path") or ""),
        "source_kind": "simulation_latest",
        "latest_signal_date": str(manifest.get("signal_date") or ""),
        "latest_execution_date": str(manifest.get("execution_date") or ""),
        "previous_run_id": str(manifest.get("previous_run_id") or ""),
        "strategy_state": _read_json(absolute_state_path),
        "trades": _read_trades(REPO_ROOT / str(manifest.get("trades_path") or "").strip())
        if str(manifest.get("trades_path") or "").strip() and (REPO_ROOT / str(manifest.get("trades_path") or "").strip()).exists()
        else [],
        "decision_log": _read_decision_log(REPO_ROOT / str(manifest.get("decision_log_path") or "").strip())
        if str(manifest.get("decision_log_path") or "").strip()
        and (REPO_ROOT / str(manifest.get("decision_log_path") or "").strip()).exists()
        else [],
    }


def load_simulation_history_detail(strategy_id: str, run_id: str = "", results_root: Path = SIMULATION_RUNS_ROOT) -> dict[str, Any]:
    account_id = ""
    if run_id:
        detail = load_simulation_detail(run_id, results_root=results_root)
        account_id = str(detail.get("account_id") or "").strip()
        strategy_id = str(detail.get("strategy_id") or strategy_id).strip()
    matching_runs: list[dict[str, Any]] = []
    for entry in _iter_simulation_run_dirs(results_root):
        try:
            detail = load_simulation_detail(entry.name, results_root=results_root)
        except FileNotFoundError:
            continue
        if strategy_id and str(detail.get("strategy_id") or "").strip() != strategy_id:
            continue
        if account_id and str(detail.get("account_id") or "").strip() != account_id:
            continue
        matching_runs.append(detail)
    if not matching_runs:
        if account_id:
            try:
                snapshot = _load_simulation_account_snapshot(account_id)
                summary = dict(snapshot.get("strategy_state", {}).get("summary", {}))
                trades = list(snapshot.get("trades", []))
                summary["trade_count"] = len(trades)
                summary["filled_trade_count"] = sum(1 for trade in trades if trade.get("status") == "filled")
                summary["rejected_trade_count"] = sum(1 for trade in trades if trade.get("status") == "rejected")
                return {
                    "strategy_id": strategy_id or str(snapshot.get("meta", {}).get("strategy_id") or ""),
                    "account_id": account_id,
                    "run_id": str(run_id or ""),
                    "result_dir": str(snapshot.get("result_dir") or ""),
                    "summary": summary,
                    "trades": trades,
                    "strategy_state": snapshot.get("strategy_state", {}),
                    "source_kind": "simulation_account_history",
                }
            except FileNotFoundError:
                pass
        raise FileNotFoundError(f"simulation history not found: {strategy_id}")
    ordered_runs = sorted(
        matching_runs,
        key=lambda item: (
            str(item.get("strategy_state", {}).get("summary", {}).get("execution_date", "") or ""),
            str(item.get("id") or ""),
        ),
    )
    trades: list[dict[str, Any]] = []
    latest_detail: dict[str, Any] | None = None
    for detail in ordered_runs:
        latest_detail = detail
        trades.extend(detail.get("trades", []))
    summary = dict((latest_detail or {}).get("strategy_state", {}).get("summary", {}))
    summary["trade_count"] = len(trades)
    summary["filled_trade_count"] = sum(1 for trade in trades if trade.get("status") == "filled")
    summary["rejected_trade_count"] = sum(1 for trade in trades if trade.get("status") == "rejected")
    return {
        "strategy_id": strategy_id,
        "account_id": account_id or str((latest_detail or {}).get("account_id") or ""),
        "run_id": str((latest_detail or {}).get("id") or ""),
        "result_dir": str((latest_detail or {}).get("result_dir") or ""),
        "summary": summary,
        "trades": trades,
        "strategy_state": (latest_detail or {}).get("strategy_state", {}),
        "source_kind": "simulation_history",
    }


def load_simulation_lineage(strategy_id: str, run_id: str = "", results_root: Path = SIMULATION_RUNS_ROOT) -> dict[str, Any]:
    account_id = ""
    if run_id:
        detail = load_simulation_detail(run_id, results_root=results_root)
        account_id = str(detail.get("account_id") or "").strip()
        strategy_id = str(detail.get("strategy_id") or strategy_id).strip()
    if account_id:
        try:
            snapshot = _load_simulation_account_snapshot(account_id)
            summary = dict(snapshot.get("strategy_state", {}).get("summary", {}))
            return {
                "strategy_id": strategy_id or str(snapshot.get("meta", {}).get("strategy_id") or ""),
                "account_id": account_id,
                "decision_log": snapshot.get("decision_log", []),
                "trades": snapshot.get("trades", []),
                "strategy_state": snapshot.get("strategy_state", {}),
                "latest_signal_date": str(summary.get("signal_date") or ""),
                "latest_execution_date": str(summary.get("execution_date") or ""),
                "source_kind": "simulation_account_lineage",
            }
        except FileNotFoundError:
            pass
    matching_runs: list[dict[str, Any]] = []
    for entry in _iter_simulation_run_dirs(results_root):
        try:
            detail = load_simulation_detail(entry.name, results_root=results_root)
        except FileNotFoundError:
            continue
        if strategy_id and str(detail.get("strategy_id") or "").strip() != strategy_id:
            continue
        if account_id and str(detail.get("account_id") or "").strip() != account_id:
            continue
        matching_runs.append(detail)
    if not matching_runs:
        raise FileNotFoundError(f"simulation lineage not found: {strategy_id}")
    decision_log: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    latest_detail: dict[str, Any] | None = None
    for detail in sorted(matching_runs, key=lambda item: (str(item.get("strategy_state", {}).get("summary", {}).get("execution_date", "") or ""), str(item.get("id") or ""))):
        latest_detail = detail
        decision_log.extend(detail.get("decision_log", []))
        trades.extend(detail.get("trades", []))
    summary = dict((latest_detail or {}).get("strategy_state", {}).get("summary", {}))
    return {
        "strategy_id": strategy_id,
        "account_id": account_id or str((latest_detail or {}).get("account_id") or ""),
        "decision_log": decision_log,
        "trades": trades,
        "strategy_state": (latest_detail or {}).get("strategy_state", {}),
        "latest_signal_date": str(summary.get("signal_date") or ""),
        "latest_execution_date": str(summary.get("execution_date") or ""),
        "source_kind": "simulation_lineage",
    }


def _build_run_args(
    config: ResearchRunConfig,
    start_date: str,
    end_date: str,
    initial_cash: float,
    output_dir: str,
    scores_path: str | None = None,
) -> dict[str, Any]:
    return {
        "scores_path": scores_path or config.score_output_path,
        "storage_root": config.storage_root,
        "start_date": start_date,
        "end_date": end_date,
        "top_k": config.top_k,
        "rebalance_every": config.rebalance_every,
        "lookback_window": config.lookback_window,
        "min_hold_bars": config.min_hold_bars,
        "keep_buffer": config.keep_buffer,
        "min_turnover_names": config.min_turnover_names,
        "min_daily_amount": config.min_daily_amount,
        "max_close_price": config.max_close_price,
        "max_names_per_industry": config.max_names_per_industry,
        "max_position_weight": config.max_position_weight,
        "exit_policy": config.exit_policy,
        "grace_rank_buffer": config.grace_rank_buffer,
        "grace_momentum_window": config.grace_momentum_window,
        "grace_min_return": config.grace_min_return,
        "trailing_stop_window": config.trailing_stop_window,
        "trailing_stop_drawdown": config.trailing_stop_drawdown,
        "trailing_stop_min_gain": config.trailing_stop_min_gain,
        "score_reversal_confirm_days": config.score_reversal_confirm_days,
        "score_reversal_threshold": config.score_reversal_threshold,
        "hybrid_price_window": config.hybrid_price_window,
        "hybrid_price_threshold": config.hybrid_price_threshold,
        "strong_keep_extra_buffer": config.strong_keep_extra_buffer,
        "strong_keep_momentum_window": config.strong_keep_momentum_window,
        "strong_keep_min_return": config.strong_keep_min_return,
        "strong_trim_slowdown": config.strong_trim_slowdown,
        "strong_trim_momentum_window": config.strong_trim_momentum_window,
        "strong_trim_min_return": config.strong_trim_min_return,
        "initial_cash": initial_cash,
        "commission_rate": config.commission_rate,
        "stamp_tax_rate": config.stamp_tax_rate,
        "slippage_rate": config.slippage_rate,
        "max_trade_participation_rate": config.max_trade_participation_rate,
        "max_pending_days": config.max_pending_days,
        "output_dir": output_dir,
    }


def _build_strategy_state_args(
    config: ResearchRunConfig,
    scores_path: str,
    trade_date: str,
    initial_cash: float,
    output_path: str,
    *,
    mode: str = "historical",
    previous_state_path: str = "",
    simulate_trade_execution: bool = True,
) -> StrategyStateConfig:
    return StrategyStateConfig(
        scores_path=scores_path,
        storage_root=config.storage_root,
        output_path=output_path,
        trade_date=trade_date,
        universe_name=config.factor_universe_name,
        mode=mode,
        previous_state_path=previous_state_path,
        simulate_trade_execution=simulate_trade_execution,
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
        initial_cash=initial_cash,
        commission_rate=config.commission_rate,
        stamp_tax_rate=config.stamp_tax_rate,
        slippage_rate=config.slippage_rate,
        max_trade_participation_rate=config.max_trade_participation_rate,
        max_pending_days=config.max_pending_days,
    )


def _latest_trade_date_from_result(output_dir: Path) -> str | None:
    equity_path = output_dir / "equity_curve.csv"
    if not equity_path.exists():
        return None
    rows = _read_equity_curve(equity_path)
    if not rows:
        return None
    return str(rows[-1]["trade_date"])


def _build_strategy_state_snapshot(
    config: ResearchRunConfig,
    initial_cash: float,
    output_dir: Path,
    scores_path: str | None = None,
) -> None:
    latest_trade_date = _latest_trade_date_from_result(output_dir)
    if not latest_trade_date:
        return
    generate_strategy_state(
        _build_strategy_state_args(
            config=config,
            scores_path=scores_path or config.score_output_path,
            trade_date=latest_trade_date,
            initial_cash=initial_cash,
            output_path=(output_dir / "strategy_state_latest.json").as_posix(),
        )
    )


class BacktestWebApp:
    def __init__(self, repo_root: Path = REPO_ROOT) -> None:
        self.repo_root = repo_root
        self.job_store = JobStore()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="backtest-web")
        LOGGER.info("backtest web app initialized repo_root=%s", self.repo_root.as_posix())

    def submit_backtest(
        self,
        config_path: str,
        start_date: str,
        end_date: str,
        initial_cash: float,
        label: str,
        scores_path: str = "",
    ) -> dict[str, Any]:
        absolute_config = (self.repo_root / config_path).resolve()
        if not absolute_config.exists():
            raise FileNotFoundError(f"config not found: {config_path}")
        config = load_research_config(absolute_config)
        resolved_scores_path = scores_path or config.score_output_path
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{timestamp}-{_slugify(label or absolute_config.stem)}"
        output_dir = (WEB_RUNS_ROOT / run_name).relative_to(self.repo_root).as_posix()
        job_id = run_name
        job_payload = {
            "id": job_id,
            "status": "queued",
            "config_path": config_path,
            "start_date": start_date,
            "end_date": end_date,
            "initial_cash": initial_cash,
            "scores_path": resolved_scores_path,
            "result_dir": output_dir,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "error": "",
        }
        self.job_store.create(job_id, job_payload)
        args = _build_run_args(config, start_date, end_date, initial_cash, output_dir, scores_path=resolved_scores_path)
        self.executor.submit(self._run_job, job_id, args, config, initial_cash, resolved_scores_path)
        return job_payload

    def submit_paper_trade(
        self,
        config_path: str,
        trade_date: str,
        initial_cash: float,
        label: str,
        scores_path: str = "",
    ) -> dict[str, Any]:
        absolute_config = (self.repo_root / config_path).resolve()
        if not absolute_config.exists():
            raise FileNotFoundError(f"config not found: {config_path}")
        config = load_research_config(absolute_config)
        paper_scores_path, manifest, paper_source_kind = _resolve_selected_scores_path(
            scores_path,
            absolute_config.stem,
            config.score_output_path,
        )
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{timestamp}-{_slugify(label or absolute_config.stem)}"
        output_dir = self.repo_root / PAPER_RUNS_ROOT.relative_to(self.repo_root) / run_name
        job_id = run_name
        job_payload = {
            "id": job_id,
            "type": "paper",
            "status": "queued",
            "config_path": config_path,
            "trade_date": trade_date,
            "initial_cash": initial_cash,
            "scores_path": paper_scores_path,
            "paper_source_kind": paper_source_kind,
            "result_dir": _display_path(output_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "error": "",
        }
        self.job_store.create(job_id, job_payload)
        self.executor.submit(
            self._run_paper_job,
            job_id,
            config,
            paper_scores_path,
            paper_source_kind,
            trade_date,
            initial_cash,
            output_dir,
            label or absolute_config.stem,
            config_path,
            manifest,
        )
        return job_payload

    def submit_paper_fill_scores(
        self,
        config_path: str,
        trade_date: str,
        scores_path: str = "",
    ) -> dict[str, Any]:
        absolute_config = (self.repo_root / config_path).resolve()
        if not absolute_config.exists():
            raise FileNotFoundError(f"config not found: {config_path}")
        config = load_research_config(absolute_config)
        resolved_scores_path, manifest, _ = _resolve_selected_scores_path(
            scores_path,
            absolute_config.stem,
            config.score_output_path,
        )
        if not bool(manifest.get("supports_incremental_update", False)):
            raise ValueError(f"selected scores source does not support incremental update: {resolved_scores_path}")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_id = f"{timestamp}-{_slugify(absolute_config.stem)}-fill-scores"
        job_payload = {
            "id": job_id,
            "type": "fill_scores",
            "status": "queued",
            "config_path": config_path,
            "trade_date": trade_date,
            "scores_path": resolved_scores_path,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "error": "",
        }
        self.job_store.create(job_id, job_payload)
        self.executor.submit(self._run_fill_scores_job, job_id, config_path, trade_date, resolved_scores_path)
        return job_payload

    def submit_simulation_plan(
        self,
        config_path: str,
        signal_date: str,
        initial_cash: float,
        label: str,
    ) -> dict[str, Any]:
        absolute_config = (self.repo_root / config_path).resolve()
        if not absolute_config.exists():
            raise FileNotFoundError(f"config not found: {config_path}")
        config = load_research_config(absolute_config)
        execution_date = _next_open_trade_date(signal_date, storage_root=config.storage_root)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{timestamp}-{_slugify(label or absolute_config.stem)}"
        job_id = run_name
        account_id = job_id
        output_dir = _simulation_plan_dir(account_id, run_name)
        job_payload = {
            "id": job_id,
            "account_id": account_id,
            "type": "simulation_plan",
            "status": "queued",
            "config_path": config_path,
            "signal_date": signal_date,
            "trade_date": execution_date,
            "initial_cash": initial_cash,
            "scores_path": resolve_dated_output_path(config.score_output_path, signal_date),
            "source_kind": "simulation_plan",
            "result_dir": _display_path(output_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "error": "",
            "previous_run_id": "",
            "previous_state_path": "",
        }
        self.job_store.create(job_id, job_payload)
        self.executor.submit(
            self._run_simulation_plan_job,
            job_id,
            absolute_config.stem,
            config,
            signal_date,
            execution_date,
            initial_cash,
            output_dir,
            label or absolute_config.stem,
            config_path,
            account_id,
            "",
            "",
        )
        return job_payload

    def submit_simulation_next_plan(self, plan_id: str) -> dict[str, Any]:
        detail = load_simulation_plan_detail(plan_id, results_root=SIMULATION_PLANS_ROOT)
        executed_run_id = str(detail.get("executed_run_id") or "").strip()
        if not bool(detail.get("next_plan_ready")):
            raise ValueError(str(detail.get("next_plan_message") or "next simulation plan is not ready"))
        execution_pending = bool(detail.get("execution_pending"))
        if executed_run_id:
            run_detail = load_simulation_detail(executed_run_id, results_root=SIMULATION_RUNS_ROOT)
            config_path = str(run_detail.get("config_path") or "").strip()
            strategy_state = run_detail.get("strategy_state", {})
            summary = strategy_state.get("summary", {}) if isinstance(strategy_state, dict) else {}
            signal_date = str(summary.get("execution_date") or "").strip()
            initial_cash = float(strategy_state.get("strategy_config", {}).get("initial_cash") or 0.0)
            account_id = str(run_detail.get("account_id") or run_detail.get("id") or "").strip()
            previous_state_path = str(run_detail.get("result_dir") or "").strip()
            if not previous_state_path:
                raise ValueError(f"simulation run missing result_dir: {executed_run_id}")
            previous_strategy_state_path = (Path(previous_state_path) / "strategy_state.json").as_posix()
            previous_run_id = executed_run_id
        else:
            if execution_pending:
                raise ValueError(f"simulation plan not executed yet: {plan_id}")
            config_path = str(detail.get("config_path") or "").strip()
            strategy_state = detail.get("strategy_state", {})
            summary = strategy_state.get("summary", {}) if isinstance(strategy_state, dict) else {}
            signal_date = str(summary.get("execution_date") or "").strip()
            initial_cash = float(strategy_state.get("strategy_config", {}).get("initial_cash") or 0.0)
            account_id = str(detail.get("account_id") or detail.get("id") or "").strip()
            plan_dir = _resolve_simulation_plan_dir(plan_id, results_root=SIMULATION_PLANS_ROOT)
            previous_strategy_state_path = (plan_dir / "strategy_state.json").as_posix()
            previous_run_id = str(detail.get("previous_run_id") or "")
        if not config_path:
            raise ValueError(f"simulation plan missing config_path: {plan_id}")
        if not signal_date:
            raise ValueError(f"simulation plan missing execution_date: {plan_id}")

        absolute_config = (self.repo_root / config_path).resolve()
        if not absolute_config.exists():
            raise FileNotFoundError(f"config not found: {config_path}")
        config = load_research_config(absolute_config)
        execution_date = _next_open_trade_date(signal_date, storage_root=config.storage_root)
        if initial_cash <= 0:
            initial_cash = float(strategy_state.get("strategy_config", {}).get("initial_cash") or config.initial_cash)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        next_label = str(detail.get("name") or absolute_config.stem)
        run_name = f"{timestamp}-{_slugify(next_label)}"
        job_id = run_name
        output_dir = _simulation_plan_dir(account_id or job_id, run_name)
        job_payload = {
            "id": job_id,
            "account_id": account_id or job_id,
            "type": "simulation_plan",
            "status": "queued",
            "config_path": config_path,
            "signal_date": signal_date,
            "trade_date": execution_date,
            "initial_cash": initial_cash,
            "scores_path": resolve_dated_output_path(config.score_output_path, signal_date),
            "source_kind": "simulation_plan_roll_forward",
            "result_dir": _display_path(output_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "error": "",
            "previous_run_id": previous_run_id,
            "previous_state_path": _display_path(Path(previous_strategy_state_path)),
        }
        self.job_store.create(job_id, job_payload)
        self.executor.submit(
            self._run_simulation_plan_job,
            job_id,
            absolute_config.stem,
            config,
            signal_date,
            execution_date,
            initial_cash,
            output_dir,
            next_label,
            config_path,
            account_id,
            previous_strategy_state_path,
            previous_run_id,
        )
        return job_payload

    def submit_simulation_execute_plan(self, plan_id: str, label: str = "") -> dict[str, Any]:
        detail = load_simulation_plan_detail(plan_id, results_root=SIMULATION_PLANS_ROOT)
        if not bool(detail.get("execution_pending")):
            raise ValueError(str(detail.get("execution_message") or f"simulation plan does not require execution: {plan_id}"))
        config_path = str(detail.get("config_path") or "").strip()
        if not config_path:
            raise ValueError(f"simulation plan missing config_path: {plan_id}")
        strategy_state = detail.get("strategy_state", {})
        summary = strategy_state.get("summary", {}) if isinstance(strategy_state, dict) else {}
        signal_date = str(summary.get("signal_date") or "").strip()
        execution_date = str(summary.get("execution_date") or "").strip()
        if not signal_date:
            raise ValueError(f"simulation plan missing signal_date: {plan_id}")
        if not execution_date:
            raise ValueError(f"simulation plan missing execution_date: {plan_id}")
        absolute_config = (self.repo_root / config_path).resolve()
        if not absolute_config.exists():
            raise FileNotFoundError(f"config not found: {config_path}")
        config = load_research_config(absolute_config)
        if not _simulation_bars_ready(config.storage_root, execution_date):
            raise ValueError(f"execution date bars not ready: {execution_date}")
        initial_cash = float(strategy_state.get("strategy_config", {}).get("initial_cash") or config.initial_cash)
        account_id = str(detail.get("account_id") or detail.get("id") or "").strip()
        plan_dir = _resolve_simulation_plan_dir(plan_id, results_root=SIMULATION_PLANS_ROOT)
        previous_strategy_state_path = (plan_dir / "strategy_state.json").as_posix()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        next_label = label or str(detail.get("name") or absolute_config.stem)
        run_name = f"{timestamp}-{_slugify(next_label)}"
        output_dir = _simulation_run_dir(account_id or run_name, run_name)
        job_id = run_name
        job_payload = {
            "id": job_id,
            "account_id": account_id or job_id,
            "type": "simulation_execute",
            "status": "queued",
            "config_path": config_path,
            "signal_date": signal_date,
            "trade_date": execution_date,
            "initial_cash": initial_cash,
            "scores_path": resolve_dated_output_path(config.score_output_path, signal_date),
            "source_kind": "simulation_plan_execute",
            "result_dir": _display_path(output_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "error": "",
            "plan_id": str(detail.get("id") or ""),
            "previous_run_id": str(detail.get("previous_run_id") or ""),
            "previous_state_path": _display_path(Path(previous_strategy_state_path)),
        }
        self.job_store.create(job_id, job_payload)
        self.executor.submit(
            self._run_simulation_execute_job,
            job_id,
            absolute_config.stem,
            config,
            signal_date,
            execution_date,
            initial_cash,
            output_dir,
            next_label,
            config_path,
            previous_strategy_state_path,
            str(detail.get("previous_run_id") or ""),
            account_id or job_id,
            str(detail.get("id") or ""),
        )
        return job_payload

    def submit_simulation_fill_scores(
        self,
        config_path: str,
        trade_date: str,
        scores_path: str = "",
    ) -> dict[str, Any]:
        return self.submit_paper_fill_scores(config_path=config_path, trade_date=trade_date, scores_path=scores_path)

    def submit_simulation(
        self,
        config_path: str,
        signal_date: str,
        initial_cash: float,
        label: str,
    ) -> dict[str, Any]:
        return self.submit_simulation_plan(
            config_path=config_path,
            signal_date=signal_date,
            initial_cash=initial_cash,
            label=label,
        )

    def submit_simulation_roll_forward(self, run_id: str, label: str = "") -> dict[str, Any]:
        detail = load_simulation_detail(run_id, results_root=SIMULATION_RUNS_ROOT)
        config_path = str(detail.get("config_path") or "").strip()
        if not config_path:
            raise ValueError(f"simulation run missing config_path: {run_id}")
        strategy_state = detail.get("strategy_state", {})
        summary = strategy_state.get("summary", {}) if isinstance(strategy_state, dict) else {}
        signal_date = str(summary.get("execution_date") or "").strip()
        if not signal_date:
            raise ValueError(f"simulation run missing execution_date: {run_id}")

        absolute_config = (self.repo_root / config_path).resolve()
        if not absolute_config.exists():
            raise FileNotFoundError(f"config not found: {config_path}")
        config = load_research_config(absolute_config)
        execution_date = _next_open_trade_date(signal_date, storage_root=config.storage_root)
        initial_cash = float(detail.get("strategy_state", {}).get("strategy_config", {}).get("initial_cash") or config.initial_cash)
        account_id = str(detail.get("account_id") or detail.get("id") or "").strip()
        previous_state_path = str(detail.get("result_dir") or "").strip()
        if not previous_state_path:
            raise ValueError(f"simulation run missing result_dir: {run_id}")
        previous_strategy_state_path = (Path(previous_state_path) / "strategy_state.json").as_posix()

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        next_label = label or str(detail.get("name") or absolute_config.stem)
        run_name = f"{timestamp}-{_slugify(next_label)}"
        output_dir = _simulation_run_dir(account_id or run_name, run_name)
        job_id = run_name
        job_payload = {
            "id": job_id,
            "account_id": account_id or job_id,
            "type": "simulation",
            "status": "queued",
            "config_path": config_path,
            "signal_date": signal_date,
            "trade_date": execution_date,
            "initial_cash": initial_cash,
            "scores_path": resolve_dated_output_path(config.score_output_path, signal_date),
            "source_kind": "roll_forward",
            "result_dir": _display_path(output_dir),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "error": "",
            "previous_run_id": str(detail.get("id") or ""),
            "previous_state_path": _display_path(Path(previous_strategy_state_path)),
        }
        self.job_store.create(job_id, job_payload)
        self.executor.submit(
            self._run_simulation_job,
            job_id,
            absolute_config.stem,
            config,
            signal_date,
            execution_date,
            initial_cash,
            output_dir,
            next_label,
            config_path,
            previous_strategy_state_path,
            str(detail.get("id") or ""),
            account_id or job_id,
        )
        return job_payload

    def _run_job(
        self,
        job_id: str,
        args: dict[str, Any],
        config: ResearchRunConfig,
        initial_cash: float,
        scores_path: str | None = None,
    ) -> None:
        self.job_store.update(job_id, status="running", started_at=datetime.now().isoformat(timespec="seconds"))
        try:
            output_dir = self.repo_root / Path(args["output_dir"])
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            cwd = Path.cwd()
            os.chdir(self.repo_root)
            try:
                run_model_backtest(**args)
                _build_strategy_state_snapshot(config, initial_cash, output_dir, scores_path=scores_path)
            finally:
                os.chdir(cwd)
            self.job_store.update(
                job_id,
                status="completed",
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )
        except Exception as exc:
            self.job_store.update(
                job_id,
                status="failed",
                error=str(exc),
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )

    def _run_paper_job(
        self,
        job_id: str,
        config: ResearchRunConfig,
        scores_path: str,
        paper_source_kind: str,
        trade_date: str,
        initial_cash: float,
        output_dir: Path,
        label: str,
        config_path: str,
        manifest: dict[str, Any],
    ) -> None:
        self.job_store.update(job_id, status="running", started_at=datetime.now().isoformat(timespec="seconds"))
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            generate_strategy_state_from_config(
                config_path=(self.repo_root / config_path).as_posix(),
                scores_path=scores_path,
                trade_date=trade_date,
                initial_cash=initial_cash,
                output_path=(output_dir / "strategy_state.json").as_posix(),
            )
            (output_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "name": label,
                        "config_path": config_path,
                        "trade_date": trade_date,
                        "initial_cash": initial_cash,
                        "scores_path": scores_path,
                        "paper_source_kind": paper_source_kind,
                        "latest_signal_date": str(manifest.get("signal_date") or ""),
                        "latest_execution_date": str(manifest.get("execution_date") or ""),
                        "latest_manifest_path": str(manifest.get("manifest_path") or ""),
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.job_store.update(
                job_id,
                status="completed",
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )
        except Exception as exc:
            self.job_store.update(
                job_id,
                status="failed",
                error=str(exc),
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )

    def _run_fill_scores_job(
        self,
        job_id: str,
        config_path: str,
        trade_date: str,
        scores_path: str,
    ) -> None:
        self.job_store.update(job_id, status="running", started_at=datetime.now().isoformat(timespec="seconds"))
        try:
            payload = fill_scores_to_signal_date(
                config_path=config_path,
                trade_date=trade_date,
                scores_path=scores_path,
                repo_root=self.repo_root,
            )
            self.job_store.update(
                job_id,
                status="completed",
                result=payload,
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )
        except Exception as exc:
            self.job_store.update(
                job_id,
                status="failed",
                error=str(exc),
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )

    def _run_simulation_plan_job(
        self,
        job_id: str,
        strategy_id: str,
        config: ResearchRunConfig,
        signal_date: str,
        execution_date: str,
        initial_cash: float,
        output_dir: Path,
        label: str,
        config_path: str,
        account_id: str = "",
        previous_state_path: str = "",
        previous_run_id: str = "",
    ) -> None:
        self._run_simulation_job(
            job_id=job_id,
            strategy_id=strategy_id,
            config=config,
            signal_date=signal_date,
            execution_date=execution_date,
            initial_cash=initial_cash,
            output_dir=output_dir,
            label=label,
            config_path=config_path,
            previous_state_path=previous_state_path,
            previous_run_id=previous_run_id,
            account_id=account_id,
            source_kind="simulation_plan",
            simulate_trade_execution=False,
            update_latest_manifest=False,
        )

    def _run_simulation_execute_job(
        self,
        job_id: str,
        strategy_id: str,
        config: ResearchRunConfig,
        signal_date: str,
        execution_date: str,
        initial_cash: float,
        output_dir: Path,
        label: str,
        config_path: str,
        previous_state_path: str = "",
        previous_run_id: str = "",
        account_id: str = "",
        plan_id: str = "",
    ) -> None:
        self._run_simulation_job(
            job_id=job_id,
            strategy_id=strategy_id,
            config=config,
            signal_date=signal_date,
            execution_date=execution_date,
            initial_cash=initial_cash,
            output_dir=output_dir,
            label=label,
            config_path=config_path,
            previous_state_path=previous_state_path,
            previous_run_id=previous_run_id,
            account_id=account_id,
            source_kind="simulation_plan_execute",
            simulate_trade_execution=True,
            update_latest_manifest=True,
            plan_id=plan_id,
        )

    def _run_simulation_job(
        self,
        job_id: str,
        strategy_id: str,
        config: ResearchRunConfig,
        signal_date: str,
        execution_date: str,
        initial_cash: float,
        output_dir: Path,
        label: str,
        config_path: str,
        previous_state_path: str = "",
        previous_run_id: str = "",
        account_id: str = "",
        *,
        source_kind: str = "generated_single_date",
        simulate_trade_execution: bool = True,
        update_latest_manifest: bool = True,
        plan_id: str = "",
    ) -> None:
        self.job_store.update(job_id, status="running", started_at=datetime.now().isoformat(timespec="seconds"))
        try:
            LOGGER.info(
                "simulation job start job_id=%s strategy_id=%s signal_date=%s execution_date=%s output_dir=%s previous_state=%s",
                job_id,
                strategy_id,
                signal_date,
                execution_date,
                output_dir.as_posix(),
                previous_state_path or "-",
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            factor_panel_path, factor_built = _ensure_simulation_factor_panel(config, signal_date)
            LOGGER.info(
                "simulation job factor ready job_id=%s factor_panel=%s factor_built=%s",
                job_id,
                factor_panel_path,
                factor_built,
            )
            train_walk_forward_single_date_from_config(
                config_path=(self.repo_root / config_path).as_posix(),
                test_month=signal_date[:7],
                as_of_date=signal_date,
                factor_panel_path=_resolve_repo_path(factor_panel_path).as_posix(),
            )
            scores_path = resolve_dated_output_path(config.score_output_path, signal_date)
            LOGGER.info(
                "simulation job scores ready job_id=%s scores_path=%s",
                job_id,
                scores_path,
            )
            previous_run = None
            resolved_previous_state_path = previous_state_path
            resolved_previous_run_id = previous_run_id
            if not resolved_previous_state_path:
                previous_run = _find_previous_simulation_run(strategy_id, execution_date)
                resolved_previous_state_path = str(previous_run.get("strategy_state_path") or "") if previous_run else ""
                resolved_previous_run_id = str(previous_run.get("run_id") or "") if previous_run else ""
            mode = "continue" if resolved_previous_state_path else "initial_entry"
            LOGGER.info(
                "simulation job state mode job_id=%s mode=%s resolved_previous_state=%s resolved_previous_run_id=%s",
                job_id,
                mode,
                resolved_previous_state_path or "-",
                resolved_previous_run_id or "-",
            )
            strategy_state = generate_strategy_state(
                _build_strategy_state_args(
                    config=config,
                    scores_path=scores_path,
                    trade_date=execution_date,
                    initial_cash=initial_cash,
                    output_path=(output_dir / "strategy_state.json").as_posix(),
                    mode=mode,
                    previous_state_path=resolved_previous_state_path,
                    simulate_trade_execution=simulate_trade_execution,
                )
            )
            meta_payload = {
                "name": label,
                "account_id": account_id or job_id,
                "strategy_id": strategy_id,
                "config_path": config_path,
                "signal_date": signal_date,
                "trade_date": execution_date,
                "initial_cash": initial_cash,
                "scores_path": scores_path,
                "source_kind": source_kind,
                "factor_panel_path": factor_panel_path,
                "factor_built": factor_built,
                "previous_state_path": resolved_previous_state_path,
                "previous_run_id": resolved_previous_run_id,
                "plan_id": plan_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            (output_dir / "meta.json").write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            _sync_simulation_account_files(
                account_id=meta_payload["account_id"],
                meta_payload=meta_payload,
                strategy_state=strategy_state,
                node_dir=output_dir,
                append_account_trades=simulate_trade_execution,
                results_root=SIMULATION_ACCOUNTS_ROOT,
            )
            _append_simulation_account_event(
                meta_payload["account_id"],
                {
                    "event_at": datetime.now().isoformat(timespec="seconds"),
                    "event_type": "execute_plan" if simulate_trade_execution else "generate_plan",
                    "account_id": meta_payload["account_id"],
                    "node_id": job_id,
                    "strategy_id": strategy_id,
                    "signal_date": signal_date,
                    "execution_date": execution_date,
                    "source_kind": source_kind,
                },
                results_root=SIMULATION_ACCOUNTS_ROOT,
            )
            if update_latest_manifest:
                manifest_payload = {
                    "strategy_id": strategy_id,
                    "account_id": meta_payload["account_id"],
                    "run_id": job_id,
                    "name": label,
                    "signal_date": str(strategy_state.get("summary", {}).get("signal_date", "")),
                    "execution_date": str(strategy_state.get("summary", {}).get("execution_date", "")),
                    "scores_path": scores_path,
                    "strategy_state_path": _simulation_account_state_path(meta_payload["account_id"]).relative_to(self.repo_root).as_posix(),
                    "trades_path": _simulation_account_trades_path(meta_payload["account_id"]).relative_to(self.repo_root).as_posix(),
                    "decision_log_path": _simulation_account_decision_log_path(meta_payload["account_id"]).relative_to(self.repo_root).as_posix(),
                    "previous_run_id": meta_payload["previous_run_id"],
                }
                _write_latest_simulation_manifest(strategy_id, manifest_payload)
            if plan_id:
                plan_detail = load_simulation_plan_detail(plan_id, results_root=SIMULATION_PLANS_ROOT)
                plan_meta_path = _simulation_plan_dir(str(plan_detail.get("account_id") or ""), plan_id) / "meta.json"
                if plan_meta_path.exists():
                    plan_meta = _read_optional_json(plan_meta_path)
                    plan_meta["executed_run_id"] = job_id
                    plan_meta["executed_at"] = datetime.now().isoformat(timespec="seconds")
                    plan_meta_path.write_text(json.dumps(plan_meta, ensure_ascii=False, indent=2), encoding="utf-8")
            LOGGER.info(
                "simulation job complete job_id=%s strategy_id=%s run_dir=%s selected=%s decision_reason=%s",
                job_id,
                strategy_id,
                output_dir.as_posix(),
                len(strategy_state.get("plan", {}).get("selected_symbols", [])),
                str(strategy_state.get("summary", {}).get("decision_reason", "")),
            )
            self.job_store.update(
                job_id,
                status="completed",
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )
        except Exception as exc:
            LOGGER.exception("simulation job failed job_id=%s strategy_id=%s error=%s", job_id, strategy_id, exc)
            self.job_store.update(
                job_id,
                status="failed",
                error=str(exc),
                finished_at=datetime.now().isoformat(timespec="seconds"),
            )


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "AshareBacktestWeb/0.1"

    @property
    def app(self) -> BacktestWebApp:
        return self.server.app  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._serve_file(STATIC_ROOT / "dashboard.html", "text/html; charset=utf-8")
            return
        if path == "/backtest":
            self._serve_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if path == "/dashboard":
            self._serve_file(STATIC_ROOT / "dashboard.html", "text/html; charset=utf-8")
            return
        if path == "/simulation":
            self._serve_file(STATIC_ROOT / "simulation.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            relative = path.removeprefix("/static/")
            content_type = "text/plain; charset=utf-8"
            if relative.endswith(".css"):
                content_type = "text/css; charset=utf-8"
            elif relative.endswith(".js"):
                content_type = "application/javascript; charset=utf-8"
            self._serve_file(STATIC_ROOT / relative, content_type)
            return
        if path == "/api/strategies":
            presets = [preset.__dict__ for preset in list_strategy_presets()]
            self._send_json({"strategies": presets, "score_files": list_score_parquet_files(include_single_day=False)})
            return
        if path == "/api/paper/strategies":
            presets = [preset.__dict__ for preset in list_strategy_presets()]
            self._send_json({"strategies": presets, "score_files": list_score_parquet_files(include_single_day=False)})
            return
        if path == "/api/simulation/strategies":
            presets = [preset.__dict__ for preset in list_strategy_presets()]
            self._send_json({"strategies": presets, "score_files": list_score_parquet_files()})
            return
        if path == "/api/dashboard/summary":
            self._send_json(build_dashboard_summary(repo_root=self.app.repo_root, config_root=CONFIG_ROOT))
            return
        if path == "/api/paper/readiness":
            config_path = str(parse_qs(parsed.query).get("config_path", [""])[0]).strip()
            trade_date = str(parse_qs(parsed.query).get("trade_date", [""])[0]).strip()
            scores_path = str(parse_qs(parsed.query).get("scores_path", [""])[0]).strip()
            if not config_path or not trade_date:
                self._send_json({"error": "missing_required_fields"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                payload = build_paper_readiness(config_path=config_path, trade_date=trade_date, scores_path=scores_path)
            except (FileNotFoundError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)
            return
        if path == "/api/simulation/readiness":
            config_path = str(parse_qs(parsed.query).get("config_path", [""])[0]).strip()
            signal_date = str(parse_qs(parsed.query).get("signal_date", [""])[0]).strip() or str(
                parse_qs(parsed.query).get("trade_date", [""])[0]
            ).strip()
            if not config_path or not signal_date:
                self._send_json({"error": "missing_required_fields"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                payload = build_simulation_readiness(config_path=config_path, signal_date=signal_date)
            except (FileNotFoundError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json(payload)
            return
        if path == "/api/runs":
            self._send_json({"runs": list_run_summaries()[:40]})
            return
        if path == "/api/paper/runs":
            self._send_json({"runs": list_paper_trade_summaries()[:40]})
            return
        if path == "/api/simulation/plans":
            self._send_json({"plans": list_simulation_plan_summaries()[:60]})
            return
        if path == "/api/simulation/runs":
            self._send_json({"runs": list_simulation_summaries()[:60]})
            return
        if path == "/api/paper/latest":
            strategy_id = Path(parse_qs(parsed.query).get("strategy_id", [""])[0]).name
            if not strategy_id:
                self._send_json({"error": "missing_strategy_id"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                detail = load_latest_paper_snapshot(strategy_id)
            except FileNotFoundError:
                self._send_json({"error": "latest_snapshot_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail)
            return
        if path == "/api/simulation/latest":
            strategy_id = Path(parse_qs(parsed.query).get("strategy_id", [""])[0]).name
            if not strategy_id:
                self._send_json({"error": "missing_strategy_id"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                detail = load_latest_simulation_snapshot(strategy_id)
            except FileNotFoundError:
                self._send_json({"error": "latest_simulation_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail)
            return
        if path == "/api/paper/history":
            strategy_id = Path(parse_qs(parsed.query).get("strategy_id", [""])[0]).name
            if not strategy_id:
                self._send_json({"error": "missing_strategy_id"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                detail = load_paper_history_detail(strategy_id)
            except FileNotFoundError:
                self._send_json({"error": "history_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail)
            return
        if path == "/api/simulation/history":
            strategy_id = Path(parse_qs(parsed.query).get("strategy_id", [""])[0]).name
            run_id = Path(parse_qs(parsed.query).get("run_id", [""])[0]).name
            if not strategy_id:
                self._send_json({"error": "missing_strategy_id"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                detail = load_simulation_history_detail(strategy_id, run_id=run_id)
            except FileNotFoundError:
                self._send_json({"error": "simulation_history_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail)
            return
        if path == "/api/paper/lineage":
            strategy_id = Path(parse_qs(parsed.query).get("strategy_id", [""])[0]).name
            if not strategy_id:
                self._send_json({"error": "missing_strategy_id"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                detail = load_latest_paper_lineage(strategy_id)
            except FileNotFoundError:
                self._send_json({"error": "lineage_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail)
            return
        if path == "/api/simulation/lineage":
            strategy_id = Path(parse_qs(parsed.query).get("strategy_id", [""])[0]).name
            run_id = Path(parse_qs(parsed.query).get("run_id", [""])[0]).name
            if not strategy_id:
                self._send_json({"error": "missing_strategy_id"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                detail = load_simulation_lineage(strategy_id, run_id=run_id)
            except FileNotFoundError:
                self._send_json({"error": "simulation_lineage_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail)
            return
        if path.startswith("/api/runs/"):
            run_id = path.split("/api/runs/", 1)[1]
            try:
                detail = load_run_detail(run_id)
            except FileNotFoundError:
                self._send_json({"error": "run_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail)
            return
        if path.startswith("/api/paper/runs/"):
            run_id = path.split("/api/paper/runs/", 1)[1]
            try:
                detail = load_paper_trade_detail(run_id)
            except FileNotFoundError:
                self._send_json({"error": "paper_run_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail)
            return
        if path.startswith("/api/simulation/runs/"):
            run_id = path.split("/api/simulation/runs/", 1)[1]
            try:
                detail = load_simulation_detail(run_id)
            except FileNotFoundError:
                self._send_json({"error": "simulation_run_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail)
            return
        if path.startswith("/api/simulation/plans/"):
            plan_id = path.split("/api/simulation/plans/", 1)[1].split("/", 1)[0]
            try:
                detail = load_simulation_plan_detail(plan_id)
            except FileNotFoundError:
                self._send_json({"error": "simulation_plan_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(detail)
            return
        if path.startswith("/api/jobs/"):
            job_id = path.split("/api/jobs/", 1)[1]
            job = self.app.job_store.get(job_id)
            if job is None:
                self._send_json({"error": "job_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            payload = {"job": job}
            if job.get("status") == "completed":
                result_dir = job.get("result_dir", "")
                run_id = Path(result_dir).name
                try:
                    payload["run"] = load_run_detail(run_id)
                except FileNotFoundError:
                    pass
            self._send_json(payload)
            return
        if path.startswith("/api/paper/jobs/"):
            job_id = path.split("/api/paper/jobs/", 1)[1]
            job = self.app.job_store.get(job_id)
            if job is None or job.get("type") not in {"paper", "fill_scores"}:
                self._send_json({"error": "job_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            payload = {"job": job}
            if job.get("status") == "completed" and job.get("type") == "paper":
                result_dir = job.get("result_dir", "")
                run_id = Path(result_dir).name
                try:
                    payload["run"] = load_paper_trade_detail(run_id)
                except FileNotFoundError:
                    pass
            self._send_json(payload)
            return
        if path.startswith("/api/simulation/jobs/"):
            job_id = path.split("/api/simulation/jobs/", 1)[1]
            job = self.app.job_store.get(job_id)
            if job is None or job.get("type") not in {"simulation_plan", "simulation_execute", "fill_scores"}:
                self._send_json({"error": "job_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            payload = {"job": job}
            if job.get("status") == "completed" and job.get("type") == "simulation_execute":
                result_dir = job.get("result_dir", "")
                run_id = Path(result_dir).name
                try:
                    payload["run"] = load_simulation_detail(run_id)
                except FileNotFoundError:
                    pass
            if job.get("status") == "completed" and job.get("type") == "simulation_plan":
                result_dir = job.get("result_dir", "")
                plan_id = Path(result_dir).name
                try:
                    payload["plan"] = load_simulation_plan_detail(plan_id)
                except FileNotFoundError:
                    pass
            self._send_json(payload)
            return
        self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/simulation/plans":
            body = self._read_json_body()
            config_path = str(body.get("config_path", "")).strip()
            signal_date = str(body.get("signal_date", "")).strip() or str(body.get("trade_date", "")).strip()
            label = str(body.get("label", "")).strip()
            initial_cash = float(body.get("initial_cash", 1_000_000.0))
            if not config_path or not signal_date:
                self._send_json({"error": "missing_required_fields"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                job = self.app.submit_simulation_plan(
                    config_path=config_path,
                    signal_date=signal_date,
                    initial_cash=initial_cash,
                    label=label,
                )
            except (FileNotFoundError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"job": job}, status=HTTPStatus.ACCEPTED)
            return
        if parsed.path.startswith("/api/simulation/plans/") and parsed.path.endswith("/execute"):
            body = self._read_json_body()
            plan_id = Path(parsed.path.split("/api/simulation/plans/", 1)[1].rsplit("/execute", 1)[0]).name
            label = str(body.get("label", "")).strip()
            if not plan_id:
                self._send_json({"error": "missing_required_fields"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                job = self.app.submit_simulation_execute_plan(plan_id=plan_id, label=label)
            except (FileNotFoundError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"job": job}, status=HTTPStatus.ACCEPTED)
            return
        if parsed.path.startswith("/api/simulation/plans/") and parsed.path.endswith("/next"):
            plan_id = Path(parsed.path.split("/api/simulation/plans/", 1)[1].rsplit("/next", 1)[0]).name
            if not plan_id:
                self._send_json({"error": "missing_required_fields"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                job = self.app.submit_simulation_next_plan(plan_id=plan_id)
            except (FileNotFoundError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"job": job}, status=HTTPStatus.ACCEPTED)
            return
        if parsed.path == "/api/simulation/fill-scores":
            body = self._read_json_body()
            config_path = str(body.get("config_path", "")).strip()
            trade_date = str(body.get("trade_date", "")).strip()
            scores_path = str(body.get("scores_path", "")).strip()
            if not config_path or not trade_date:
                self._send_json({"error": "missing_required_fields"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                job = self.app.submit_simulation_fill_scores(
                    config_path=config_path,
                    trade_date=trade_date,
                    scores_path=scores_path,
                )
            except (FileNotFoundError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"job": job}, status=HTTPStatus.ACCEPTED)
            return
        if parsed.path == "/api/paper/generate":
            body = self._read_json_body()
            config_path = str(body.get("config_path", "")).strip()
            trade_date = str(body.get("trade_date", "")).strip()
            label = str(body.get("label", "")).strip()
            scores_path = str(body.get("scores_path", "")).strip()
            initial_cash = float(body.get("initial_cash", 1_000_000.0))
            if not config_path or not trade_date:
                self._send_json({"error": "missing_required_fields"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                job = self.app.submit_paper_trade(
                    config_path=config_path,
                    trade_date=trade_date,
                    initial_cash=initial_cash,
                    label=label,
                    scores_path=scores_path,
                )
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"job": job}, status=HTTPStatus.ACCEPTED)
            return
        if parsed.path == "/api/paper/fill-scores":
            body = self._read_json_body()
            config_path = str(body.get("config_path", "")).strip()
            trade_date = str(body.get("trade_date", "")).strip()
            scores_path = str(body.get("scores_path", "")).strip()
            if not config_path or not trade_date:
                self._send_json({"error": "missing_required_fields"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                job = self.app.submit_paper_fill_scores(
                    config_path=config_path,
                    trade_date=trade_date,
                    scores_path=scores_path,
                )
            except (FileNotFoundError, ValueError) as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"job": job}, status=HTTPStatus.ACCEPTED)
            return
        if parsed.path != "/api/backtests":
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        body = self._read_json_body()
        config_path = str(body.get("config_path", "")).strip()
        start_date = str(body.get("start_date", "")).strip()
        end_date = str(body.get("end_date", "")).strip()
        label = str(body.get("label", "")).strip()
        scores_path = str(body.get("scores_path", "")).strip()
        initial_cash = float(body.get("initial_cash", 1_000_000.0))
        if not config_path or not start_date or not end_date:
            self._send_json({"error": "missing_required_fields"}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            job = self.app.submit_backtest(
                config_path=config_path,
                start_date=start_date,
                end_date=end_date,
                initial_cash=initial_cash,
                label=label,
                scores_path=scores_path,
            )
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"job": job}, status=HTTPStatus.ACCEPTED)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists() or not path.is_file():
            self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
            return
        payload = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def create_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    app = BacktestWebApp()
    server = ThreadingHTTPServer((host, port), RequestHandler)
    server.app = app  # type: ignore[attr-defined]
    return server


def main() -> None:
    host = os.environ.get("ASHARE_WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("ASHARE_WEB_PORT", "8765"))
    log_path = configure_file_logging(level=logging.INFO)
    server = create_server(host=host, port=port)
    print(f"ASHARE_WEB http://{host}:{port}")
    LOGGER.info("web server start host=%s port=%s log_path=%s", host, port, log_path.as_posix())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        LOGGER.info("web server stop host=%s port=%s", host, port)
        server.server_close()


if __name__ == "__main__":
    main()
