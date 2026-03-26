from __future__ import annotations

import csv
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from ashare_backtest.cli.main import run_model_backtest
from ashare_backtest.cli.research_config import ResearchRunConfig, load_research_config
from ashare_backtest.research import StrategyStateConfig, generate_strategy_state

REPO_ROOT = Path(__file__).resolve().parents[3]
STATIC_ROOT = Path(__file__).resolve().parent / "static"
RESULTS_ROOT = REPO_ROOT / "results"
WEB_RUNS_ROOT = RESULTS_ROOT / "web_runs"
PAPER_RUNS_ROOT = RESULTS_ROOT / "paper_runs"
PAPER_BACKTEST_ROOT = RESULTS_ROOT / "paper_backtests"
CONFIG_ROOT = REPO_ROOT / "configs"
BARS_PATH = REPO_ROOT / "storage" / "parquet" / "bars" / "daily.parquet"
BENCHMARK_PATH = REPO_ROOT / "storage" / "parquet" / "benchmarks" / "000300.SH.parquet"


@dataclass(frozen=True)
class StrategyPreset:
    id: str
    name: str
    config_path: str
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


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


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


def list_score_parquet_files(models_root: Path = REPO_ROOT / "research" / "models") -> list[dict[str, str]]:
    if not models_root.exists():
        return []
    files: list[dict[str, str]] = []
    for path in models_root.rglob("*.parquet"):
        if not path.is_file() or "scores" not in path.name:
            continue
        start_date, end_date = _score_date_range(path)
        files.append(
            {
                "path": _display_path(path),
                "start_date": start_date,
                "end_date": end_date,
            }
        )
    deduped: dict[str, dict[str, str]] = {}
    for item in files:
        deduped[item["path"]] = item
    return [deduped[path] for path in sorted(deduped)]


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


def _preset_from_config(path: Path, config: ResearchRunConfig) -> StrategyPreset:
    display_name = path.stem.replace("_", " ")
    resolved_paper_scores_path, manifest, paper_source_kind = _resolve_paper_scores_path(path.stem, config.score_output_path)
    paper_score_start_date, paper_score_end_date = _score_date_range(_resolve_repo_path(resolved_paper_scores_path))
    return StrategyPreset(
        id=path.stem,
        name=display_name,
        config_path=path.relative_to(REPO_ROOT).as_posix(),
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
    strategy_state_path = target / "strategy_state_latest.json"
    strategy_state = _read_json(strategy_state_path) if strategy_state_path.exists() else None
    return {
        "id": safe_run_id,
        "name": safe_run_id,
        "result_dir": _display_path(target),
        "summary": summary,
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
) -> StrategyStateConfig:
    return StrategyStateConfig(
        scores_path=scores_path,
        storage_root=config.storage_root,
        output_path=output_path,
        trade_date=trade_date,
        mode="historical",
        top_k=config.top_k,
        rebalance_every=config.rebalance_every,
        lookback_window=config.lookback_window,
        min_hold_bars=config.min_hold_bars,
        keep_buffer=config.keep_buffer,
        min_turnover_names=config.min_turnover_names,
        min_daily_amount=config.min_daily_amount,
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

    def submit_paper_backtest(self, config_path: str, start_date: str, initial_cash: float, label: str) -> dict[str, Any]:
        absolute_config = (self.repo_root / config_path).resolve()
        if not absolute_config.exists():
            raise FileNotFoundError(f"config not found: {config_path}")
        config = load_research_config(absolute_config)
        paper_scores_path, manifest, paper_source_kind = _resolve_paper_scores_path(absolute_config.stem, config.score_output_path)
        _, end_date = _score_date_range(_resolve_repo_path(paper_scores_path))
        if not start_date or not end_date:
            raise FileNotFoundError(f"paper score range not found: {absolute_config.stem}")
        if start_date > end_date:
            raise ValueError("paper backtest start_date must be on or before latest score date")

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{timestamp}-{_slugify(label or f'{absolute_config.stem}-paper-backtest')}"
        output_dir = (PAPER_BACKTEST_ROOT / run_name).relative_to(self.repo_root).as_posix()
        job_id = run_name
        job_payload = {
            "id": job_id,
            "type": "paper_backtest",
            "status": "queued",
            "config_path": config_path,
            "start_date": start_date,
            "end_date": end_date,
            "initial_cash": initial_cash,
            "scores_path": paper_scores_path,
            "paper_source_kind": paper_source_kind,
            "result_dir": output_dir,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "error": "",
        }
        self.job_store.create(job_id, job_payload)
        args = _build_run_args(config, start_date, end_date, initial_cash, output_dir, scores_path=paper_scores_path)
        self.executor.submit(
            self._run_paper_backtest_job,
            job_id,
            args,
            config,
            initial_cash,
            paper_scores_path,
        )
        return job_payload

    def submit_paper_trade(self, config_path: str, trade_date: str, initial_cash: float, label: str) -> dict[str, Any]:
        absolute_config = (self.repo_root / config_path).resolve()
        if not absolute_config.exists():
            raise FileNotFoundError(f"config not found: {config_path}")
        config = load_research_config(absolute_config)
        paper_scores_path, manifest, paper_source_kind = _resolve_paper_scores_path(absolute_config.stem, config.score_output_path)
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

    def _run_paper_backtest_job(
        self,
        job_id: str,
        args: dict[str, Any],
        config: ResearchRunConfig,
        initial_cash: float,
        scores_path: str,
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
            generate_strategy_state(
                _build_strategy_state_args(
                    config=config,
                    scores_path=scores_path,
                    trade_date=trade_date,
                    initial_cash=initial_cash,
                    output_path=(output_dir / "strategy_state.json").as_posix(),
                )
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


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "AshareBacktestWeb/0.1"

    @property
    def app(self) -> BacktestWebApp:
        return self.server.app  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._serve_file(STATIC_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if path == "/paper":
            self._serve_file(STATIC_ROOT / "paper.html", "text/html; charset=utf-8")
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
            self._send_json({"strategies": presets, "score_files": list_score_parquet_files()})
            return
        if path == "/api/paper/strategies":
            presets = [preset.__dict__ for preset in list_strategy_presets()]
            self._send_json({"strategies": presets})
            return
        if path == "/api/runs":
            self._send_json({"runs": list_run_summaries()[:40]})
            return
        if path == "/api/paper/runs":
            self._send_json({"runs": list_paper_trade_summaries()[:40]})
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
            if job is None or job.get("type") not in {"paper", "paper_backtest"}:
                self._send_json({"error": "job_not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            payload = {"job": job}
            if job.get("status") == "completed":
                result_dir = job.get("result_dir", "")
                run_id = Path(result_dir).name
                if job.get("type") == "paper":
                    try:
                        payload["run"] = load_paper_trade_detail(run_id)
                    except FileNotFoundError:
                        pass
                else:
                    try:
                        payload["run"] = load_run_detail(run_id)
                    except FileNotFoundError:
                        pass
            self._send_json(payload)
            return
        self._send_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/paper/generate":
            body = self._read_json_body()
            config_path = str(body.get("config_path", "")).strip()
            trade_date = str(body.get("trade_date", "")).strip()
            label = str(body.get("label", "")).strip()
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
                )
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            self._send_json({"job": job}, status=HTTPStatus.ACCEPTED)
            return
        if parsed.path == "/api/paper/backtest":
            body = self._read_json_body()
            config_path = str(body.get("config_path", "")).strip()
            start_date = str(body.get("start_date", "")).strip()
            label = str(body.get("label", "")).strip()
            initial_cash = float(body.get("initial_cash", 1_000_000.0))
            if not config_path or not start_date:
                self._send_json({"error": "missing_required_fields"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                job = self.app.submit_paper_backtest(
                    config_path=config_path,
                    start_date=start_date,
                    initial_cash=initial_cash,
                    label=label,
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
    server = create_server(host=host, port=port)
    print(f"ASHARE_WEB http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
