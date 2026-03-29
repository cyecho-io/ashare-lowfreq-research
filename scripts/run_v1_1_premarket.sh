#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="$ROOT_DIR/.venv/bin/ashare-backtest"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
ENV_FILE="$ROOT_DIR/.env"

SQLITE_PATH="storage/source/ashare_arena_sync.db"
STORAGE_ROOT="storage"
FACTOR_OUTPUT_DIR="research/factors/research_industry_v4_v1_1/tradable_core/start_2024-01-02"
FACTOR_OUTPUT=""
SIGNAL_DATE=""
TRADE_DATE=""
FORCE_SYNC=0
FORCE_IMPORT=0
FORCE_FACTORS=0
FORCE_INFERENCE=0
FORCE_PREMARKET=0
CALENDAR_BUFFER_DAYS=7

usage() {
  cat <<'EOF'
Usage:
  scripts/run_v1_1_premarket.sh [options]

Options:
  --signal-date YYYY-MM-DD   指定信号日 T。默认在同步后自动取 SQLite 最新 trade_date
  --trade-date YYYY-MM-DD    指定执行日 T+1。默认自动取信号日后的下一个开市日
  --sqlite-path PATH         SQLite 源库路径，默认 storage/source/ashare_arena_sync.db
  --storage-root PATH        Parquet 存储根目录，默认 storage
  --force-sync               强制执行 Tushare -> SQLite 同步
  --force-import             强制执行 SQLite -> Parquet 刷新
  --force-factors            强制重建因子面板
  --force-inference          强制重跑信号日 walk_forward 打分
  --force-premarket          强制重建盘前参考文件
  -h, --help                 查看帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --signal-date)
      SIGNAL_DATE="${2:-}"
      shift 2
      ;;
    --trade-date)
      TRADE_DATE="${2:-}"
      shift 2
      ;;
    --sqlite-path)
      SQLITE_PATH="${2:-}"
      shift 2
      ;;
    --storage-root)
      STORAGE_ROOT="${2:-}"
      shift 2
      ;;
    --force-sync)
      FORCE_SYNC=1
      shift
      ;;
    --force-import)
      FORCE_IMPORT=1
      shift
      ;;
    --force-factors)
      FORCE_FACTORS=1
      shift
      ;;
    --force-inference)
      FORCE_INFERENCE=1
      shift
      ;;
    --force-premarket)
      FORCE_PREMARKET=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -x "$CLI" ]]; then
  echo "Missing CLI: $CLI" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"

normalize_date() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
import pandas as pd
print(pd.Timestamp(sys.argv[1]).date().isoformat())
PY
}

compact_date() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
import pandas as pd
print(pd.Timestamp(sys.argv[1]).strftime("%Y%m%d"))
PY
}

plus_days_compact() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import sys
import pandas as pd
base = pd.Timestamp(sys.argv[1])
days = int(sys.argv[2])
print((base + pd.Timedelta(days=days)).strftime("%Y%m%d"))
PY
}

sqlite_max_date() {
  "$PYTHON_BIN" - "$SQLITE_PATH" <<'PY'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
row = conn.execute("select max(trade_date) from equity_daily_bars").fetchone()
print("" if row is None or row[0] is None else row[0])
PY
}

sqlite_calendar_has_open_date() {
  "$PYTHON_BIN" - "$SQLITE_PATH" "$1" <<'PY'
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1])
row = conn.execute(
    "select 1 from trading_calendar where trade_date = ? and is_open = 1 limit 1",
    (sys.argv[2],),
).fetchone()
print("1" if row else "0")
PY
}

parquet_max_date() {
  "$PYTHON_BIN" - "$STORAGE_ROOT" <<'PY'
import sys
from pathlib import Path
import pandas as pd
path = Path(sys.argv[1]) / "parquet" / "bars" / "daily.parquet"
if not path.exists():
    print("")
else:
    frame = pd.read_parquet(path, columns=["trade_date"])
    print("" if frame.empty else frame["trade_date"].max().date().isoformat())
PY
}

parquet_calendar_has_open_date() {
  "$PYTHON_BIN" - "$STORAGE_ROOT" "$1" <<'PY'
import sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1])
target = sys.argv[2]
path = root / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
if not path.exists():
    print("0")
else:
    frame = pd.read_parquet(path, columns=["trade_date", "is_open"])
    matched = frame.loc[(frame["trade_date"] == pd.Timestamp(target)) & (frame["is_open"])]
    print("1" if not matched.empty else "0")
PY
}

factor_max_date() {
  "$PYTHON_BIN" - "$FACTOR_OUTPUT" <<'PY'
import sys
from pathlib import Path
import pandas as pd
path = Path(sys.argv[1])
if not path.exists():
    print("")
else:
    frame = pd.read_parquet(path, columns=["trade_date"])
    print("" if frame.empty else frame["trade_date"].max().date().isoformat())
PY
}

next_open_trade_date() {
  "$PYTHON_BIN" - "$STORAGE_ROOT" "$1" <<'PY'
import sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1])
signal_date = pd.Timestamp(sys.argv[2])
path = root / "parquet" / "calendar" / "ashare_trading_calendar.parquet"
frame = pd.read_parquet(path, columns=["trade_date", "is_open"]).sort_values("trade_date")
future = frame.loc[(frame["is_open"]) & (frame["trade_date"] > signal_date), "trade_date"]
if future.empty:
    raise SystemExit("No next open trade date found in calendar parquet.")
print(future.iloc[0].date().isoformat())
PY
}

scores_max_date() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
from pathlib import Path
import pandas as pd
path = Path(sys.argv[1])
if not path.exists():
    print("")
else:
    frame = pd.read_parquet(path, columns=["trade_date"])
    print("" if frame.empty else pd.to_datetime(frame["trade_date"]).max().date().isoformat())
PY
}

premarket_execution_date() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("")
else:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(payload.get("summary", {}).get("execution_date", ""))
PY
}

strategy_state_execution_date() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("")
else:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(payload.get("summary", {}).get("execution_date", ""))
PY
}

show_step() {
  printf '\n[%s] %s\n' "$1" "$2"
}

run_cmd() {
  echo "+ $*"
  "$@"
}

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

show_step "Init" "Project root: $ROOT_DIR"
echo "SQLite path: $SQLITE_PATH"
echo "Storage root: $STORAGE_ROOT"

if [[ -n "$SIGNAL_DATE" ]]; then
  SIGNAL_DATE="$(normalize_date "$SIGNAL_DATE")"
  echo "Requested signal date: $SIGNAL_DATE"
fi

if [[ -n "$TRADE_DATE" ]]; then
  TRADE_DATE="$(normalize_date "$TRADE_DATE")"
  echo "Requested trade date: $TRADE_DATE"
fi

CURRENT_SQLITE_MAX="$(sqlite_max_date)"
show_step "Check" "Current SQLite max trade_date: ${CURRENT_SQLITE_MAX:-<empty>}"
SQLITE_HAS_TRADE_DATE=0
if [[ -n "$TRADE_DATE" ]]; then
  SQLITE_HAS_TRADE_DATE="$(sqlite_calendar_has_open_date "$TRADE_DATE")"
  echo "SQLite calendar contains open trade date $TRADE_DATE: $SQLITE_HAS_TRADE_DATE"
fi

NEED_SYNC=0
if [[ "$FORCE_SYNC" -eq 1 ]]; then
  NEED_SYNC=1
elif [[ -n "$SIGNAL_DATE" ]]; then
  if [[ -z "$CURRENT_SQLITE_MAX" || "$CURRENT_SQLITE_MAX" < "$SIGNAL_DATE" ]]; then
    NEED_SYNC=1
  fi
fi
if [[ "$NEED_SYNC" -eq 0 && -n "$TRADE_DATE" && "$SQLITE_HAS_TRADE_DATE" != "1" ]]; then
  NEED_SYNC=1
fi

if [[ "$NEED_SYNC" -eq 1 ]]; then
  show_step "Sync" "Refreshing SQLite from Tushare"
  if [[ -z "${TUSHARE_TOKEN:-}" ]]; then
    echo "TUSHARE_TOKEN is required when sync is needed." >&2
    exit 1
  fi
  SYNC_ARGS=(sync-tushare-sqlite --sqlite-path "$SQLITE_PATH")
  if [[ -n "$SIGNAL_DATE" ]]; then
    if [[ -n "$TRADE_DATE" ]]; then
      SYNC_ARGS+=(--end "$(compact_date "$TRADE_DATE")")
    else
      SYNC_ARGS+=(--end "$(plus_days_compact "$SIGNAL_DATE" "$CALENDAR_BUFFER_DAYS")")
    fi
  fi
  run_cmd "$CLI" "${SYNC_ARGS[@]}"
  CURRENT_SQLITE_MAX="$(sqlite_max_date)"
  echo "SQLite max trade_date after sync: $CURRENT_SQLITE_MAX"
else
  show_step "Skip" "SQLite sync not needed"
fi

if [[ -z "$SIGNAL_DATE" ]]; then
  SIGNAL_DATE="$CURRENT_SQLITE_MAX"
  echo "Auto-selected signal date: $SIGNAL_DATE"
fi

CURRENT_PARQUET_MAX="$(parquet_max_date)"
show_step "Check" "Current Parquet max trade_date: ${CURRENT_PARQUET_MAX:-<empty>}"
PARQUET_HAS_TRADE_DATE=0
if [[ -n "$TRADE_DATE" ]]; then
  PARQUET_HAS_TRADE_DATE="$(parquet_calendar_has_open_date "$TRADE_DATE")"
  echo "Parquet calendar contains open trade date $TRADE_DATE: $PARQUET_HAS_TRADE_DATE"
fi

if [[ "$FORCE_IMPORT" -eq 1 || -z "$CURRENT_PARQUET_MAX" || "$CURRENT_PARQUET_MAX" < "$SIGNAL_DATE" || ( -n "$TRADE_DATE" && "$PARQUET_HAS_TRADE_DATE" != "1" ) ]]; then
  show_step "Import" "Refreshing Parquet snapshot from SQLite"
  run_cmd "$CLI" import-sqlite "$SQLITE_PATH" --storage-root "$STORAGE_ROOT"
  CURRENT_PARQUET_MAX="$(parquet_max_date)"
  echo "Parquet max trade_date after import: $CURRENT_PARQUET_MAX"
  if [[ -n "$TRADE_DATE" ]]; then
    PARQUET_HAS_TRADE_DATE="$(parquet_calendar_has_open_date "$TRADE_DATE")"
    echo "Parquet calendar contains open trade date $TRADE_DATE after import: $PARQUET_HAS_TRADE_DATE"
  fi
else
  show_step "Skip" "Parquet import not needed"
fi

show_step "Universe" "Available universes"
run_cmd "$CLI" list-universes --storage-root "$STORAGE_ROOT"

if [[ -z "$TRADE_DATE" ]]; then
  TRADE_DATE="$(next_open_trade_date "$SIGNAL_DATE")"
  echo "Auto-selected trade date: $TRADE_DATE"
fi

FACTOR_OUTPUT="${FACTOR_OUTPUT_DIR}/asof_${SIGNAL_DATE}.parquet"

CURRENT_FACTOR_MAX="$(factor_max_date)"
show_step "Check" "Current factor panel max trade_date: ${CURRENT_FACTOR_MAX:-<empty>}"

if [[ "$FORCE_FACTORS" -eq 1 || -z "$CURRENT_FACTOR_MAX" || "$CURRENT_FACTOR_MAX" != "$SIGNAL_DATE" ]]; then
  show_step "Factors" "Building signal-date factor snapshot"
  run_cmd "$CLI" build-factors \
    --storage-root "$STORAGE_ROOT" \
    --universe-name tradable_core \
    --output-path "$FACTOR_OUTPUT" \
    --start-date 2024-01-02 \
    --as-of-date "$SIGNAL_DATE"
  CURRENT_FACTOR_MAX="$(factor_max_date)"
  echo "Factor panel max trade_date after build: $CURRENT_FACTOR_MAX"
else
  show_step "Skip" "Factor panel already up to date"
fi

SCORES_PATH="research/models/walk_forward_scores_industry_v4_v1_1_${SIGNAL_DATE}.parquet"
METRICS_PATH="research/models/walk_forward_metrics_industry_v4_v1_1_${SIGNAL_DATE}.json"
CURRENT_SCORES_MAX="$(scores_max_date "$SCORES_PATH")"
show_step "Check" "Current walk-forward score file: $SCORES_PATH max trade_date=${CURRENT_SCORES_MAX:-<empty>}"

if [[ "$FORCE_INFERENCE" -eq 1 || ! -f "$SCORES_PATH" || "$CURRENT_SCORES_MAX" != "$SIGNAL_DATE" ]]; then
  show_step "Scores" "Running walk-forward as-of-date scoring"
  run_cmd "$CLI" train-lgbm-walk-forward-as-of-date \
    --factor-panel-path "$FACTOR_OUTPUT" \
    --label-column industry_excess_fwd_return_5 \
    --train-window-months 12 \
    --as-of-date "$SIGNAL_DATE" \
    --output-scores-path "$SCORES_PATH" \
    --output-metrics-path "$METRICS_PATH"
  CURRENT_SCORES_MAX="$(scores_max_date "$SCORES_PATH")"
  echo "Score file max trade_date after scoring: $CURRENT_SCORES_MAX"
else
  show_step "Skip" "Walk-forward score file already up to date"
fi

PREMARKET_PATH="research/models/premarket_reference_industry_v4_v1_1_${TRADE_DATE}.json"
STRATEGY_STATE_PATH="research/models/strategy_state_v1_1.json"
LATEST_DIR="research/models/latest/research_industry_v4_v1_1"
LATEST_SCORES_PATH="${LATEST_DIR}/scores.parquet"
LATEST_METRICS_PATH="${LATEST_DIR}/metrics.json"
LATEST_STATE_PATH="${LATEST_DIR}/strategy_state.json"
LATEST_TRADES_PATH="${LATEST_DIR}/trades.csv"
LATEST_DECISION_LOG_PATH="${LATEST_DIR}/decision_log.csv"
LATEST_PREMARKET_PATH="${LATEST_DIR}/premarket_reference.json"
LATEST_MANIFEST_PATH="${LATEST_DIR}/manifest.json"
CURRENT_EXEC_DATE="$(strategy_state_execution_date "$STRATEGY_STATE_PATH")"
show_step "Check" "Current strategy state: $STRATEGY_STATE_PATH execution_date=${CURRENT_EXEC_DATE:-<empty>}"

if [[ "$FORCE_PREMARKET" -eq 1 || ! -f "$STRATEGY_STATE_PATH" || "$CURRENT_EXEC_DATE" != "$TRADE_DATE" ]]; then
  STATE_MODE="continue"
  if [[ ! -f "$STRATEGY_STATE_PATH" ]]; then
    STATE_MODE="initial_entry"
  fi
  show_step "State" "Generating strategy state (mode=$STATE_MODE)"
  STATE_ARGS=(generate-strategy-state
    --scores-path "$SCORES_PATH"
    --storage-root "$STORAGE_ROOT"
    --trade-date "$TRADE_DATE"
    --mode "$STATE_MODE"
    --top-k 6
    --rebalance-every 5
    --lookback-window 20
    --min-hold-bars 8
    --keep-buffer 2
    --min-turnover-names 3
    --max-names-per-industry 2
    --output-path "$STRATEGY_STATE_PATH")
  if [[ -f "$STRATEGY_STATE_PATH" ]]; then
    STATE_ARGS+=(--previous-state-path "$STRATEGY_STATE_PATH")
  fi
  run_cmd "$CLI" "${STATE_ARGS[@]}"
else
  show_step "Skip" "Strategy state already up to date"
fi

show_step "Done" "Strategy state ready"
echo "Signal date: $SIGNAL_DATE"
echo "Trade date: $TRADE_DATE"
echo "Scores path: $SCORES_PATH"
echo "Strategy state path: $STRATEGY_STATE_PATH"

show_step "Latest" "Refreshing latest strategy artifacts"
mkdir -p "$LATEST_DIR"
cp "$SCORES_PATH" "$LATEST_SCORES_PATH"
cp "$METRICS_PATH" "$LATEST_METRICS_PATH"
run_cmd "$CLI" generate-strategy-state \
  --scores-path "$LATEST_SCORES_PATH" \
  --storage-root "$STORAGE_ROOT" \
  --trade-date "$TRADE_DATE" \
  --mode historical \
  --top-k 6 \
  --rebalance-every 5 \
  --lookback-window 20 \
  --min-hold-bars 8 \
  --keep-buffer 2 \
  --min-turnover-names 3 \
  --max-names-per-industry 2 \
  --output-path "$LATEST_STATE_PATH"
if [[ -f "$PREMARKET_PATH" ]]; then
  cp "$PREMARKET_PATH" "$LATEST_PREMARKET_PATH"
fi

"$PYTHON_BIN" - "$LATEST_MANIFEST_PATH" "$SCORES_PATH" "$METRICS_PATH" "$STRATEGY_STATE_PATH" "$PREMARKET_PATH" "$SIGNAL_DATE" "$TRADE_DATE" "$LATEST_TRADES_PATH" "$LATEST_DECISION_LOG_PATH" <<'PY'
import json, sys
from datetime import datetime
from pathlib import Path

manifest_path = Path(sys.argv[1])
scores_path = sys.argv[2]
metrics_path = sys.argv[3]
strategy_state_path = sys.argv[4]
premarket_path = sys.argv[5]
signal_date = sys.argv[6]
execution_date = sys.argv[7]
latest_trades_path = sys.argv[8]
latest_decision_log_path = sys.argv[9]

payload = {
    "strategy_id": "research_industry_v4_v1_1",
    "signal_date": signal_date,
    "execution_date": execution_date,
    "scores_path": "research/models/latest/research_industry_v4_v1_1/scores.parquet",
    "metrics_path": "research/models/latest/research_industry_v4_v1_1/metrics.json",
    "strategy_state_path": "research/models/latest/research_industry_v4_v1_1/strategy_state.json",
    "trades_path": (
        "research/models/latest/research_industry_v4_v1_1/trades.csv"
        if Path(latest_trades_path).exists()
        else ""
    ),
    "decision_log_path": (
        "research/models/latest/research_industry_v4_v1_1/decision_log.csv"
        if Path(latest_decision_log_path).exists()
        else ""
    ),
    "premarket_reference_path": (
        "research/models/latest/research_industry_v4_v1_1/premarket_reference.json"
        if Path(premarket_path).exists()
        else ""
    ),
    "source_scores_path": scores_path,
    "source_metrics_path": metrics_path,
    "source_strategy_state_path": strategy_state_path,
    "source_premarket_reference_path": premarket_path if Path(premarket_path).exists() else "",
    "generated_at": datetime.now().isoformat(timespec="seconds"),
}
manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY

"$PYTHON_BIN" - "$STRATEGY_STATE_PATH" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
summary = payload.get("summary", {})
plan = payload.get("plan", {})
selected = plan.get("selected_symbols", [])
actions = plan.get("actions", [])
print("Summary:", summary)
print("Selected symbols:", ", ".join(selected[:20]) if selected else "<empty>")
print("Action count:", len(actions))
PY
