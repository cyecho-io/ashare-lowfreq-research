#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[bootstrap] project root: $ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[bootstrap] python3 not found"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "[bootstrap] creating .venv"
  python3 -m venv .venv
fi

echo "[bootstrap] installing project"
"$ROOT_DIR/.venv/bin/python" -m pip install --upgrade pip
"$ROOT_DIR/.venv/bin/python" -m pip install -e ".[dev]"

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  echo "[bootstrap] creating .env from template"
  cp .env.example .env
fi

if [ -d "storage/demo" ]; then
  echo "[bootstrap] demo storage directory detected at storage/demo"
  echo "[bootstrap] if tracked demo parquet files exist, point commands to --storage-root storage/demo"
else
  echo "[bootstrap] storage/demo is missing"
fi

cat <<'EOF'

Next steps:

1. Activate the environment:
   source .venv/bin/activate

2. If you have real data access, fill TUSHARE_TOKEN in .env and run sync commands.

3. If you only want to evaluate the public demo flow, prepare or download a small dataset under storage/demo/.

4. Start the web console:
   ashare-backtest-web

5. Run tests:
   python3 -m pytest

EOF
