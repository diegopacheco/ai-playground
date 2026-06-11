#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
PIDFILE=".burr-ui.pid"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi

source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Burr UI already running (pid $(cat "$PIDFILE"))."
else
  burr --no-open > burr-ui.log 2>&1 &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 30); do
    if curl -sf http://localhost:7241 >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  echo "Burr UI started at http://localhost:7241 (pid $(cat "$PIDFILE"))."
fi

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=sk-..."
  exit 1
fi

python chatbot.py
