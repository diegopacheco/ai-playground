#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
PORT="${PORT:-3000}"
STOPPED=0
if [ -f server.pid ]; then
  if kill "$(cat server.pid)" 2>/dev/null; then
    STOPPED=1
  fi
  rm -f server.pid
fi
PIDS="$(lsof -ti ":${PORT}" 2>/dev/null || true)"
if [ -n "$PIDS" ]; then
  echo "$PIDS" | xargs kill 2>/dev/null || true
  STOPPED=1
fi
if [ "$STOPPED" = "1" ]; then
  echo "stopped"
else
  echo "not running"
fi
