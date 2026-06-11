#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PIDFILE=".burr-ui.pid"

if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE")"
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "Stopped Burr UI (pid $PID)."
  else
    echo "Burr UI was not running."
  fi
  rm -f "$PIDFILE"
else
  echo "No Burr UI pid file found; nothing to stop."
fi
