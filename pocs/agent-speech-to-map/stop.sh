#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN="$ROOT/.run"

for name in backend frontend; do
  PIDFILE="$RUN/$name.pid"
  if [ -f "$PIDFILE" ]; then
    PID="$(cat "$PIDFILE")"
    pkill -P "$PID" 2>/dev/null || true
    kill "$PID" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "$name stopped"
  fi
done
