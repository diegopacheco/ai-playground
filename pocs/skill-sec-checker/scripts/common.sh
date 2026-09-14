#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_DIR="$ROOT/.run"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"
. "$ROOT/scripts/ports.env"

port_up() {
  lsof -iTCP:"$1" -sTCP:LISTEN -n -P >/dev/null 2>&1
}

wait_port() {
  local port="$1" tries=0
  while ! port_up "$port"; do
    tries=$((tries + 1))
    if [ "$tries" -ge 30 ]; then
      echo "port $port did not open in 30 seconds" >&2
      return 1
    fi
    sleep 1
  done
}

pid_of() {
  if [ -f "$RUN_DIR/$1.pid" ]; then cat "$RUN_DIR/$1.pid"; fi
}
