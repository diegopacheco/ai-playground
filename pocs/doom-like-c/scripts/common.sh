#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$ROOT/.run"
LOG_DIR="$RUN_DIR/logs"
GAME_NAME="duckdoom"
GAME_BIN="$ROOT/build/duckdoom"
GAME_PID_FILE="$RUN_DIR/duckdoom.pid"
GAME_LOG_FILE="$LOG_DIR/duckdoom.log"

fail() {
  echo "error: $*" >&2
  exit 1
}

ensure_run_dirs() {
  mkdir -p "$LOG_DIR"
}

game_pid() {
  if [ -f "$GAME_PID_FILE" ]; then
    cat "$GAME_PID_FILE"
  fi
}

is_running() {
  local pid
  pid="$(game_pid)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

is_stopped() {
  ! is_running
}

wait_until() {
  local tries="$1"
  shift
  local attempt=0
  while [ "$attempt" -lt "$tries" ]; do
    if "$@"; then
      return 0
    fi
    sleep 1
    attempt=$((attempt + 1))
  done
  return 1
}
