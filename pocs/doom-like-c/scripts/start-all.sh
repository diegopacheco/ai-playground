#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"

if is_running; then
  echo "$GAME_NAME already running with pid $(game_pid)"
  exit 0
fi

ensure_run_dirs
make all >/dev/null || fail "build failed, run make all to see the errors"

nohup "$GAME_BIN" >"$GAME_LOG_FILE" 2>&1 &
echo "$!" >"$GAME_PID_FILE"

wait_until 5 is_running || fail "$GAME_NAME did not start, see $GAME_LOG_FILE"
sleep 1
is_running || fail "$GAME_NAME exited right after start, see $GAME_LOG_FILE"

echo "$GAME_NAME running with pid $(game_pid)"
echo "window: DUCK DOOM (native SDL2 window, no network port)"
echo "log: $GAME_LOG_FILE"
