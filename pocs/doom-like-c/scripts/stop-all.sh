#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
cd "$ROOT"

if is_stopped; then
  rm -f "$GAME_PID_FILE"
  echo "$GAME_NAME already stopped"
  exit 0
fi

pid="$(game_pid)"
kill "$pid" 2>/dev/null || true
if ! wait_until 10 is_stopped; then
  kill -9 "$pid" 2>/dev/null || true
  wait_until 5 is_stopped || fail "could not stop $GAME_NAME pid $pid"
fi
rm -f "$GAME_PID_FILE"
echo "$GAME_NAME stopped"
