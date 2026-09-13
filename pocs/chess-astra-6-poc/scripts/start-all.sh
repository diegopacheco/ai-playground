#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
require bun
require lsof
if owned_pid >/dev/null && [ -n "$(port_pid)" ]; then
  printf 'Frontend already running on %s.\n' "$FRONTEND"
  exit 0
fi
[ -z "$(port_pid)" ] || fail "Port $FRONTEND is occupied by another process"
nohup bun "$ROOT/node_modules/vite/bin/vite.js" >"$RUN/logs/frontend.log" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$RUN/frontend.pid"
for ((attempt=0; attempt<30; attempt++)); do
  kill -0 "$pid" 2>/dev/null || fail "Frontend exited; read $RUN/logs/frontend.log"
  if curl -fsS "http://127.0.0.1:$FRONTEND/" >/dev/null 2>&1; then
    printf 'Frontend UP http://127.0.0.1:%s PID %s\n' "$FRONTEND" "$pid"
    exit 0
  fi
  sleep 1
done
kill "$pid" 2>/dev/null || true
rm -f "$RUN/frontend.pid"
fail "Frontend did not become ready within 30 seconds"
