#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if owned_pid >/dev/null && [ -n "$(port_pid)" ]; then
  printf 'WEB already running on %s\n' "$WEB"
  exit 0
fi
[ -z "$(port_pid)" ] || fail "Port $WEB is occupied by another process"
nohup node "$ROOT/server.mjs" >"$RUN/logs/web.log" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$RUN/web.pid"
for ((i=0;i<30;i++)); do
  kill -0 "$pid" 2>/dev/null || fail 'WEB exited; see .run/logs/web.log'
  if curl --silent --fail "http://localhost:$WEB/" >/dev/null; then
    printf 'WEB ready at http://localhost:%s\n' "$WEB"
    exit 0
  fi
  sleep 1
done
kill "$pid" 2>/dev/null || true
rm -f "$RUN/web.pid"
fail 'WEB did not become ready within 30 seconds'
