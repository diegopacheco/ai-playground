#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
if owned && healthy; then printf 'Brisa is already running on port %s\n' "$WEB"; exit 0; fi
if lsof -ti "tcp:$WEB" -sTCP:LISTEN >/dev/null 2>&1; then fail "Port $WEB is in use. Stop its owner before starting Brisa."; fi
[ -f node_modules/three/build/three.module.js ] || fail 'Run ./scripts/setup.sh first'
nohup node "$ROOT/server.mjs" >"$RUN/logs/web.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN/web.pid"
for ((attempt=0; attempt<30; attempt++)); do
  if owned && healthy; then printf 'Brisa is running at http://localhost:%s\n' "$WEB"; exit 0; fi
  if ! owned; then cat "$RUN/logs/web.log" >&2; fail 'Brisa exited during startup'; fi
  sleep 1
done
"$ROOT/scripts/stop-all.sh"
fail 'Brisa did not become ready within 30 seconds'
