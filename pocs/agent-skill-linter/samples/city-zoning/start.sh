#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
SERVER_PORT="${SERVER_PORT:-4000}"
WEB_PORT="${WEB_PORT:-5180}"

for port in "$SERVER_PORT" "$WEB_PORT"; do
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN > /dev/null 2>&1; then
    echo "port $port is already in use; free it or set SERVER_PORT / WEB_PORT and retry"
    exit 1
  fi
done

PORT="$SERVER_PORT" node "$ROOT/server/src/index.js" > "$ROOT/server.log" 2>&1 &
echo $! > "$ROOT/server.pid"

if [ ! -d "$ROOT/web/node_modules" ]; then
  (cd "$ROOT/web" && bun install)
fi
(cd "$ROOT/web" && bun run dev -- --port "$WEB_PORT" --strictPort) > "$ROOT/web.log" 2>&1 &
echo $! > "$ROOT/web.pid"

for i in $(seq 1 60); do
  if curl -sf "http://localhost:$SERVER_PORT/api/health" > /dev/null; then
    echo "city-zoning-server up on http://localhost:$SERVER_PORT"
    break
  fi
  sleep 1
done

for i in $(seq 1 60); do
  if curl -sf "http://localhost:$WEB_PORT" > /dev/null; then
    echo "city-zoning-web up on http://localhost:$WEB_PORT"
    exit 0
  fi
  sleep 1
done
echo "web failed to start, see web.log"
exit 1
