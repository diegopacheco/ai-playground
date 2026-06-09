#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN="$ROOT/.run"
mkdir -p "$RUN"

if [ -z "$OPENAI_API_KEY" ]; then
  echo "OPENAI_API_KEY is not set"
  exit 1
fi

free_port() {
  local p="$1"
  while lsof -iTCP:"$p" -sTCP:LISTEN -P >/dev/null 2>&1; do
    p=$((p + 1))
  done
  echo "$p"
}

wait_for() {
  local url="$1" needle="$2" name="$3" log="$4" n=0
  until curl -sf "$url" 2>/dev/null | grep -q "$needle"; do
    n=$((n + 1))
    if [ "$n" -gt 60 ]; then
      echo "$name did not become ready, see $log"
      tail -n 20 "$log"
      exit 1
    fi
    sleep 1
  done
}

PORT="$(free_port "${PORT:-8080}")"
FRONTEND_PORT="$(free_port "${FRONTEND_PORT:-5173}")"

cd "$ROOT/backend"
go build -o "$RUN/backend-bin" .
PORT="$PORT" OPENAI_API_KEY="$OPENAI_API_KEY" "$RUN/backend-bin" >"$RUN/backend.log" 2>&1 &
echo $! >"$RUN/backend.pid"

cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
  npm install >"$RUN/frontend-install.log" 2>&1
fi
VITE_API_BASE="http://localhost:$PORT" npm run dev -- --port "$FRONTEND_PORT" --strictPort >"$RUN/frontend.log" 2>&1 &
echo $! >"$RUN/frontend.pid"

wait_for "http://localhost:$PORT/api/health" '"status":"ok"' "backend" "$RUN/backend.log"
echo "backend ready on http://localhost:$PORT"

wait_for "http://localhost:$FRONTEND_PORT" "Speech to Map" "frontend" "$RUN/frontend.log"
echo "frontend ready on http://localhost:$FRONTEND_PORT"

echo "open http://localhost:$FRONTEND_PORT"
