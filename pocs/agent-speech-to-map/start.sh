#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
RUN="$ROOT/.run"
mkdir -p "$RUN"

if [ -z "$OPENAI_API_KEY" ]; then
  echo "OPENAI_API_KEY is not set"
  exit 1
fi

PORT="${PORT:-8080}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

cd "$ROOT/backend"
go build -o "$RUN/backend-bin" .
PORT="$PORT" OPENAI_API_KEY="$OPENAI_API_KEY" "$RUN/backend-bin" >"$RUN/backend.log" 2>&1 &
echo $! >"$RUN/backend.pid"

cd "$ROOT/frontend"
if [ ! -d node_modules ]; then
  npm install >"$RUN/frontend-install.log" 2>&1
fi
VITE_API_BASE="http://localhost:$PORT" npm run dev -- --port "$FRONTEND_PORT" >"$RUN/frontend.log" 2>&1 &
echo $! >"$RUN/frontend.pid"

until curl -sf "http://localhost:$PORT/api/health" >/dev/null 2>&1; do
  sleep 1
done
echo "backend ready on http://localhost:$PORT"

until curl -sf "http://localhost:$FRONTEND_PORT" >/dev/null 2>&1; do
  sleep 1
done
echo "frontend ready on http://localhost:$FRONTEND_PORT"

echo "open http://localhost:$FRONTEND_PORT"
