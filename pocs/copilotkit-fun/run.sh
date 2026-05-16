#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$ROOT/backend/node_modules" ]; then
  (cd "$ROOT/backend" && npm install)
fi

if [ ! -d "$ROOT/frontend/node_modules" ]; then
  (cd "$ROOT/frontend" && npm install)
fi

if [ -z "$OPENAI_API_KEY" ]; then
  echo "WARNING: OPENAI_API_KEY is not set. The Copilot chat will fail until you export it."
fi

(cd "$ROOT/backend" && npm start) &
BACKEND_PID=$!

(cd "$ROOT/frontend" && npm run dev) &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true" EXIT INT TERM

echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo "Open http://localhost:3000"

wait
