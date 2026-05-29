#!/bin/bash
set -e

if [ -z "$GEMINI_API_KEY" ]; then
  echo "GEMINI_API_KEY env var is not set. Export it before running."
  exit 1
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT/backend"
[ -d node_modules ] || npm install
cd "$ROOT/frontend"
[ -d node_modules ] || npm install

cd "$ROOT/backend"
GEMINI_API_KEY="$GEMINI_API_KEY" npm start > "$ROOT/backend.log" 2>&1 &
echo $! > "$ROOT/.backend.pid"

cd "$ROOT/frontend"
npm run dev > "$ROOT/frontend.log" 2>&1 &
echo $! > "$ROOT/.frontend.pid"

until curl -s http://localhost:3001/api/status/ping > /dev/null 2>&1; do sleep 1; done
until curl -s http://localhost:5173 > /dev/null 2>&1; do sleep 1; done

echo "Backend:  http://localhost:3001"
echo "Frontend: http://localhost:5173"
echo "Logs: backend.log / frontend.log"
