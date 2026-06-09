#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
node engine.test.mjs
PORT="${PORT:-3007}"
OPENAI_API_KEY="${OPENAI_API_KEY:-test}" PORT="${PORT}" node server.js &
PID=$!
echo $PID > server.test.pid
until curl -s "http://localhost:${PORT}/" >/dev/null 2>&1; do
  sleep 1
done
if curl -s "http://localhost:${PORT}/" | grep -q "LLM AI Spreadsheet"; then
  echo "static ok"
else
  echo "static failed"
  kill "$PID" 2>/dev/null || true
  rm -f server.test.pid
  exit 1
fi
curl -s -X POST "http://localhost:${PORT}/api/ai" -H 'Content-Type: application/json' -d '{"prompt":"ping"}' -o /dev/null -w "ai endpoint status %{http_code}\n"
kill "$PID" 2>/dev/null || true
rm -f server.test.pid
echo "done"
