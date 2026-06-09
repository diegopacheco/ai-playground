#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set. Copy env.sample to .env and set it."
  exit 1
fi
PORT="${PORT:-3000}"
if lsof -ti ":${PORT}" >/dev/null 2>&1; then
  echo "port ${PORT} is already in use. run ./stop.sh first, or set PORT to a free port."
  exit 1
fi
node server.js &
PID=$!
echo "$PID" > server.pid
until curl -s "http://localhost:${PORT}/" >/dev/null 2>&1; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "server failed to start"
    rm -f server.pid
    exit 1
  fi
  sleep 1
done
echo "ready http://localhost:${PORT}"
