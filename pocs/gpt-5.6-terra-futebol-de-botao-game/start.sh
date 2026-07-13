#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PORT="${PORT:-8091}"
if [ -f .server.pid ]; then
  PID=$(cat .server.pid)
  if kill -0 "$PID" 2>/dev/null; then
    echo "Mesa 12 already running at http://127.0.0.1:$PORT"
    exit 0
  fi
  rm -f .server.pid
fi
if command -v python3 >/dev/null 2>&1; then
  python3 server.py "$PORT" > .server.log 2>&1 &
else
  python server.py "$PORT" > .server.log 2>&1 &
fi
PID=$!
echo "$PID" > .server.pid
for i in {1..30}; do
  if ! kill -0 "$PID" 2>/dev/null; then
    cat .server.log
    rm -f .server.pid
    exit 1
  fi
  if curl -fsS "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"game": "mesa12"'; then
    echo "Mesa 12 running at http://127.0.0.1:$PORT"
    exit 0
  fi
  sleep 1
done
kill "$PID" 2>/dev/null || true
rm -f .server.pid
echo "Mesa 12 failed to start"
exit 1
