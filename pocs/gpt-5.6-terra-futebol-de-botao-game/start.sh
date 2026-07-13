#!/bin/bash
set -e
PORT="${PORT:-8080}"
if command -v python3 >/dev/null 2>&1; then
  python3 server.py "$PORT" > .server.log 2>&1 &
else
  python server.py "$PORT" > .server.log 2>&1 &
fi
echo $! > .server.pid
echo "Mesa 12 running at http://127.0.0.1:$PORT"
