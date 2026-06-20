#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PORT="${PORT:-4321}"
if [ -f .server.pid ] && kill -0 "$(cat .server.pid)" 2>/dev/null; then
  echo "Times 2 already running on http://localhost:${PORT} (pid $(cat .server.pid))"
  exit 0
fi
PORT="$PORT" node server.js >server.log 2>&1 &
echo $! >.server.pid
for i in $(seq 1 30); do
  if curl -s "http://localhost:${PORT}/" >/dev/null 2>&1; then
    echo "Times 2 running on http://localhost:${PORT} (pid $(cat .server.pid))"
    exit 0
  fi
  sleep 1
done
echo "Times 2 failed to start, see server.log"
exit 1
