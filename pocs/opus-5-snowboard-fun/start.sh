#!/bin/bash
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8123}"

if [ -f .server.pid ] && kill -0 "$(cat .server.pid)" 2>/dev/null; then
  echo "already running on port $PORT (pid $(cat .server.pid))"
  exit 0
fi

PORT="$PORT" node server.js > server.log 2>&1 &
echo $! > .server.pid

for i in $(seq 1 30); do
  if curl -sf "http://localhost:$PORT/" > /dev/null; then
    echo "alpine carve running: http://localhost:$PORT"
    exit 0
  fi
  sleep 1
done

echo "failed to start, see server.log"
cat server.log
exit 1
