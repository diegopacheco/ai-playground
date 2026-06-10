#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PORT="${PORT:-7820}"

node scan.js

if [ -f .server.pid ] && kill -0 "$(cat .server.pid)" 2>/dev/null; then
  echo "Server already running (pid $(cat .server.pid)) on http://127.0.0.1:${PORT}"
  exit 0
fi

PORT="$PORT" node server.js > server.log 2>&1 &
echo $! > .server.pid

for i in $(seq 1 30); do
  if curl -s "http://127.0.0.1:${PORT}/data.json" -o /dev/null 2>/dev/null; then
    break
  fi
  sleep 1
done

URL="http://127.0.0.1:${PORT}"
echo "Heatmap up at ${URL} (pid $(cat .server.pid))"
if command -v open >/dev/null 2>&1; then
  open "${URL}"
fi
