#!/usr/bin/env bash
set -e
if [ -f .reelmark-server.pid ] && [ -f .reelmark-web.pid ] && kill -0 "$(<.reelmark-server.pid)" 2>/dev/null && kill -0 "$(<.reelmark-web.pid)" 2>/dev/null; then
  echo "Reelmark is already running"
  exit 0
fi
if [ -f .reelmark-server.pid ] || [ -f .reelmark-web.pid ]; then
  ./stop.sh
fi
bun run server > .reelmark-server.log 2>&1 &
echo $! > .reelmark-server.pid
bun run dev > .reelmark-web.log 2>&1 &
echo $! > .reelmark-web.pid
for _ in {1..60}; do
  if ! kill -0 "$(<.reelmark-server.pid)" 2>/dev/null || ! kill -0 "$(<.reelmark-web.pid)" 2>/dev/null; then
    echo "Reelmark could not start"
    ./stop.sh
    exit 1
  fi
  if curl -sf http://127.0.0.1:3001/api/health >/dev/null && curl -sf http://127.0.0.1:5173 >/dev/null; then
    echo "Open the UI at http://127.0.0.1:5173/"
    exit 0
  fi
  sleep 1
done
echo "Reelmark did not become ready"
./stop.sh
exit 1
