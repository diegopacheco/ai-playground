#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ -f server.pid ]; then
  kill "$(cat server.pid)" 2>/dev/null || true
  rm -f server.pid
fi
pkill -f "air-goalkeeper-game-cv/server.py" 2>/dev/null
for port in 18080 18765; do
  for pid in $(lsof -ti tcp:$port 2>/dev/null); do
    kill "$pid" 2>/dev/null || true
  done
done
echo "stopped"
