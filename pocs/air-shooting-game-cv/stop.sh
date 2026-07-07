#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ -f server.pid ]; then
  kill "$(cat server.pid)" 2>/dev/null
  rm -f server.pid
fi
pkill -f "air-shooting-game-cv/server.py" 2>/dev/null
echo "stopped"
