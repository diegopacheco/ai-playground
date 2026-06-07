#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ -f server.pid ]; then
  kill "$(cat server.pid)" 2>/dev/null
  rm -f server.pid
fi
pkill -f "server.py" 2>/dev/null
echo "stopped"
