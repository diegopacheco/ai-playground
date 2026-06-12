#!/bin/bash
cd "$(dirname "$0")"
if [ -f server.pid ]; then
  kill "$(cat server.pid)" 2>/dev/null || true
  rm -f server.pid
fi
pkill -f "java -cp out Server" 2>/dev/null || true
echo "maze race stopped"
