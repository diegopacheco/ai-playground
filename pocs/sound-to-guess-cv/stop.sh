#!/usr/bin/env bash
cd "$(dirname "$0")"

if [ -f server.pid ]; then
  kill "$(cat server.pid)" 2>/dev/null
  rm -f server.pid
fi
pkill -f "http.server 8090" 2>/dev/null
echo "stopped"
