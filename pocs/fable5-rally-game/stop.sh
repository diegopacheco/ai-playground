#!/bin/bash
cd "$(dirname "$0")"
if [ -f server.pid ]; then
  kill "$(cat server.pid)" 2>/dev/null
  rm -f server.pid
  echo "Server stopped"
else
  echo "No server running"
fi
