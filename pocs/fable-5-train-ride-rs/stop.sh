#!/bin/bash
cd "$(dirname "$0")"
if [ -f server.pid ]; then
  kill "$(cat server.pid)" 2>/dev/null
  rm -f server.pid
  echo "stopped"
else
  echo "not running"
fi
