#!/bin/bash
cd "$(dirname "$0")"
if [ -f .backend.pid ]; then
  kill "$(cat .backend.pid)" 2>/dev/null
  rm -f .backend.pid
  echo "Backend stopped"
else
  echo "No backend PID file found"
fi
