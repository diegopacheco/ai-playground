#!/bin/bash
cd "$(dirname "$0")"
if [ -f .frontend.pid ]; then
  kill "$(cat .frontend.pid)" 2>/dev/null
  rm -f .frontend.pid
  echo "Frontend stopped"
else
  echo "No frontend PID file found"
fi
