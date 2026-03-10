#!/bin/bash
cd "$(dirname "$0")"
if [ -f app.pid ]; then
  kill "$(cat app.pid)" 2>/dev/null
  rm -f app.pid
  echo "Stopped stock-dashboard"
else
  echo "No PID file found"
fi
