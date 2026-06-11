#!/bin/bash
cd "$(dirname "$0")"
PORT=8422
PIDS=$(lsof -ti tcp:$PORT)
if [ -n "$PIDS" ]; then
  kill $PIDS
  echo "stopped"
else
  echo "not running"
fi
rm -f .server.pid
