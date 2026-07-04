#!/bin/bash
cd "$(dirname "$0")"
if [ -f .server.pid ]; then
  kill "$(cat .server.pid)" 2>/dev/null
  rm .server.pid
  echo "Fable Fighters stopped"
else
  echo "Not running"
fi
