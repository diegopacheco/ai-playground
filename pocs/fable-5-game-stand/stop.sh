#!/bin/bash
cd "$(dirname "$0")"
if [ -f .server.pid ]; then
  kill "$(cat .server.pid)" 2>/dev/null
  rm -f .server.pid
  echo "Game Stand stopped"
else
  pkill -f "python app.py" 2>/dev/null
  echo "no pid file, killed by name"
fi
