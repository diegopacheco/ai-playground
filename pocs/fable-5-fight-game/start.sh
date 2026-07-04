#!/bin/bash
cd "$(dirname "$0")"
if [ -f .server.pid ] && kill -0 "$(cat .server.pid)" 2>/dev/null; then
  echo "Already running at http://localhost:8123"
  exit 0
fi
python3 -m http.server 8123 >/dev/null 2>&1 &
echo $! > .server.pid
echo "Fable Fighters running at http://localhost:8123"
