#!/usr/bin/env bash
set -e
port="${PORT:-8080}"
if [ -f .server.pid ] && kill -0 "$(cat .server.pid)" 2>/dev/null; then
  echo "Terra Vivo already running at http://127.0.0.1:$port"
  exit 0
fi
nohup python3 -m http.server "$port" --bind 127.0.0.1 > .server.log 2>&1 &
echo $! > .server.pid
echo "Terra Vivo running at http://127.0.0.1:$port"
