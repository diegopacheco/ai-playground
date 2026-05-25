#!/usr/bin/env bash
set -e
PORT=8753
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
if [ -f .server.pid ] && kill -0 "$(cat .server.pid)" 2>/dev/null; then
  echo "already running on http://localhost:$PORT"
  exit 0
fi
python3 -m http.server "$PORT" >/dev/null 2>&1 &
echo $! > .server.pid
until curl -s "http://localhost:$PORT" >/dev/null 2>&1; do
  sleep 1
done
echo "MEGA SLOP running at http://localhost:$PORT"
if command -v open >/dev/null 2>&1; then
  open "http://localhost:$PORT"
fi
