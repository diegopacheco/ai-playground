#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ -f .server.pid ] && kill -0 "$(cat .server.pid)" 2>/dev/null; then
  echo "Wave simulator is already running at http://localhost:4173"
  exit 0
fi
nohup python3 -m http.server 4173 > .server.log 2>&1 &
echo $! > .server.pid
for _ in $(seq 1 30); do
  if curl -fsS http://localhost:4173/ >/dev/null 2>&1; then
    echo "Wave simulator is running at http://localhost:4173"
    exit 0
  fi
  sleep 1
done
echo "Wave simulator failed to start"
rm -f .server.pid
exit 1
