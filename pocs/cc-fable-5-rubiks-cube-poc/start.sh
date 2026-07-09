#!/bin/bash
cd "$(dirname "$0")"
if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
  echo "already running on http://localhost:8093"
  exit 0
fi
python3 -m http.server 8093 >/dev/null 2>&1 &
echo $! > server.pid
for i in $(seq 1 30); do
  if curl -s -o /dev/null http://localhost:8093; then
    echo "running on http://localhost:8093"
    exit 0
  fi
  sleep 1
done
echo "failed to start"
exit 1
