#!/bin/bash
cd "$(dirname "$0")"
if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
  echo "already running at http://localhost:8098"
  exit 0
fi
python3 -m http.server 8098 >/dev/null 2>&1 &
echo $! > server.pid
for i in $(seq 1 10); do
  if curl -s -o /dev/null http://localhost:8098; then
    break
  fi
  sleep 1
done
echo "running at http://localhost:8098"
