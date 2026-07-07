#!/bin/bash
cd "$(dirname "$0")"
lsof -ti tcp:8017 | xargs kill -9 2>/dev/null
python3 server.py >/dev/null 2>&1 &
echo $! > .server.pid
for i in $(seq 1 20); do
  if curl -s -o /dev/null http://localhost:8017/; then
    echo "Split or Steal running at http://localhost:8017"
    exit 0
  fi
  sleep 1
done
echo "server failed to start"
exit 1
