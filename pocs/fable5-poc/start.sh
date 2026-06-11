#!/bin/bash
cd "$(dirname "$0")"
PORT=8422
if lsof -ti tcp:$PORT > /dev/null 2>&1; then
  echo "already running on http://localhost:$PORT"
  exit 0
fi
python3 -m http.server $PORT > /dev/null 2>&1 &
echo $! > .server.pid
for i in $(seq 1 30); do
  if curl -s -o /dev/null "http://localhost:$PORT"; then
    echo "running on http://localhost:$PORT"
    exit 0
  fi
  sleep 1
done
echo "failed to start"
exit 1
