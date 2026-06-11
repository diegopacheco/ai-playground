#!/bin/bash
cd "$(dirname "$0")"
PORT=8377
if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
  echo "BLASTGRID already running at http://localhost:$PORT"
  exit 0
fi
python3 -m http.server $PORT >/dev/null 2>&1 &
echo $! > server.pid
for i in $(seq 1 30); do
  if curl -s -o /dev/null "http://localhost:$PORT"; then
    echo "BLASTGRID running at http://localhost:$PORT"
    exit 0
  fi
  sleep 1
done
echo "failed to start server"
rm -f server.pid
exit 1
