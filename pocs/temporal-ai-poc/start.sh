#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

podman-compose up -d

echo "Waiting for Temporal server on localhost:7233"
until (exec 3<>/dev/tcp/localhost/7233) 2>/dev/null; do
  sleep 1
done
exec 3>&- || true

echo "Waiting for Temporal UI on localhost:8233"
until (exec 3<>/dev/tcp/localhost/8233) 2>/dev/null; do
  sleep 1
done
exec 3>&- || true

if [ ! -d node_modules ]; then
  npm install
fi

if [ -f worker.pid ] && kill -0 "$(cat worker.pid)" 2>/dev/null; then
  echo "Worker already running (pid $(cat worker.pid))"
else
  node src/worker.ts > worker.log 2>&1 &
  echo $! > worker.pid
  echo "Worker started (pid $(cat worker.pid)), logs in worker.log"
fi

echo ""
echo "Temporal UI:  http://localhost:8233"
echo "Run a pipeline: ./test.sh"
