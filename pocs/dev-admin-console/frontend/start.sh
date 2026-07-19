#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
./stop.sh > /dev/null 2>&1 || true
for attempt in $(seq 1 15); do
  if ! lsof -ti tcp:4321 > /dev/null 2>&1; then
    break
  fi
  sleep 1
done
if [ ! -d node_modules ]; then
  bun install
fi
nohup bun run dev > /tmp/dev-admin-console-frontend.log 2>&1 &
echo $! > /tmp/dev-admin-console-frontend.pid
for attempt in $(seq 1 60); do
  if curl -fsS http://localhost:4321/ > /dev/null 2>&1; then
    echo "frontend ready on http://localhost:4321"
    exit 0
  fi
  sleep 1
done
tail -20 /tmp/dev-admin-console-frontend.log
exit 1
