#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ -f .jigsaw.pid ] && kill -0 "$(cat .jigsaw.pid)" 2>/dev/null; then
  exit 0
fi
node server.js > .jigsaw.log 2>&1 &
process_id=$!
echo "$process_id" > .jigsaw.pid
for attempt in $(seq 1 30); do
  if curl -fs http://127.0.0.1:4177/health >/dev/null; then
    echo "Big Cat Jigsaw is running at http://localhost:4177"
    exit 0
  fi
  sleep 1
done
kill "$process_id" 2>/dev/null || true
rm -f .jigsaw.pid
exit 1
