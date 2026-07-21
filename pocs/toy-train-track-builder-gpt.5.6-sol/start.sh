#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ ! -d node_modules ]; then
  npm install
fi
if [ -f .blockrails.pid ] && kill -0 "$(cat .blockrails.pid)" 2>/dev/null; then
  echo "BlockRails is already running"
  exit 0
fi
nohup ./node_modules/.bin/vite --host 0.0.0.0 > /tmp/blockrails.log 2>&1 &
echo $! > .blockrails.pid
for _ in {1..60}; do
  pid="$(cat .blockrails.pid)"
  if kill -0 "$pid" 2>/dev/null && curl -fsS http://127.0.0.1:5173 >/dev/null 2>&1; then
    echo "BlockRails is running at http://127.0.0.1:5173"
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
echo "BlockRails did not start"
rm -f .blockrails.pid
exit 1
