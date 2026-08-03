#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f .fly-catcher.state ] && [ ! -f .fly-catcher.pid ]; then
  echo "Fly Catcher is not running"
  exit 0
fi
if [ -f .fly-catcher.state ]; then
  pid="$(node -p "try{JSON.parse(require('node:fs').readFileSync('.fly-catcher.state')).pid}catch{''}")"
else
  pid="$(<.fly-catcher.pid)"
fi
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  for _ in $(seq 1 10); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
fi
rm -f .fly-catcher.state .fly-catcher.pid
echo "Fly Catcher stopped"
