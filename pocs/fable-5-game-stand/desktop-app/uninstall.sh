#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ -f .desktop.pid ]; then
  pid="$(cat .desktop.pid)"
  kill "$pid" 2>/dev/null || true
  for i in $(seq 1 30); do
    if ! kill -0 "$pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f .desktop.pid
fi
rm -rf node_modules .venv
echo "Game Stand desktop app uninstalled"
