#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "[1/3] building rust backend"
(cd backend && cargo build --release --quiet)

echo "[2/3] starting backend on 8080"
./backend/target/release/memory-backend &
echo $! > .backend.pid
until lsof -iTCP:8080 -sTCP:LISTEN -P -n >/dev/null 2>&1; do
  sleep 1
  if ! kill -0 "$(cat .backend.pid)" 2>/dev/null; then
    echo "backend failed to start"
    rm -f .backend.pid
    exit 1
  fi
done
echo "backend up at http://localhost:8080"

echo "[3/3] building and starting frontend on 8000"
deno task build
deno task serve &
echo $! > .frontend.pid
until lsof -iTCP:8000 -sTCP:LISTEN -P -n >/dev/null 2>&1; do
  sleep 1
  if ! kill -0 "$(cat .frontend.pid)" 2>/dev/null; then
    echo "frontend failed to start"
    rm -f .frontend.pid
    exit 1
  fi
done
echo "frontend up at http://localhost:8000"
