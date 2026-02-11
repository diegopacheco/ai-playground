#!/usr/bin/env bash
set -euo pipefail
cd backend
cargo run >/tmp/twitter_like_backend.log 2>&1 &
PID=$!
cleanup() {
  kill $PID 2>/dev/null || true
  wait $PID 2>/dev/null || true
}
trap cleanup EXIT
for i in $(seq 1 40); do
  if curl -sf http://127.0.0.1:3001/health >/dev/null; then
    break
  fi
  sleep 1
done
cd ..
k6 run --log-output=none tests/k6.js
