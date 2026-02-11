#!/usr/bin/env bash
set -euo pipefail
mkdir -p .run
cd backend
cargo run > ../.run/chaos-backend.log 2>&1 &
PID=$!
cd ..
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:3001/health >/dev/null; then
    break
  fi
  sleep 1
done
curl -sf http://127.0.0.1:3001/health >/dev/null
kill $PID
for i in $(seq 1 10); do
  if ! curl -sf http://127.0.0.1:3001/health >/dev/null; then
    break
  fi
  sleep 1
done
cd backend
cargo run > ../.run/chaos-backend.log 2>&1 &
PID2=$!
cd ..
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:3001/health >/dev/null; then
    break
  fi
  sleep 1
done
curl -sf http://127.0.0.1:3001/health >/dev/null
kill $PID2
wait $PID2 2>/dev/null || true
echo chaos-ok
