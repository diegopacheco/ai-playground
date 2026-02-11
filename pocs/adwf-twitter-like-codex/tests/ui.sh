#!/usr/bin/env bash
set -euo pipefail
rm -f db/app.db
cd backend
cargo run >/tmp/twitter_like_backend.log 2>&1 &
BPID=$!
cd ..
cd frontend
bunx --bun serve . -l 4173 >/tmp/twitter_like_frontend.log 2>&1 &
FPID=$!
cleanup() {
  kill $FPID 2>/dev/null || true
  wait $FPID 2>/dev/null || true
  kill $BPID 2>/dev/null || true
  wait $BPID 2>/dev/null || true
}
trap cleanup EXIT
for i in $(seq 1 40); do
  if curl -sf http://127.0.0.1:3001/health >/dev/null; then
    break
  fi
  sleep 1
done
for i in $(seq 1 40); do
  if curl -sf http://127.0.0.1:4173 >/dev/null; then
    break
  fi
  sleep 1
done
bunx playwright test tests/ui.spec.ts --reporter=html
