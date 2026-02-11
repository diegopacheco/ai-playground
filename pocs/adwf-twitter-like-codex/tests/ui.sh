#!/usr/bin/env bash
set -euo pipefail
cd frontend
bunx --bun serve . -l 4173 >/tmp/twitter_like_frontend.log 2>&1 &
FPID=$!
for i in $(seq 1 40); do
  if curl -sf http://127.0.0.1:4173 >/dev/null; then
    break
  fi
  sleep 1
done
bunx playwright test tests/ui.spec.ts
kill $FPID
wait $FPID 2>/dev/null || true
