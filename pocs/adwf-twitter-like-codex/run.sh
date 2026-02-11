#!/usr/bin/env bash
set -euo pipefail
mkdir -p .run
cd backend
cargo run > ../.run/backend.log 2>&1 &
echo $! > ../.run/backend.pid
cd ../frontend
bun run start > ../.run/frontend.log 2>&1 &
echo $! > ../.run/frontend.pid
cd ..
for i in $(seq 1 60); do
  b=0
  f=0
  if curl -sf http://127.0.0.1:3001/health >/dev/null; then
    b=1
  fi
  if curl -sf http://127.0.0.1:4173 >/dev/null; then
    f=1
  fi
  if [ "$b" = "1" ] && [ "$f" = "1" ]; then
    break
  fi
  sleep 1
done
echo backend=http://127.0.0.1:3001
echo frontend=http://127.0.0.1:4173
