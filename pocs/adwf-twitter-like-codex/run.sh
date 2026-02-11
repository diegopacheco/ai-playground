#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT_DIR/.run"
cd "$ROOT_DIR/backend"
nohup cargo run > "$ROOT_DIR/.run/backend.log" 2>&1 < /dev/null &
echo $! > "$ROOT_DIR/.run/backend.pid"
cd "$ROOT_DIR/frontend"
nohup bun run start > "$ROOT_DIR/.run/frontend.log" 2>&1 < /dev/null &
echo $! > "$ROOT_DIR/.run/frontend.pid"
cd "$ROOT_DIR"
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
if ! curl -sf http://127.0.0.1:3001/health >/dev/null; then
  echo backend-failed
  exit 1
fi
if ! curl -sf http://127.0.0.1:4173 >/dev/null; then
  echo frontend-failed
  exit 1
fi
echo backend=http://127.0.0.1:3001
echo frontend=http://127.0.0.1:4173
