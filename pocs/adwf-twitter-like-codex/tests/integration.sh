#!/usr/bin/env bash
set -euo pipefail
rm -f db/app.db
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
curl -sf -X POST http://127.0.0.1:3001/api/users -H 'content-type: application/json' -d '{"username":"bob"}' >/tmp/u.json
curl -sf -X POST http://127.0.0.1:3001/api/posts -H 'content-type: application/json' -d '{"user_id":1,"content":"first post"}' >/tmp/p.json
curl -sf http://127.0.0.1:3001/api/posts >/tmp/l.json
cat /tmp/l.json
