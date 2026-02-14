#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PID=""

cleanup() {
  if [ -n "$BACKEND_PID" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  "$SCRIPT_DIR/db/stop-db.sh" 2>/dev/null || true
}
trap cleanup EXIT

kill $(lsof -ti:8080) 2>/dev/null || true

"$SCRIPT_DIR/db/start-db.sh"
"$SCRIPT_DIR/db/create-schema.sh"

cd "$SCRIPT_DIR/backend"
DATABASE_URL="postgresql://twitter:twitter123@localhost:5432/twitter" cargo run &
BACKEND_PID=$!

while ! curl -s "http://localhost:8080/api/users/1" > /dev/null 2>&1; do
  sleep 1
done
echo "Backend is ready."

k6 run "$SCRIPT_DIR/tests/stress/load-test.js" 2>&1
