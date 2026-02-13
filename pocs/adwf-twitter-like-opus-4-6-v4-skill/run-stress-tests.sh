#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if ! command -v k6 > /dev/null 2>&1; then
  echo "k6 is not installed. Install it with: brew install k6"
  echo "Skipping stress tests."
  exit 0
fi
BACKEND_DIR="$SCRIPT_DIR/backend"
DB_FILE="$BACKEND_DIR/twitter.db"
BACKEND_PID=""
cleanup() {
  if [ -n "$BACKEND_PID" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi
  rm -f "$DB_FILE"
}
trap cleanup EXIT
rm -f "$DB_FILE"
cd "$BACKEND_DIR" && ./target/debug/twitter-backend &
BACKEND_PID=$!
READY=0
for i in $(seq 1 30); do
  if curl -s http://localhost:8080/api/posts > /dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done
if [ $READY -eq 0 ]; then
  echo "FAIL: Backend did not start"
  exit 1
fi
k6 run "$SCRIPT_DIR/stress-tests/load-test.js" 2>&1
echo "STRESS TESTS COMPLETED"
