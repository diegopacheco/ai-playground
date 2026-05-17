#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "[1/3] running backend unit + integration tests"
(cd backend && cargo test --quiet)

echo "[2/3] building frontend bundle"
deno task build >/dev/null

echo "[3/3] end-to-end api smoke check"
./backend/target/debug/memory-backend &
PID=$!
echo $PID > .test-backend.pid
trap 'kill $PID 2>/dev/null; rm -f .test-backend.pid' EXIT

until curl -fs http://127.0.0.1:8080/api/health >/dev/null 2>&1; do
  sleep 1
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "backend died before becoming ready"
    exit 1
  fi
done

HEALTH=$(curl -fs http://127.0.0.1:8080/api/health)
echo "  health: $HEALTH"

DECK_COUNT=$(curl -fs -X POST http://127.0.0.1:8080/api/games | grep -o '"id"' | wc -l | tr -d ' ')
echo "  /api/games returned $DECK_COUNT cards"
if [ "$DECK_COUNT" != "16" ]; then
  echo "  expected 16 cards"
  exit 1
fi

POST_STATUS=$(curl -fs -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" \
  -d '{"name":"diego","moves":10,"seconds":42}' \
  http://127.0.0.1:8080/api/scores)
echo "  POST /api/scores status: $POST_STATUS"
if [ "$POST_STATUS" != "201" ]; then
  echo "  expected 201"
  exit 1
fi

SCORES=$(curl -fs http://127.0.0.1:8080/api/scores)
echo "  GET /api/scores: $SCORES"
case "$SCORES" in
  *diego*) ;;
  *) echo "  expected diego in scores"; exit 1 ;;
esac

echo "all tests passed"
