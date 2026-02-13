#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_DIR/backend"
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
EXISTING_PID=$(lsof -ti :8080 2>/dev/null)
if [ -n "$EXISTING_PID" ]; then
  kill $EXISTING_PID 2>/dev/null || true
  sleep 1
fi
rm -f "$DB_FILE"
cd "$BACKEND_DIR" && cargo build --quiet 2>&1
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
echo "Backend is ready"
echo "=== Test: Register User ==="
REGISTER_RESP=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8080/api/users \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","email":"test@test.com","password":"password123"}')
REGISTER_CODE=$(echo "$REGISTER_RESP" | tail -1)
REGISTER_BODY=$(echo "$REGISTER_RESP" | sed '$d')
if [ "$REGISTER_CODE" != "201" ]; then
  echo "FAIL: Register returned $REGISTER_CODE"
  echo "$REGISTER_BODY"
  exit 1
fi
echo "PASS: Register (201)"
echo "=== Test: Login ==="
LOGIN_RESP=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8080/api/users/login \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser","password":"password123"}')
LOGIN_CODE=$(echo "$LOGIN_RESP" | tail -1)
LOGIN_BODY=$(echo "$LOGIN_RESP" | sed '$d')
if [ "$LOGIN_CODE" != "200" ]; then
  echo "FAIL: Login returned $LOGIN_CODE"
  echo "$LOGIN_BODY"
  exit 1
fi
TOKEN=$(echo "$LOGIN_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])")
USER_ID=$(echo "$LOGIN_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['user']['id'])")
echo "PASS: Login (200), got token"
echo "=== Test: Create Post ==="
POST_RESP=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8080/api/posts \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"content":"Hello world from integration test!"}')
POST_CODE=$(echo "$POST_RESP" | tail -1)
POST_BODY=$(echo "$POST_RESP" | sed '$d')
if [ "$POST_CODE" != "200" ] && [ "$POST_CODE" != "201" ]; then
  echo "FAIL: Create post returned $POST_CODE"
  echo "$POST_BODY"
  exit 1
fi
POST_ID=$(echo "$POST_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "PASS: Create Post ($POST_CODE), id=$POST_ID"
echo "=== Test: Get All Posts ==="
POSTS_RESP=$(curl -s -w "\n%{http_code}" http://localhost:8080/api/posts)
POSTS_CODE=$(echo "$POSTS_RESP" | tail -1)
if [ "$POSTS_CODE" != "200" ]; then
  echo "FAIL: Get posts returned $POSTS_CODE"
  exit 1
fi
echo "PASS: Get Posts (200)"
echo "=== Test: Get Single Post ==="
SINGLE_RESP=$(curl -s -w "\n%{http_code}" http://localhost:8080/api/posts/$POST_ID)
SINGLE_CODE=$(echo "$SINGLE_RESP" | tail -1)
if [ "$SINGLE_CODE" != "200" ]; then
  echo "FAIL: Get single post returned $SINGLE_CODE"
  exit 1
fi
echo "PASS: Get Single Post (200)"
echo "=== Test: Like Post ==="
LIKE_RESP=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8080/api/posts/$POST_ID/like \
  -H "Authorization: Bearer $TOKEN")
LIKE_CODE=$(echo "$LIKE_RESP" | tail -1)
if [ "$LIKE_CODE" != "200" ] && [ "$LIKE_CODE" != "201" ]; then
  echo "FAIL: Like returned $LIKE_CODE"
  exit 1
fi
echo "PASS: Like Post ($LIKE_CODE)"
echo "=== Test: Get Like Count ==="
LIKES_RESP=$(curl -s -w "\n%{http_code}" http://localhost:8080/api/posts/$POST_ID/likes)
LIKES_CODE=$(echo "$LIKES_RESP" | tail -1)
LIKES_BODY=$(echo "$LIKES_RESP" | sed '$d')
if [ "$LIKES_CODE" != "200" ]; then
  echo "FAIL: Like count returned $LIKES_CODE"
  exit 1
fi
echo "PASS: Like Count (200)"
echo "=== Test: Register Second User ==="
REG2_RESP=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8080/api/users \
  -H "Content-Type: application/json" \
  -d '{"username":"testuser2","email":"test2@test.com","password":"password123"}')
REG2_CODE=$(echo "$REG2_RESP" | tail -1)
if [ "$REG2_CODE" != "201" ]; then
  echo "FAIL: Register user2 returned $REG2_CODE"
  exit 1
fi
USER2_BODY=$(echo "$REG2_RESP" | sed '$d')
USER2_ID=$(echo "$USER2_BODY" | python3 -c "import sys,json; print(json.load(sys.stdin)['user']['id'])")
echo "PASS: Register User2 (201)"
echo "=== Test: Follow User ==="
FOLLOW_RESP=$(curl -s -w "\n%{http_code}" -X POST http://localhost:8080/api/users/$USER2_ID/follow \
  -H "Authorization: Bearer $TOKEN")
FOLLOW_CODE=$(echo "$FOLLOW_RESP" | tail -1)
if [ "$FOLLOW_CODE" != "200" ] && [ "$FOLLOW_CODE" != "201" ]; then
  echo "FAIL: Follow returned $FOLLOW_CODE"
  exit 1
fi
echo "PASS: Follow User ($FOLLOW_CODE)"
echo "=== Test: Get Feed ==="
FEED_RESP=$(curl -s -w "\n%{http_code}" http://localhost:8080/api/feed \
  -H "Authorization: Bearer $TOKEN")
FEED_CODE=$(echo "$FEED_RESP" | tail -1)
if [ "$FEED_CODE" != "200" ]; then
  echo "FAIL: Feed returned $FEED_CODE"
  exit 1
fi
echo "PASS: Get Feed (200)"
echo "=== Test: Unlike Post ==="
UNLIKE_RESP=$(curl -s -w "\n%{http_code}" -X DELETE http://localhost:8080/api/posts/$POST_ID/like \
  -H "Authorization: Bearer $TOKEN")
UNLIKE_CODE=$(echo "$UNLIKE_RESP" | tail -1)
if [ "$UNLIKE_CODE" != "200" ]; then
  echo "FAIL: Unlike returned $UNLIKE_CODE"
  exit 1
fi
echo "PASS: Unlike Post (200)"
echo "=== Test: Unfollow User ==="
UNFOLLOW_RESP=$(curl -s -w "\n%{http_code}" -X DELETE http://localhost:8080/api/users/$USER2_ID/follow \
  -H "Authorization: Bearer $TOKEN")
UNFOLLOW_CODE=$(echo "$UNFOLLOW_RESP" | tail -1)
if [ "$UNFOLLOW_CODE" != "200" ]; then
  echo "FAIL: Unfollow returned $UNFOLLOW_CODE"
  exit 1
fi
echo "PASS: Unfollow User (200)"
echo "=== Test: Delete Post ==="
DEL_RESP=$(curl -s -w "\n%{http_code}" -X DELETE http://localhost:8080/api/posts/$POST_ID \
  -H "Authorization: Bearer $TOKEN")
DEL_CODE=$(echo "$DEL_RESP" | tail -1)
if [ "$DEL_CODE" != "200" ]; then
  echo "FAIL: Delete post returned $DEL_CODE"
  exit 1
fi
echo "PASS: Delete Post (200)"
echo ""
echo "ALL INTEGRATION TESTS PASSED"
