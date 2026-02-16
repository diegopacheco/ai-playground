#!/bin/bash
API="http://localhost:8080/api"
PASS=0
FAIL=0

check() {
  local name="$1"
  local expected="$2"
  local actual="$3"
  if echo "$actual" | grep -qF "$expected"; then
    echo "[PASS] $name"
    PASS=$((PASS + 1))
  else
    echo "[FAIL] $name (expected '$expected', got '$actual')"
    FAIL=$((FAIL + 1))
  fi
}

echo "=== Starting backend for tests ==="
rm -f backend/twitter.db
cd backend && cargo run &
SERVER_PID=$!
cd ..
for i in $(seq 1 30); do
  curl -s "$API/tweets" > /dev/null 2>&1 && break
  sleep 1
done

echo ""
echo "=== Register Tests ==="
RES=$(curl -s -X POST "$API/register" -H "Content-Type: application/json" -d '{"username":"testuser","password":"pass123"}')
check "Register new user" "token" "$RES"
TOKEN=$(echo "$RES" | grep -o '"token":"[^"]*"' | cut -d'"' -f4)

RES=$(curl -s -X POST "$API/register" -H "Content-Type: application/json" -d '{"username":"testuser","password":"pass123"}')
check "Register duplicate user fails" "already taken" "$RES"

echo ""
echo "=== Login Tests ==="
RES=$(curl -s -X POST "$API/login" -H "Content-Type: application/json" -d '{"username":"testuser","password":"pass123"}')
check "Login valid user" "token" "$RES"
TOKEN=$(echo "$RES" | grep -o '"token":"[^"]*"' | cut -d'"' -f4)

RES=$(curl -s -X POST "$API/login" -H "Content-Type: application/json" -d '{"username":"testuser","password":"wrong"}')
check "Login wrong password fails" "Invalid credentials" "$RES"

echo ""
echo "=== Tweet Tests ==="
RES=$(curl -s -X POST "$API/tweets" -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" -d '{"content":"Hello World!"}')
check "Create tweet" "Hello World" "$RES"
TWEET_ID=$(echo "$RES" | grep -o '"id":[0-9]*' | head -1 | cut -d':' -f2)

RES=$(curl -s -X POST "$API/tweets" -H "Content-Type: application/json" -d '{"content":"No auth tweet"}')
check "Create tweet without auth fails" "Not authenticated" "$RES"

RES=$(curl -s "$API/tweets")
check "Get all tweets" "Hello World" "$RES"

echo ""
echo "=== Like Tests ==="
RES=$(curl -s -X POST "$API/tweets/$TWEET_ID/like")
check "Like tweet" '"likes":1' "$RES"

RES=$(curl -s -X POST "$API/tweets/$TWEET_ID/like")
check "Like tweet again" '"likes":2' "$RES"

echo ""
echo "=== Search Tests ==="
curl -s -X POST "$API/tweets" -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" -d '{"content":"Rust is great"}' > /dev/null
curl -s -X POST "$API/tweets" -H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN" -d '{"content":"React is fun"}' > /dev/null

RES=$(curl -s "$API/tweets/search?q=Rust")
check "Search tweets by content" "Rust is great" "$RES"

RES=$(curl -s "$API/tweets/search?q=testuser")
check "Search tweets by username" "Hello World" "$RES"

RES=$(curl -s "$API/tweets/search?q=nonexistent")
if [ "$RES" = "[]" ]; then
  echo "[PASS] Search no results is empty array"
  PASS=$((PASS + 1))
else
  echo "[FAIL] Search no results is empty array (got '$RES')"
  FAIL=$((FAIL + 1))
fi

echo ""
echo "=== Image Upload Test ==="
echo "fake image data" > /tmp/test_image.png
RES=$(curl -s -X POST "$API/upload" -F "file=@/tmp/test_image.png")
check "Upload image" "uploads" "$RES"
rm -f /tmp/test_image.png

echo ""
echo "=== Delete Tests ==="
RES2=$(curl -s -X POST "$API/register" -H "Content-Type: application/json" -d '{"username":"other","password":"pass"}')
OTHER_TOKEN=$(echo "$RES2" | grep -o '"token":"[^"]*"' | cut -d'"' -f4)
RES=$(curl -s -X DELETE "$API/tweets/$TWEET_ID" -H "Authorization: Bearer $OTHER_TOKEN")
check "Delete other users tweet fails" "not yours" "$RES"

RES=$(curl -s -X DELETE "$API/tweets/$TWEET_ID" -H "Authorization: Bearer $TOKEN")
check "Delete own tweet" "deleted" "$RES"

echo ""
echo "=== Results ==="
echo "Passed: $PASS"
echo "Failed: $FAIL"
TOTAL=$((PASS + FAIL))
echo "Total:  $TOTAL"

kill $SERVER_PID 2>/dev/null
rm -f backend/twitter.db

if [ $FAIL -gt 0 ]; then
  exit 1
fi
echo "All tests passed!"
