#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_URL="http://localhost:8080/api"
PASSED=0
FAILED=0
BACKEND_PID=""

cleanup() {
  if [ -n "$BACKEND_PID" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
  "$PROJECT_ROOT/db/stop-db.sh" 2>/dev/null || true
}
trap cleanup EXIT

get_body() {
  echo "$1" | sed '$d'
}

get_status() {
  echo "$1" | tail -1
}

assert_status() {
  local description="$1"
  local expected="$2"
  local actual="$3"
  if [ "$actual" = "$expected" ]; then
    echo "PASS: $description (HTTP $actual)"
    PASSED=$((PASSED + 1))
  else
    echo "FAIL: $description (expected HTTP $expected, got HTTP $actual)"
    FAILED=$((FAILED + 1))
  fi
}

assert_contains() {
  local description="$1"
  local needle="$2"
  local haystack="$3"
  if echo "$haystack" | grep -q "$needle"; then
    echo "PASS: $description"
    PASSED=$((PASSED + 1))
  else
    echo "FAIL: $description (response does not contain '$needle')"
    FAILED=$((FAILED + 1))
  fi
}

echo "Starting database..."
"$PROJECT_ROOT/db/start-db.sh"
echo "Applying schema..."
"$PROJECT_ROOT/db/create-schema.sh"

echo "Building and starting backend..."
cd "$PROJECT_ROOT/backend"
DATABASE_URL="postgresql://twitter:twitter123@localhost:5432/twitter" cargo run &
BACKEND_PID=$!

while ! curl -s "$BASE_URL/users/1" > /dev/null 2>&1; do
  sleep 1
done
echo "Backend is ready."

TIMESTAMP=$(date +%s)
USERNAME="testuser_${TIMESTAMP}"
EMAIL="test_${TIMESTAMP}@test.com"

echo ""
echo "=== Register ==="
REGISTER_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/auth/register" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"$USERNAME\",\"email\":\"$EMAIL\",\"password\":\"password123\"}")
REGISTER_BODY=$(get_body "$REGISTER_RESPONSE")
REGISTER_STATUS=$(get_status "$REGISTER_RESPONSE")
assert_status "Register user" "200" "$REGISTER_STATUS"
assert_contains "Register returns token" "token" "$REGISTER_BODY"

echo ""
echo "=== Login ==="
LOGIN_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL\",\"password\":\"password123\"}")
LOGIN_BODY=$(get_body "$LOGIN_RESPONSE")
LOGIN_STATUS=$(get_status "$LOGIN_RESPONSE")
assert_status "Login user" "200" "$LOGIN_STATUS"
assert_contains "Login returns token" "token" "$LOGIN_BODY"

TOKEN=$(echo "$LOGIN_BODY" | grep -o '"token":"[^"]*"' | head -1 | cut -d'"' -f4)
USER_ID=$(echo "$LOGIN_BODY" | grep -o '"id":[0-9]*' | head -1 | cut -d: -f2)

echo ""
echo "=== Create Tweet ==="
TWEET_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/tweets" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"content":"Hello from integration test"}')
TWEET_BODY=$(get_body "$TWEET_RESPONSE")
TWEET_STATUS=$(get_status "$TWEET_RESPONSE")
assert_status "Create tweet" "201" "$TWEET_STATUS"
assert_contains "Tweet has content" "Hello from integration test" "$TWEET_BODY"

TWEET_ID=$(echo "$TWEET_BODY" | grep -o '"id":[0-9]*' | head -1 | cut -d: -f2)

echo ""
echo "=== Get Tweet ==="
GET_TWEET_RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/tweets/$TWEET_ID" \
  -H "Authorization: Bearer $TOKEN")
GET_TWEET_STATUS=$(get_status "$GET_TWEET_RESPONSE")
assert_status "Get tweet" "200" "$GET_TWEET_STATUS"

echo ""
echo "=== Get Feed ==="
FEED_RESPONSE=$(curl -s -w "\n%{http_code}" "$BASE_URL/tweets/feed" \
  -H "Authorization: Bearer $TOKEN")
FEED_STATUS=$(get_status "$FEED_RESPONSE")
assert_status "Get feed" "200" "$FEED_STATUS"

echo ""
echo "=== Like Tweet ==="
LIKE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/tweets/$TWEET_ID/like" \
  -H "Authorization: Bearer $TOKEN")
LIKE_STATUS=$(get_status "$LIKE_RESPONSE")
assert_status "Like tweet" "200" "$LIKE_STATUS"

echo ""
echo "=== Unlike Tweet ==="
UNLIKE_RESPONSE=$(curl -s -w "\n%{http_code}" -X DELETE "$BASE_URL/tweets/$TWEET_ID/like" \
  -H "Authorization: Bearer $TOKEN")
UNLIKE_STATUS=$(get_status "$UNLIKE_RESPONSE")
assert_status "Unlike tweet" "200" "$UNLIKE_STATUS"

TIMESTAMP2=$(date +%s)
USERNAME2="testuser2_${TIMESTAMP2}"
EMAIL2="test2_${TIMESTAMP2}@test.com"
curl -s -X POST "$BASE_URL/auth/register" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"$USERNAME2\",\"email\":\"$EMAIL2\",\"password\":\"password123\"}" > /dev/null
LOGIN2_BODY=$(curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"$EMAIL2\",\"password\":\"password123\"}")
USER2_ID=$(echo "$LOGIN2_BODY" | grep -o '"id":[0-9]*' | head -1 | cut -d: -f2)

echo ""
echo "=== Follow User ==="
FOLLOW_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "$BASE_URL/users/$USER2_ID/follow" \
  -H "Authorization: Bearer $TOKEN")
FOLLOW_STATUS=$(get_status "$FOLLOW_RESPONSE")
assert_status "Follow user" "200" "$FOLLOW_STATUS"

echo ""
echo "=== Unfollow User ==="
UNFOLLOW_RESPONSE=$(curl -s -w "\n%{http_code}" -X DELETE "$BASE_URL/users/$USER2_ID/follow" \
  -H "Authorization: Bearer $TOKEN")
UNFOLLOW_STATUS=$(get_status "$UNFOLLOW_RESPONSE")
assert_status "Unfollow user" "200" "$UNFOLLOW_STATUS"

echo ""
echo "=== Delete Tweet ==="
DELETE_RESPONSE=$(curl -s -w "\n%{http_code}" -X DELETE "$BASE_URL/tweets/$TWEET_ID" \
  -H "Authorization: Bearer $TOKEN")
DELETE_STATUS=$(get_status "$DELETE_RESPONSE")
assert_status "Delete tweet" "204" "$DELETE_STATUS"

echo ""
echo "========================="
echo "Results: $PASSED passed, $FAILED failed"
echo "========================="

if [ "$FAILED" -gt 0 ]; then
  exit 1
fi
exit 0
