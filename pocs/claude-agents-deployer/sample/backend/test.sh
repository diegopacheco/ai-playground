#!/bin/bash
BASE_URL="http://localhost:8080/api"

echo "=== Creating a user ==="
USER_RESPONSE=$(curl -s -X POST "$BASE_URL/users" \
  -H "Content-Type: application/json" \
  -d '{"name":"John Doe","email":"john@test.com"}')
echo "$USER_RESPONSE"

echo ""
echo "=== Listing users ==="
curl -s "$BASE_URL/users"

echo ""
echo "=== Creating a post ==="
POST_RESPONSE=$(curl -s -X POST "$BASE_URL/posts" \
  -H "Content-Type: application/json" \
  -d '{"title":"First Post","content":"Hello World","author":"John Doe"}')
echo "$POST_RESPONSE"
POST_ID=$(echo "$POST_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])" 2>/dev/null)

echo ""
echo "=== Listing posts ==="
curl -s "$BASE_URL/posts"

echo ""
echo "=== Updating post ==="
curl -s -X PUT "$BASE_URL/posts/$POST_ID" \
  -H "Content-Type: application/json" \
  -d '{"title":"Updated First Post"}'

echo ""
echo "=== Creating a comment ==="
curl -s -X POST "$BASE_URL/posts/$POST_ID/comments" \
  -H "Content-Type: application/json" \
  -d '{"content":"Great post!","author":"Jane Doe"}'

echo ""
echo "=== Listing comments for post ==="
curl -s "$BASE_URL/posts/$POST_ID/comments"

echo ""
echo "=== All tests completed ==="
