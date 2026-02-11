#!/bin/bash

echo "k6 Performance Test Setup Verification"
echo "======================================="
echo ""

echo "1. Checking k6 installation..."
if command -v k6 &> /dev/null; then
  K6_VERSION=$(k6 version)
  echo "   k6 is installed: $K6_VERSION"
else
  echo "   k6 is NOT installed"
  echo "   Install with: brew install k6"
  exit 1
fi
echo ""

echo "2. Checking backend server..."
BASE_URL=${BASE_URL:-http://localhost:8000}
if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
  echo "   Backend is running at $BASE_URL"
else
  echo "   Backend is NOT responding at $BASE_URL"
  echo "   Start the server with: ./start.sh"
  exit 1
fi
echo ""

echo "3. Checking test files..."
TEST_FILES=(
  "k6-tests/baseline-test.js"
  "k6-tests/auth-test.js"
  "k6-tests/user-profile-test.js"
  "k6-tests/tweet-feed-test.js"
  "k6-tests/social-interactions-test.js"
  "k6-tests/load-test.js"
  "k6-tests/stress-test.js"
)

ALL_EXIST=true
for file in "${TEST_FILES[@]}"; do
  if [ -f "$file" ]; then
    echo "   $file"
  else
    echo "   MISSING: $file"
    ALL_EXIST=false
  fi
done

if [ "$ALL_EXIST" = false ]; then
  exit 1
fi
echo ""

echo "4. Checking results directory..."
if [ ! -d "k6-results" ]; then
  echo "   Creating k6-results directory..."
  mkdir -p k6-results
fi
echo "   k6-results directory ready"
echo ""

echo "5. Testing basic API connectivity..."
REGISTER_PAYLOAD='{"username":"verify_test","email":"verify@test.com","password":"password123"}'
REGISTER_RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/k6-verify-response.json -X POST "$BASE_URL/api/auth/register" \
  -H "Content-Type: application/json" \
  -d "$REGISTER_PAYLOAD")

if [ "$REGISTER_RESPONSE" = "201" ] || [ "$REGISTER_RESPONSE" = "400" ]; then
  echo "   API is responding correctly"
else
  echo "   API returned unexpected status: $REGISTER_RESPONSE"
  exit 1
fi
echo ""

echo "All checks passed!"
echo ""
echo "Ready to run performance tests:"
echo "  ./run-single-k6-test.sh baseline    - Quick 30s baseline test"
echo "  ./run-single-k6-test.sh load        - 4.5 min load test"
echo "  ./run-single-k6-test.sh stress      - 9 min stress test"
echo "  ./run-k6-tests.sh                   - Run all tests (~30 min)"
echo ""
