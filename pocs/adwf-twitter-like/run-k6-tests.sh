#!/bin/bash

BASE_URL=${BASE_URL:-http://localhost:8000}
RESULTS_DIR="k6-results"

mkdir -p "$RESULTS_DIR"

echo "Starting k6 Performance Tests"
echo "Base URL: $BASE_URL"
echo "Results directory: $RESULTS_DIR"
echo ""

run_test() {
  local test_name=$1
  local test_file=$2
  local output_file="$RESULTS_DIR/${test_name}-$(date +%Y%m%d-%H%M%S).json"

  echo "Running $test_name..."
  k6 run --out json="$output_file" --summary-export="$RESULTS_DIR/${test_name}-summary.json" "$test_file"

  if [ $? -eq 0 ]; then
    echo "$test_name completed successfully"
  else
    echo "$test_name failed"
  fi
  echo ""
}

curl -s "$BASE_URL/health" > /dev/null 2>&1
if [ $? -ne 0 ]; then
  echo "Warning: Backend not responding at $BASE_URL"
  echo "Make sure the server is running before proceeding"
  read -p "Continue anyway? (y/n) " -n 1 -r
  echo ""
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

run_test "baseline" "k6-tests/baseline-test.js"
sleep 5

run_test "auth" "k6-tests/auth-test.js"
sleep 5

run_test "user-profile" "k6-tests/user-profile-test.js"
sleep 5

run_test "tweet-feed" "k6-tests/tweet-feed-test.js"
sleep 5

run_test "social-interactions" "k6-tests/social-interactions-test.js"
sleep 5

run_test "load" "k6-tests/load-test.js"
sleep 5

run_test "stress" "k6-tests/stress-test.js"

echo "All tests completed"
echo "Results saved to $RESULTS_DIR/"
echo ""
echo "Summary files:"
ls -lh "$RESULTS_DIR"/*-summary.json 2>/dev/null
