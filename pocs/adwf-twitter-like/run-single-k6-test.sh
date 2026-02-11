#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <test-name>"
  echo ""
  echo "Available tests:"
  echo "  baseline           - 10 VUs for 30s baseline performance"
  echo "  auth               - Authentication endpoint stress test"
  echo "  user-profile       - User profile operations test"
  echo "  tweet-feed         - Tweet feed retrieval performance"
  echo "  social-interactions - Likes, retweets, comments, follows"
  echo "  load               - Load test ramping to 50 VUs"
  echo "  stress             - Stress test ramping to 100 VUs"
  exit 1
fi

TEST_NAME=$1
BASE_URL=${BASE_URL:-http://localhost:8000}
TEST_FILE="k6-tests/${TEST_NAME}-test.js"

if [ ! -f "$TEST_FILE" ]; then
  echo "Test file not found: $TEST_FILE"
  exit 1
fi

mkdir -p k6-results

OUTPUT_FILE="k6-results/${TEST_NAME}-$(date +%Y%m%d-%H%M%S).json"
SUMMARY_FILE="k6-results/${TEST_NAME}-summary.json"

echo "Running $TEST_NAME test"
echo "Base URL: $BASE_URL"
echo "Output: $OUTPUT_FILE"
echo ""

k6 run --out json="$OUTPUT_FILE" --summary-export="$SUMMARY_FILE" "$TEST_FILE"

echo ""
echo "Test completed"
echo "Results: $OUTPUT_FILE"
echo "Summary: $SUMMARY_FILE"
