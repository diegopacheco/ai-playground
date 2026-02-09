#!/bin/bash
SCRIPT_DIR="$(dirname "$0")"

echo "Running smoke test..."
k6 run "$SCRIPT_DIR/smoke-test.js"

echo "Running load test..."
k6 run "$SCRIPT_DIR/load-test.js"

echo "Running stress test..."
k6 run "$SCRIPT_DIR/stress-test.js"

echo "All k6 tests completed."
