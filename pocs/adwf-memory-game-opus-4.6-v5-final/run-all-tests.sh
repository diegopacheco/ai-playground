#!/bin/bash
echo "=== Backend Unit Tests ==="
cd backend && cargo test 2>&1
BACKEND_RESULT=$?
echo ""
echo "=== Frontend Unit Tests ==="
cd ../frontend && bun run test 2>&1
FRONTEND_RESULT=$?
echo ""
if [ $BACKEND_RESULT -eq 0 ] && [ $FRONTEND_RESULT -eq 0 ]; then
  echo "ALL TESTS PASSED"
else
  echo "SOME TESTS FAILED"
  exit 1
fi
